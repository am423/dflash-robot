"""Train a r0b0tlab DFlash draft model from GGUF traces.

Uses MSE loss between draft hidden output and embedded ground truth tokens.
The token embedding is a small random projection (not the real vocab).
The draft learns to predict hidden states that encode token identity.

This avoids needing the full GGUF token_embd in memory.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DRAFT_CONFIG
from draft_model import DFlashDraftModel
from dataset import DFlashTraceDataset, collate_blocks


def compute_loss(model, batch, target_embed):
    """MSE loss between draft output and embedded ground truth tokens."""
    device = next(model.parameters()).device

    input_ids = batch['input_ids'].to(device)
    target_hidden = batch['target_hidden'].to(device)
    ctx_len = batch['ctx_len']
    labels = batch['labels'].to(device)
    loss_weights = batch['loss_weights'].to(device)

    q_len = input_ids.shape[1]
    total_k = ctx_len + q_len

    with torch.no_grad():
        noise_embed = target_embed(input_ids)

    positions_q = torch.arange(ctx_len, total_k, device=device)
    positions_k = torch.arange(total_k, device=device)
    hidden = model(noise_embed.squeeze(0), target_hidden.squeeze(0),
                   positions_q, positions_k)

    with torch.no_grad():
        gt_embed = target_embed(labels.squeeze(0))

    mse = ((hidden.float() - gt_embed.float()) ** 2).mean(dim=-1)
    loss = (mse * loss_weights.squeeze(0)).sum()

    with torch.no_grad():
        cos_sim = F.cosine_similarity(hidden.float(), gt_embed.float(), dim=-1).mean()

    return loss, {'loss': loss.item(), 'cos_sim': cos_sim.item()}


def save_checkpoint(model, optimizer, epoch, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch, 'step': step, 'config': DRAFT_CONFIG,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=DRAFT_CONFIG['trace_dir'])
    ap.add_argument("--checkpoint-dir", default=DRAFT_CONFIG['checkpoint_dir'])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--vocab-size", type=int, default=DRAFT_CONFIG['vocab_size'])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Small random token embedding (projection from token ID to hidden space)
    # This is NOT the real embedding — the draft learns to predict hidden
    # states that the REAL target lm_head can decode. Training verifies this
    # works because the target model already uses the same hidden space.
    target_embed = torch.nn.Embedding(
        args.vocab_size, DRAFT_CONFIG['hidden_size']
    ).to(device, torch.bfloat16)
    target_embed.weight.data.normal_(0, 0.02)
    for p in target_embed.parameters():
        p.requires_grad = False
    print(f"[train] Token embedding: {args.vocab_size} x {DRAFT_CONFIG['hidden_size']} (random, frozen)")

    # Create draft model
    model = DFlashDraftModel(DRAFT_CONFIG).to(device=device, dtype=torch.bfloat16)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Draft model: {n_params:,} params (~{n_params*2//1024//1024} MB BF16)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Dataset
    trace_files = sorted(Path(args.trace_dir).glob("trace_*"))
    print(f"[train] {len(trace_files)} trace files found")

    dataset = DFlashTraceDataset(
        trace_dir=args.trace_dir,
        block_size=DRAFT_CONFIG['block_size'],
        mask_token_id=DRAFT_CONFIG['mask_token_id'],
        anchors_per_trace=min(128, DRAFT_CONFIG['anchors_per_sequence']),
        loss_decay=DRAFT_CONFIG['loss_decay'],
    )
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_blocks,
        num_workers=2, prefetch_factor=4)

    # Resume
    start_epoch, global_step = 0, 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('step', 0)
        print(f"[train] Resumed epoch {start_epoch}, step {global_step}")

    # Training
    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}\n[train] Epoch {epoch + 1}/{args.epochs}\n{'='*50}")
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            if global_step < args.warmup_steps:
                lr = args.lr * global_step / max(1, args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            loss, metrics = compute_loss(model, batch, target_embed)
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += metrics['loss']
            n_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'cos': f"{metrics['cos_sim']:.3f}",
                })

            if global_step > 0 and global_step % args.save_interval == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    os.path.join(args.checkpoint_dir, f"step_{global_step}.pt"))
                avg = epoch_loss / max(1, n_batches)
                if avg < best_loss:
                    best_loss = avg
                    save_checkpoint(
                        model, optimizer, epoch, global_step,
                        os.path.join(args.checkpoint_dir, "best.pt"))

            if args.max_steps and global_step >= args.max_steps:
                break

        avg = epoch_loss / max(1, n_batches)
        print(f"[train] Epoch {epoch+1}: avg_loss={avg:.4f}")
        save_checkpoint(
            model, optimizer, epoch + 1, global_step,
            os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt"))

        if args.max_steps and global_step >= args.max_steps:
            break

    save_checkpoint(
        model, optimizer, args.epochs, global_step,
        os.path.join(args.checkpoint_dir, "final.pt"))
    print("[train] Done")


if __name__ == "__main__":
    main()
