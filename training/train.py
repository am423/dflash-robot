"""Train a DFlash draft model from target model traces.

Implements the DFlash training algorithm (Section 4.2 of paper):
  - Random anchor sampling
  - Position-decayed cross-entropy loss
  - Calibration-aware loss (our addition)
  - Gradient accumulation for larger effective batch size
  - Checkpointing and resumption
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DRAFT_CONFIG
from draft_model import DFlashDraftModel
from dataset import DFlashTraceDataset, collate_blocks


def compute_loss(
    model: DFlashDraftModel,
    batch: dict,
    target_embed: torch.nn.Embedding,
    target_lm_head: torch.nn.Linear,
    calibration_lambda: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute DFlash training loss for one block.

    Returns (loss, metrics_dict).
    """
    device = next(model.parameters()).device
    block_size = DRAFT_CONFIG['block_size']

    input_ids = batch['input_ids'].to(device)          # [1, block_size]
    target_hidden = batch['target_hidden'].to(device)  # [1, ctx_len+q_len, feat_dim]
    ctx_len = batch['ctx_len']
    labels = batch['labels'].to(device)                 # [1, block_size]
    loss_weights = batch['loss_weights'].to(device)     # [1, block_size]

    batch_size, q_len = input_ids.shape  # q_len = block_size = 16
    total_k = ctx_len + q_len

    # Step 1: Embed noise block using TARGET's frozen embedding table
    with torch.no_grad():
        noise_embed = target_embed(input_ids)  # [1, block_size, hidden_size]

    # Step 2: Positions
    positions_q = torch.arange(ctx_len, total_k, device=device)  # [q_len]
    positions_k = torch.arange(total_k, device=device)           # [ctx_len + q_len]

    # Step 3: Draft forward
    hidden = model(
        noise_embed.squeeze(0),                     # [q_len, hidden_size]
        target_hidden.squeeze(0),                    # [ctx_len+q_len, feat_dim]
        positions_q,                                 # [q_len]
        positions_k,                                 # [total_k]
    )  # [q_len, hidden_size]

    # Step 4: Project through TARGET's frozen lm_head to get logits
    # NOTE: not wrapped in no_grad — lm_head weights are frozen but
    # logits need gradients to flow back to the draft model
    logits = target_lm_head(hidden)  # [q_len, vocab]

    # Step 5: Position-decayed cross-entropy loss
    ce_loss = F.cross_entropy(
        logits,
        labels.squeeze(0),
        reduction='none'  # per-position
    )  # [q_len]
    weighted_ce = (ce_loss * loss_weights.squeeze(0)).sum()

    # Step 6: Calibration-aware loss (our addition)
    # Penalize when draft confidence doesn't match correctness
    draft_probs = F.softmax(logits.float(), dim=-1)
    draft_confidence = draft_probs.max(dim=-1).values   # [q_len]
    draft_preds = logits.argmax(dim=-1)                  # [q_len]
    correct_mask = (draft_preds == labels.squeeze(0)).float()  # [q_len]

    cal_loss = ((draft_confidence - correct_mask) ** 2).mean()

    total_loss = weighted_ce + calibration_lambda * cal_loss

    # Metrics
    with torch.no_grad():
        accuracy = correct_mask.mean().item()
        avg_confidence = draft_confidence.mean().item()
        total_correct = correct_mask.sum().item()

    metrics = {
        'ce_loss': weighted_ce.item(),
        'cal_loss': cal_loss.item(),
        'total_loss': total_loss.item(),
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'correct': total_correct,
        'block_size': q_len,
    }

    return total_loss, metrics


def load_target_embedding(model_id: str, device: torch.device):
    """Load target model's embedding table and lm_head (frozen)."""
    from transformers import AutoModelForCausalLM
    from transformers import BitsAndBytesConfig

    print(f"[train] Loading target embedding from {model_id}")

    # Load in 4-bit (just need embeddings + lm_head, rest can be offloaded)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    # Move to target device if not already there
    if embed.weight.device != device:
        embed = embed.to(device)
    if lm_head.weight.device != device:
        lm_head = lm_head.to(device)

    return embed, lm_head


def save_checkpoint(model, optimizer, epoch, step, loss_history, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss_history': loss_history,
        'config': DRAFT_CONFIG,
    }, path)
    print(f"[train] Checkpoint saved to {path}")


def load_checkpoint(path: str, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt['step'], ckpt.get('loss_history', [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=DRAFT_CONFIG['trace_dir'])
    ap.add_argument("--checkpoint-dir", default=DRAFT_CONFIG['checkpoint_dir'])
    ap.add_argument("--model-id", default=DRAFT_CONFIG['target_model_id'])
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--epochs", type=int, default=DRAFT_CONFIG['max_epochs'])
    ap.add_argument("--lr", type=float, default=DRAFT_CONFIG['learning_rate'])
    ap.add_argument("--batch-size", type=int, default=DRAFT_CONFIG['batch_size'])
    ap.add_argument("--grad-accum", type=int, default=DRAFT_CONFIG['grad_accum_steps'])
    ap.add_argument("--warmup-ratio", type=float, default=DRAFT_CONFIG['warmup_ratio'])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--save-interval", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Config: {json.dumps(DRAFT_CONFIG, indent=2)}")

    # Load frozen target embeddings
    target_embed, target_lm_head = load_target_embedding(args.model_id, device)
    for p in target_embed.parameters():
        p.requires_grad = False
    for p in target_lm_head.parameters():
        p.requires_grad = False

    # Create draft model
    model = DFlashDraftModel(DRAFT_CONFIG).to(device)
    model = model.to(dtype=torch.bfloat16)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Draft model: {n_params:,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=DRAFT_CONFIG['weight_decay'],
    )

    # Dataset
    dataset = DFlashTraceDataset(
        trace_dir=args.trace_dir,
        block_size=DRAFT_CONFIG['block_size'],
        mask_token_id=DRAFT_CONFIG['mask_token_id'],
        anchors_per_trace=DRAFT_CONFIG['anchors_per_sequence'],
        loss_decay=DRAFT_CONFIG['loss_decay'],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_blocks,
        num_workers=2,
        prefetch_factor=4,
    )

    # Calculate total steps for cosine schedule
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[train] Estimated {total_steps} steps, {warmup_steps} warmup steps")

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps)

    # Resume
    start_epoch = 0
    global_step = 0
    loss_history = []
    if args.resume:
        start_epoch, global_step, loss_history = load_checkpoint(
            args.resume, model, optimizer, device)
        print(f"[train] Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        print(f"\n[train] Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_metrics = {'ce': 0.0, 'cal': 0.0, 'acc': 0.0, 'conf': 0.0}
        epoch_batches = 0

        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Warmup LR
            if global_step < warmup_steps:
                lr_scale = global_step / max(1, warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr * lr_scale

            # Forward + loss
            loss, metrics = compute_loss(
                model, batch, target_embed, target_lm_head,
                calibration_lambda=DRAFT_CONFIG['calibration_lambda'])
            loss = loss / args.grad_accum
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), DRAFT_CONFIG['grad_clip'])
                optimizer.step()

                # Cosine schedule after warmup
                if global_step >= warmup_steps:
                    scheduler.step()

                optimizer.zero_grad()
                global_step += 1

            # Track metrics
            epoch_loss += metrics['total_loss']
            epoch_metrics['ce'] += metrics['ce_loss']
            epoch_metrics['cal'] += metrics['cal_loss']
            epoch_metrics['acc'] += metrics['accuracy']
            epoch_metrics['conf'] += metrics['avg_confidence']
            epoch_batches += 1

            # Progress bar
            if (batch_idx + 1) % args.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.3f}",
                    'acc': f"{metrics['accuracy']:.2f}",
                    'conf': f"{metrics['avg_confidence']:.2f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                })

            # Checkpoint
            if global_step > 0 and global_step % args.save_interval == 0:
                avg_loss = epoch_loss / max(1, epoch_batches)
                path = os.path.join(args.checkpoint_dir, f"step_{global_step}.pt")
                save_checkpoint(model, optimizer, epoch, global_step, loss_history, path)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(args.checkpoint_dir, "best.pt")
                    save_checkpoint(model, optimizer, epoch, global_step, loss_history, best_path)

            if args.max_steps and global_step >= args.max_steps:
                break

        # End of epoch
        avg_loss = epoch_loss / max(1, epoch_batches)
        avg_ce = epoch_metrics['ce'] / max(1, epoch_batches)
        avg_cal = epoch_metrics['cal'] / max(1, epoch_batches)
        avg_acc = epoch_metrics['acc'] / max(1, epoch_batches)
        avg_conf = epoch_metrics['conf'] / max(1, epoch_batches)

        print(f"[train] Epoch {epoch+1} avg: loss={avg_loss:.3f} "
              f"ce={avg_ce:.3f} cal={avg_cal:.3f} "
              f"acc={avg_acc:.3f} conf={avg_conf:.3f}")

        loss_history.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_ce': avg_ce,
            'avg_cal': avg_cal,
            'avg_acc': avg_acc,
            'avg_conf': avg_conf,
        })

        # Save epoch checkpoint
        path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch + 1, global_step, loss_history, path)

        if args.max_steps and global_step >= args.max_steps:
            break

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, args.epochs, global_step, loss_history, final_path)
    print(f"[train] Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
