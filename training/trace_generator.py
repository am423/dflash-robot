"""Generate training traces by running the target model and capturing hidden states.

For each prompt:
  1. Run target model with output_hidden_states=True
  2. For each response position, record:
     - Concatenated hidden states from capture layers
     - Ground truth next token
     - Target logits (for distillation loss)
  3. Save as compressed .pt file

Supports checkpointing so interrupted runs can resume.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import DRAFT_CONFIG


def load_target_model(model_id: str, device_map: str = "auto"):
    """Load target model in 4-bit for trace generation."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def extract_features(
    hidden_states: tuple,
    capture_layer_ids: list[int],
    response_start: int,
    response_end: int,
) -> torch.Tensor:
    """Extract concatenated hidden states at capture layers for response positions.

    hidden_states: tuple of tensors [batch, seq_len, hidden_size]
    Returns: [response_len, n_capture * hidden_size] bf16
    """
    feats = []
    for layer_id in capture_layer_ids:
        # layer_id is 0-indexed in hidden_states tuple (0 = embedding, 1..L = layers)
        # We want the OUTPUT of layer layer_id (after attention + FFN)
        # hidden_states[0] = embeddings, hidden_states[1] = output of layer 0
        hs = hidden_states[layer_id + 1]  # +1 because [0] is embedding
        feats.append(hs[0, response_start:response_end])  # batch=0, response slice
    return torch.cat(feats, dim=-1)  # [response_len, n_capture * hidden_size]


def generate_trace(
    model,
    tokenizer,
    prompt: str,
    capture_layer_ids: list[int],
    max_new_tokens: int = 256,
) -> Optional[dict]:
    """Generate one training trace from a prompt.

    Returns dict with:
      - prompt_tokens: [] i64  (prompt token IDs)
      - hidden_features: [response_len, n_capture*hidden] bf16
      - next_tokens: [response_len] i64  (ground truth)
      - target_logits: [response_len, vocab] bf16  (for distillation)
    """
    with torch.no_grad():
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        # Generate with hidden states
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for training traces
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Process generated sequence
        generated_ids = outputs.sequences[0]
        response_start = prompt_len
        response_end = generated_ids.shape[0] - prompt_len  # exclude last token
        if response_end < 4:
            return None  # too short

        # Extract hidden states at each generation step
        # outputs.hidden_states is a tuple of tuples:
        #   outputs.hidden_states[step][layer] — step=0..N, layer=0..L+1
        n_steps = len(outputs.hidden_states)
        hidden_size = outputs.hidden_states[0][0].shape[-1]
        n_capture = len(capture_layer_ids)

        hidden_features = torch.zeros(
            n_steps, n_capture * hidden_size,
            dtype=torch.bfloat16, device="cpu")

        for step in range(n_steps):
            feats = []
            for layer_id in capture_layer_ids:
                hs = outputs.hidden_states[step][layer_id + 1][0]
                feats.append(hs)  # [hidden_size]
            hidden_features[step] = torch.cat(feats, dim=0).cpu()

        # Ground truth: tokens[response_start:response_start+n_steps]
        next_tokens = generated_ids[response_start : response_start + n_steps].cpu()

        # Target logits from generation (we don't get logits from generate() by default)
        # Option A: run a separate forward pass for logits
        # Option B: store -1 and compute during training from logit head
        # For now, store -1 placeholder (distillation uses model forward)
        target_logits = torch.zeros(n_steps, 1, dtype=torch.bfloat16)  # placeholder

        return {
            'prompt_tokens': inputs.input_ids[0].cpu(),
            'hidden_features': hidden_features,   # [n_steps, n_capture*hidden] bf16
            'next_tokens': next_tokens,           # [n_steps]
            'response_len': n_steps,
        }


def save_trace(trace: dict, path: str):
    """Save trace as compressed .pt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(trace, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True,
                    help="Path to prompts file (one JSONL per line: {text: ...})")
    ap.add_argument("--output-dir", default=DRAFT_CONFIG['trace_dir'])
    ap.add_argument("--model-id", default=DRAFT_CONFIG['target_model_id'])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="Limit number of prompts (for testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip prompts that already have trace files")
    args = ap.parse_args()

    capture_ids = DRAFT_CONFIG['capture_layer_ids']

    print(f"[trace] Loading target model: {args.model_id}")
    model, tokenizer = load_target_model(args.model_id)

    # Load prompts
    print(f"[trace] Loading prompts from {args.prompts}")
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    prompts.append(data.get('text', data.get('prompt', line.strip())))
                except json.JSONDecodeError:
                    prompts.append(line.strip())

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    print(f"[trace] Processing {len(prompts)} prompts")

    output_dir = Path(args.output_dir)
    success = 0
    skipped = 0
    failed = 0

    for i, prompt in enumerate(tqdm(prompts, desc="traces")):
        trace_path = output_dir / f"trace_{i:06d}.pt"

        if args.resume and trace_path.exists():
            skipped += 1
            continue

        try:
            trace = generate_trace(
                model, tokenizer, prompt, capture_ids, args.max_new_tokens)
            if trace is None:
                failed += 1
                continue
            save_trace(trace, str(trace_path))
            success += 1
        except Exception as e:
            tqdm.write(f"[trace] error on prompt {i}: {e}")
            failed += 1
            torch.cuda.empty_cache()

    print(f"[trace] Done: {success} succeeded, {skipped} skipped, {failed} failed")
    print(f"[trace] Trace files in {output_dir}")


if __name__ == "__main__":
    main()
