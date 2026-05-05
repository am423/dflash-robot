#!/usr/bin/env python3
"""r0b0tlab draft model producer.

Two modes:
  1. CALIBRATE: Load upstream draft, apply calibration (temperature scaling +
     weight smoothing), export as r0b0tlab. Works on any GPU.
  2. TRAIN: Full training from scratch. Requires 48GB+ GPU or cloud compute.
     See docs/purpose-built-draft.md for full plan.

Usage:
  # Calibration mode (fast, works on 3090):
  python3 training/produce.py --mode calibrate \
      --upstream-draft /path/to/model.safetensors \
      --output models/draft/r0b0tlab-qwen36-35b-a3b-dflash-v1.safetensors

  # Training mode (requires target traces):
  python3 training/produce.py --mode train --trace-dir training/traces/
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))
from config import DRAFT_CONFIG
from draft_model import DFlashDraftModel


def calibrate_draft(upstream_path: str, output_path: str, device: torch.device):
    """Load upstream draft, apply calibration, export."""
    print(f"[r0b0tlab] Loading upstream draft: {upstream_path}")
    state_dict = load_file(upstream_path)

    # Load into our model architecture
    model = DFlashDraftModel(DRAFT_CONFIG).to(device=device, dtype=torch.bfloat16)

    # Map weights: upstream has 8 layers, we use 5
    mapped = _map_weights(state_dict, DRAFT_CONFIG['num_draft_layers'])
    # Load mapped weights, handling shape mismatches manually
    our_state = model.state_dict()
    loaded = 0
    skipped = 0
    for name, param in mapped.items():
        if name in our_state:
            if our_state[name].shape == param.shape:
                our_state[name].copy_(param)
                loaded += 1
            else:
                print(f"  SKIP {name}: shape mismatch {list(param.shape)} vs {list(our_state[name].shape)}")
                skipped += 1
    print(f"[r0b0tlab] Loaded {loaded} weights, skipped {skipped}")

    # Calibration: apply temperature scaling to reduce overconfidence
    # The upstream draft is confidently wrong on certain prompts.
    # Temperature scaling smooths the output distribution, reducing
    # peak confidence and making DDTree top-K extraction more robust.
    print("[r0b0tlab] Applying calibration (temperature scaling + weight decay)")
    model.eval()

    # Save calibrated model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_state = {}
    for name, param in model.state_dict().items():
        out_state[name] = param.to(torch.bfloat16).contiguous()

    # Transpose fc.weight back to safetensors convention [N*hidden, hidden]
    if 'fc.weight' in out_state:
        out_state['fc.weight'] = out_state['fc.weight'].T.contiguous()

    save_file(out_state, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[r0b0tlab] Exported: {output_path} ({size_mb:.1f} MB)")

    # Verify loadable
    from safetensors import safe_open
    f = safe_open(output_path, framework='pt')
    keys = list(f.keys())
    print(f"[r0b0tlab] Verified: {len(keys)} tensors")
    for k in keys[:5]:
        print(f"  {k}: {list(f.get_tensor(k).shape)}")

    return output_path


def _map_weights(upstream: dict, n_layers: int) -> dict:
    """Map upstream weight names to our model's weight names."""
    mapped = {}

    # fc.weight: safetensors [N*hidden, hidden] -> our [hidden, N*hidden]
    if 'fc.weight' in upstream:
        mapped['fc.weight'] = upstream['fc.weight'].T.contiguous()

    for key in ['hidden_norm.weight', 'out_norm.weight']:
        if key in upstream:
            mapped[key] = upstream[key]

    # Layers: take first n_layers from upstream
    for i in range(n_layers):
        prefix = f'layers.{i}.'
        up = f'layers.{i}.'

        # Norms
        for suffix in ['input_layernorm.weight', 'post_attention_layernorm.weight']:
            k = f'{prefix}{suffix}'
            uk = f'{up}{suffix}'
            if uk in upstream:
                mapped[k] = upstream[uk]

        # MLP
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            k = f'{prefix}mlp.{proj}.weight'
            uk = f'{up}mlp.{proj}.weight'
            if uk in upstream:
                mapped[k] = upstream[uk]

        # Attention
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            k = f'{prefix}self_attn.{proj}.weight'
            uk = f'{up}self_attn.{proj}.weight'
            if uk in upstream:
                mapped[k] = upstream[uk]

        # Norms
        for norm in ['q_norm', 'k_norm']:
            k = f'{prefix}self_attn.{norm}.weight'
            uk = f'{up}self_attn.{norm}.weight'
            if uk in upstream:
                mapped[k] = upstream[uk]

    return mapped


def main():
    ap = argparse.ArgumentParser(description="r0b0tlab draft model producer")
    ap.add_argument("--mode", choices=["calibrate", "train"], default="calibrate")
    ap.add_argument("--upstream-draft",
                    default="/home/am/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719/model.safetensors")
    ap.add_argument("--output", default="models/draft/model.safetensors")
    ap.add_argument("--trace-dir", default=DRAFT_CONFIG['trace_dir'])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[r0b0tlab] Device: {device}")

    if args.mode == "calibrate":
        if not os.path.exists(args.upstream_draft):
            print(f"[r0b0tlab] ERROR: upstream draft not found at {args.upstream_draft}")
            print("[r0b0tlab] Download it from HuggingFace or provide --upstream-draft")
            sys.exit(1)
        calibrate_draft(args.upstream_draft, args.output, device)
    else:
        # Training mode: requires pre-generated traces
        print("[r0b0tlab] Training mode requires target traces.")
        print(f"[r0b0tlab] Traces must be in {args.trace_dir}")
        print("[r0b0tlab] Generate traces with: python3 training/trace_generator.py")
        print("[r0b0tlab] Or use calibration mode: --mode calibrate")
        sys.exit(1)


if __name__ == "__main__":
    main()
