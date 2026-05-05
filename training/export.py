"""Export trained DFlash draft model to safetensors format.

Produces a model.safetensors file compatible with dflash-robot's
load_draft_safetensors() loader.

Output matches r0b0tlab's format exactly:
  - fc.weight: [hidden, N*hidden]
  - hidden_norm.weight: [hidden]
  - layers.{i}.input_layernorm.weight: [hidden]
  - layers.{i}.self_attn.{q,k,v,o}_proj.weight
  - layers.{i}.self_attn.{q,k}_norm.weight
  - layers.{i}.post_attention_layernorm.weight
  - layers.{i}.mlp.{gate,up,down}_proj.weight
  - out_norm.weight: [hidden]

Also saves a config.json with training metadata.
"""

import argparse
import json
import os
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from config import DRAFT_CONFIG
from draft_model import DFlashDraftModel


def export_to_safetensors(
    model: DFlashDraftModel,
    output_path: str,
    config: dict,
    dtype: torch.dtype = torch.bfloat16,
):
    """Export draft model weights to safetensors format.

    Weight naming matches r0b0tlab's convention:
      - PyTorch: fc.weight [hidden, N*hidden]
      - Safetensors: fc.weight with shape [N*hidden, hidden] (column-major for ggml)
    """
    state_dict = {}
    model_state = model.state_dict()

    # FC: PyTorch [hidden, N*hidden] — already correct for safetensors (hidden first)
    if 'fc.weight' in model_state:
        state_dict['fc.weight'] = model_state['fc.weight'].to(dtype).contiguous()
    if 'hidden_norm.weight' in model_state:
        state_dict['hidden_norm.weight'] = model_state['hidden_norm.weight'].to(dtype)
    if 'out_norm.weight' in model_state:
        state_dict['norm.weight'] = model_state['out_norm.weight'].to(dtype)

    # Layers
    n_layers = config['num_draft_layers']
    for i in range(n_layers):
        prefix = f'layers.{i}.'
        layer_prefix_pyt = f'layers.{i}.'

        # Input layernorm
        key = f'{layer_prefix_pyt}input_layernorm.weight'
        state_dict[f'{prefix}input_layernorm.weight'] = model_state[key].to(dtype)

        # Self-attention: nn.Linear.weight has shape [out, in]
        # We want safetensors [out, in] which is what PyTorch already gives us
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            key = f'{layer_prefix_pyt}self_attn.{proj}.weight'
            if key in model_state:
                state_dict[f'{prefix}self_attn.{proj}.weight'] = model_state[key].to(dtype)

        # Attention norms
        for norm in ['q_norm', 'k_norm']:
            key = f'{layer_prefix_pyt}self_attn.{norm}.weight'
            if key in model_state:
                state_dict[f'{prefix}self_attn.{norm}.weight'] = model_state[key].to(dtype)

        # Post-attention layernorm
        key = f'{layer_prefix_pyt}post_attention_layernorm.weight'
        state_dict[f'{prefix}post_attention_layernorm.weight'] = model_state[key].to(dtype)

        # MLP
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            key = f'{layer_prefix_pyt}mlp.{proj}.weight'
            if key in model_state:
                state_dict[f'{prefix}mlp.{proj}.weight'] = model_state[key].to(dtype)

    # Save safetensors
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(state_dict, output_path)
    print(f"[export] Saved {len(state_dict)} tensors to {output_path}")

    # Print size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[export] File size: {size_mb:.1f} MB")

    # Validate shapes match expected
    hidden = config['hidden_size']
    n_feat = config['n_target_features']
    n_head = config['num_attention_heads']
    n_kv = config['num_key_value_heads']
    head_dim = config['head_dim']
    intermediate = config['intermediate_size']

    expected = {
        'fc.weight': [hidden, n_feat * hidden],
        'hidden_norm.weight': [hidden],
        'norm.weight': [hidden],
    }
    for i in range(n_layers):
        p = f'layers.{i}'
        expected[f'{p}.input_layernorm.weight'] = [hidden]
        expected[f'{p}.self_attn.q_proj.weight'] = [n_head * head_dim, hidden]
        expected[f'{p}.self_attn.k_proj.weight'] = [n_kv * head_dim, hidden]
        expected[f'{p}.self_attn.v_proj.weight'] = [n_kv * head_dim, hidden]
        expected[f'{p}.self_attn.o_proj.weight'] = [hidden, n_head * head_dim]
        expected[f'{p}.self_attn.q_norm.weight'] = [head_dim]
        expected[f'{p}.self_attn.k_norm.weight'] = [head_dim]
        expected[f'{p}.post_attention_layernorm.weight'] = [hidden]
        expected[f'{p}.mlp.gate_proj.weight'] = [intermediate, hidden]
        expected[f'{p}.mlp.up_proj.weight'] = [intermediate, hidden]
        expected[f'{p}.mlp.down_proj.weight'] = [hidden, intermediate]

    for name, expected_shape in expected.items():
        if name in state_dict:
            actual = list(state_dict[name].shape)
            if actual != expected_shape:
                print(f"[export] WARNING: {name} shape {actual} != expected {expected_shape}")
        else:
            print(f"[export] WARNING: {name} not found in state_dict")
            if i < n_layers: break  # only warn once per layer

    return state_dict


def save_config(output_dir: str, config: dict, total_params: int):
    """Save model config as JSON for reference."""
    metadata = {
        'hidden_size': config['hidden_size'],
        'num_hidden_layers': config['num_draft_layers'],
        'num_attention_heads': config['num_attention_heads'],
        'num_key_value_heads': config['num_key_value_heads'],
        'head_dim': config['head_dim'],
        'intermediate_size': config['intermediate_size'],
        'n_target_features': config['n_target_features'],
        'block_size': config['block_size'],
        'mask_token_id': config['mask_token_id'],
        'loss_decay': config['loss_decay'],
        'rms_norm_eps': config['rms_norm_eps'],
        'rope_theta': config['rope_theta'],
        'total_params': total_params,
        'target_model_id': config['target_model_id'],
        'capture_layer_ids': config['capture_layer_ids'],
    }
    path = os.path.join(output_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[export] Config saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to training checkpoint (.pt)")
    ap.add_argument("--output-dir", default="models/draft/")
    ap.add_argument("--output-name", default="model.safetensors")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] Loading checkpoint: {args.checkpoint}")

    # Load model from checkpoint
    model = DFlashDraftModel(DRAFT_CONFIG).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[export] Model: {total_params:,} params")
    print(f"[export] Epoch: {ckpt.get('epoch', '?')}, Step: {ckpt.get('step', '?')}")

    # Export safetensors
    output_path = os.path.join(args.output_dir, args.output_name)
    export_to_safetensors(model, output_path, DRAFT_CONFIG, dtype)

    # Save config
    save_config(args.output_dir, DRAFT_CONFIG, total_params)

    # Save as a copy named for the specific model
    model_name = f"r0b0tlab-qwen36-35b-a3b-dflash-v1.safetensors"
    copy_path = os.path.join(args.output_dir, model_name)
    if copy_path != output_path:
        import shutil
        shutil.copy2(output_path, copy_path)
        print(f"[export] Copied to {copy_path}")

    print(f"[export] Done. Draft model ready at {output_path}")


if __name__ == "__main__":
    main()
