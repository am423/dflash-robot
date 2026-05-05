"""End-to-end draft model training pipeline for r0b0tlab.

Usage:
    python3 training/pipeline.py --prompts training/data/train_prompts.jsonl

Steps:
    1. Verify target model is available (HF cache or force download)
    2. Generate training traces (target model + hidden states)
    3. Train draft model (6 epochs, DFlash loss + calibration)
    4. Export to safetensors
    5. Validate with test_dflash
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

from config import DRAFT_CONFIG


def step(msg: str):
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    print(f"{'='*60}")


def verify_model():
    """Verify HF model is fully cached."""
    from transformers import AutoModelForCausalLM

    step("Verifying target model availability")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            DRAFT_CONFIG['target_model_id'],
            trust_remote_code=True,
            local_files_only=True,
        )
        n_layer = len(model.model.layers)
        hidden = model.config.hidden_size
        del model
        print(f"  Model cached: {n_layer} layers, hidden={hidden}")
        return True
    except Exception as e:
        print(f"  Model not ready: {e}")
        return False


def generate_traces(prompts_file: str, trace_dir: str):
    """Generate training traces from target model."""
    from trace_generator import load_target_model, generate_trace, save_trace

    step("Generating training traces")

    # Load prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    prompts.append(data.get('text', data.get('prompt', line.strip())))
                except json.JSONDecodeError:
                    prompts.append(line.strip())

    print(f"  {len(prompts)} prompts")

    # Load model
    print("  Loading target model (4-bit)...")
    model, tokenizer = load_target_model(DRAFT_CONFIG['target_model_id'])

    capture_ids = DRAFT_CONFIG['capture_layer_ids']
    os.makedirs(trace_dir, exist_ok=True)

    success = 0
    for i, prompt in enumerate(prompts):
        trace_path = os.path.join(trace_dir, f"trace_{i:06d}.pt")
        if os.path.exists(trace_path):
            success += 1
            continue

        try:
            trace = generate_trace(model, tokenizer, prompt, capture_ids,
                                   max_new_tokens=128)
            if trace:
                save_trace(trace, trace_path)
                success += 1
        except Exception as e:
            print(f"  Error on prompt {i}: {e}")
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} traces generated ({success} ok)")

    del model
    torch.cuda.empty_cache()
    print(f"  Generated {success} traces in {trace_dir}")
    return success


def train_model(trace_dir: str, checkpoint_dir: str):
    """Train draft model on generated traces."""
    step("Training draft model")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Use train.py as subprocess for clean state
    cmd = [
        sys.executable, "training/train.py",
        "--trace-dir", trace_dir,
        "--checkpoint-dir", checkpoint_dir,
        "--epochs", "3",  # Start with 3 for quick iteration
        "--lr", str(DRAFT_CONFIG['learning_rate']),
        "--log-interval", "50",
        "--save-interval", "500",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))


def export_model(checkpoint_dir: str, output_dir: str):
    """Export trained model to safetensors."""
    step("Exporting draft model")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    best_ckpt = os.path.join(checkpoint_dir, "best.pt")
    final_ckpt = os.path.join(checkpoint_dir, "final.pt")
    epoch_ckpt = os.path.join(checkpoint_dir, "epoch_3.pt")

    ckpt = None
    for candidate in [best_ckpt, epoch_ckpt, final_ckpt]:
        if os.path.exists(candidate):
            ckpt = candidate
            break

    if not ckpt:
        print("  No checkpoint found!")
        return False

    print(f"  Using checkpoint: {ckpt}")

    cmd = [
        sys.executable, "training/export.py",
        "--checkpoint", ckpt,
        "--output-dir", output_dir,
        "--output-name", "model.safetensors",
    ]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))

    # Also save as named version
    named = os.path.join(output_dir, DRAFT_CONFIG['export_path'].split('/')[-1])
    if os.path.exists(os.path.join(output_dir, "model.safetensors")):
        import shutil
        shutil.copy2(os.path.join(output_dir, "model.safetensors"), named)
        print(f"  Exported: {named}")

    return True


def validate_draft(draft_path: str, target_gguf: str):
    """Quick validation with test_dflash."""
    step("Validating draft model")

    # Tokenize a test prompt
    prompt_bin = "/tmp/dflash_validate_prompt.bin"
    test_prompt = "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    "

    import struct
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        DRAFT_CONFIG['target_model_id'], trust_remote_code=True)
    ids = tok.encode(test_prompt, add_special_tokens=False)
    with open(prompt_bin, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<i", int(tid)))

    # Run test_dflash
    cmd = [
        "./build/test_dflash",
        target_gguf,
        draft_path,
        prompt_bin,
        "64",
        "/tmp/dflash_validate_out.bin",
        "--fast-rollback",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=os.path.dirname(os.path.dirname(__file__)))

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return False

    # Parse output
    for line in result.stdout.split("\n"):
        if "tok/s" in line and "generated" in line:
            print(f"  {line.strip()}")

    return True


def main():
    ap = argparse.ArgumentParser(description="r0b0tlab draft model training pipeline")
    ap.add_argument("--prompts", default="training/data/train_prompts.jsonl")
    ap.add_argument("--trace-dir", default=DRAFT_CONFIG['trace_dir'])
    ap.add_argument("--checkpoint-dir", default=DRAFT_CONFIG['checkpoint_dir'])
    ap.add_argument("--output-dir", default="models/draft/")
    ap.add_argument("--target-gguf", default=DRAFT_CONFIG['target_gguf_path'])
    ap.add_argument("--skip-traces", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    if not args.skip_traces:
        if not verify_model():
            print("\nModel not fully cached. Download with:")
            print(f"  from huggingface_hub import snapshot_download")
            print(f"  snapshot_download('{DRAFT_CONFIG['target_model_id']}')")
            sys.exit(1)
        generate_traces(args.prompts, args.trace_dir)

    if not args.skip_train:
        train_model(args.trace_dir, args.checkpoint_dir)

    if not args.skip_export:
        export_model(args.checkpoint_dir, args.output_dir)

    if not args.skip_validate:
        draft_path = os.path.join(args.output_dir, "model.safetensors")
        if os.path.exists(draft_path):
            validate_draft(draft_path, args.target_gguf)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
