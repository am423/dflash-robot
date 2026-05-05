#!/usr/bin/env python3
"""Generate training traces using the GGUF target model via test_generate.

For each prompt in the prompts file:
  1. Tokenize with HF tokenizer (skip if already tokenized)
  2. Run test_generate --dump-traces=<dir>
  3. Trace files are written as trace_XXXXXX.pt (binary format)

Usage:
  python3 training/generate_traces.py --target models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
      --prompts training/data/train_prompts.jsonl --output-dir training/traces/
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path


def tokenize_prompt(prompt: str, tokenizer, out_path: str) -> int:
    """Tokenize a prompt to binary int32 file."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with open(out_path, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<i", int(tid)))
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Path to target GGUF")
    ap.add_argument("--prompts", required=True, help="JSONL prompts file")
    ap.add_argument("--output-dir", default="training/traces/")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--n-gen", type=int, default=64, help="Tokens to generate per prompt")
    ap.add_argument("--max-prompts", type=int, default=0, help="Limit prompts (0=all)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--binary", default="./build/test_generate")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = Path(args.output_dir) / "_tokenized"
    tmp_dir.mkdir(exist_ok=True)

    # Load tokenizer
    print(f"[gen] Loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    prompts.append(data.get('text', data.get('prompt', line.strip())))
                except json.JSONDecodeError:
                    prompts.append(line.strip())

    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    print(f"[gen] {len(prompts)} prompts, generating {args.n_gen} tokens each")

    # Count existing traces
    existing = len(list(Path(args.output_dir).glob("trace_*")))
    if existing > 0:
        print(f"[gen] {existing} traces already exist")

    t0 = time.time()
    success = 0
    for i, prompt in enumerate(prompts):
        # Tokenize
        tok_path = tmp_dir / f"prompt_{i:04d}.bin"
        if not tok_path.exists():
            n_tok = tokenize_prompt(prompt, tok, str(tok_path))
        else:
            with open(tok_path, "rb") as f:
                n_tok = len(f.read()) // 4

        # Run test_generate
        cmd = [
            args.binary,
            args.target,
            str(tok_path),
            str(args.n_gen),
            "/tmp/trace_gen_out.bin",
            f"--dump-traces={args.output_dir}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"  [{i:03d}] FAILED (exit {result.returncode}): {result.stderr[-200:]}")
                continue

            # Count traces generated
            for line in result.stdout.split("\n"):
                if "[trace]" in line and "saved" in line:
                    pass  # trace written

            success += 1
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                tps = (success * args.n_gen) / max(0.1, elapsed)
                print(f"  [{i+1}/{len(prompts)}] {success} traces, {elapsed:.0f}s elapsed, ~{tps:.0f} tok/s")

        except subprocess.TimeoutExpired:
            print(f"  [{i:03d}] TIMEOUT")
        except Exception as e:
            print(f"  [{i:03d}] ERROR: {e}")

    elapsed = time.time() - t0
    traces = len(list(Path(args.output_dir).glob("trace_*")))
    print(f"\n[gen] Done: {traces} traces in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Estimate total positions
    total_pos = 0
    for p in sorted(Path(args.output_dir).glob("trace_*"))[:5]:
        with open(p, "rb") as f:
            n = struct.unpack("<i", f.read(4))[0]
            total_pos += n
    if traces > 0:
        avg = total_pos / min(5, traces)
        est = int(avg * traces)
        print(f"[gen] Estimated {est} training positions (~{est // 1000}K)")


if __name__ == "__main__":
    main()
