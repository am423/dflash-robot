#!/usr/bin/env python3
"""Oracle: compare dflash-robot target AR output vs llama.cpp greedy output.

Usage:
    python scripts/oracle_target.py <model.gguf> [--prompts 5] [--tokens 32] [--verbose]

Requires:
    - test_generate binary built (dflash-robot target AR)
    - llama-cli binary (llama.cpp reference)
    - HF tokenizer matching the model

Exit 0 if all prompts match, 1 if any diverge.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time

# Paths relative to dflash-robot repo root
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_GENERATE = os.path.join(REPO, "build", "test_generate")
LLAMA_CLI = os.path.join(REPO, "deps", "llama.cpp", "build", "bin", "llama-cli")

PROMPTS = [
    "def fibonacci(n):",
    "Solve: 17 * 23 =",
    "The capital of France is",
    "Here is a Python function to reverse a linked list:\n",
    "In quantum mechanics, the Schr\u00f6dinger equation describes",
    "The three laws of thermodynamics are:",
    "A haiku about machine learning:",
    "Write a function to check if a number is prime:\n",
]


def tokenize(prompt, model_name, add_bos=True):
    """Tokenize a prompt using transformers."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=add_bos)
    return ids


def write_token_file(path, token_ids):
    """Write int32 token IDs to binary file."""
    with open(path, "wb") as f:
        for t in token_ids:
            f.write(struct.pack("<i", int(t)))


def read_token_file(path):
    """Read int32 token IDs from binary file."""
    tokens = []
    with open(path, "rb") as f:
        while True:
            data = f.read(4)
            if not data:
                break
            tokens.append(struct.unpack("<i", data)[0])
    return tokens


def run_dflash_target(model_path, prompt_ids, n_gen, temp=0):
    """Run test_generate and return generated token IDs."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fin, \
         tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fout:
        write_token_file(fin.name, prompt_ids)

        env = os.environ.copy()
        if temp == 0:
            env["DFLASH_TEMP"] = "0"

        cmd = [TEST_GENERATE, model_path, fin.name, str(n_gen), fout.name]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env
        )
        if proc.returncode != 0:
            print(f"  test_generate FAILED (exit {proc.returncode})")
            print(f"  stderr: {proc.stderr[-500:]}")
            os.unlink(fin.name)
            os.unlink(fout.name)
            return None

        # Read generated tokens
        gen_tokens = read_token_file(fout.name)
        os.unlink(fin.name)
        os.unlink(fout.name)
        return gen_tokens


def run_llama_cpp(model_path, prompt, n_gen, temp=0):
    """Run llama-cli with a text prompt and return generated token IDs."""
    # llama-cli takes text prompt with -p
    # Use --temp 0 for greedy, --seed 42 for reproducibility
    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_gen),
        "--temp", "0",
        "--seed", "42",
        "-ngl", "99",
        "--no-display-prompt",
        "--log-disable",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )

    if proc.returncode != 0:
        print(f"  llama-cli FAILED (exit {proc.returncode})")
        print(f"  stderr: {proc.stderr[-500:]}")
        return None

    # llama-cli output is text. We need token IDs.
    # Strategy: tokenize the output using the same tokenizer and compare.
    # But that loses token-level fidelity.
    # Better approach: use llama-tokenize or parse the output differently.
    # For now, we compare text output character-by-character.
    return proc.stdout.strip()


def find_divergence(a, b):
    """Find first position where token sequences differ."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1  # identical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to GGUF model")
    ap.add_argument("--prompts", type=int, default=5, help="Number of prompts to test")
    ap.add_argument("--tokens", type=int, default=32, help="Tokens to generate")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--hf-model", default="Qwen/Qwen3.6-35B-A3B",
                    help="HF model name for tokenizer")
    args = ap.parse_args()

    if not os.path.exists(TEST_GENERATE):
        print(f"ERROR: test_generate not found at {TEST_GENERATE}")
        print("Build dflash-robot first: cd build && cmake --build . -j$(nproc)")
        sys.exit(1)

    if not os.path.exists(LLAMA_CLI):
        print(f"ERROR: llama-cli not found at {LLAMA_CLI}")
        print("Build llama.cpp first: cd deps/llama.cpp/build && cmake --build . -j$(nproc) --target llama-cli")
        sys.exit(1)

    # Detect architecture from GGUF to pick correct tokenizer
    import gguf
    reader = gguf.GGUFReader(args.model)
    arch = None
    for key in reader.fields:
        if key == "general.architecture":
            arch = str(reader.fields[key].parts[reader.fields[key].data[0]])
            break

    # Map architecture to HF model name for tokenizer
    ARCH_TO_HF = {
        "qwen35": "Qwen/Qwen3.5-27B",
        "qwen35moe": "Qwen/Qwen3.6-35B-A3B",
    }
    hf_model = ARCH_TO_HF.get(arch, args.hf_model)
    print(f"Architecture: {arch}, tokenizer: {hf_model}")

    prompts = PROMPTS[: args.prompts]
    passed = 0
    failed = 0

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:60]}...")

        # Tokenize for dflash-robot (needs int32 binary)
        try:
            token_ids = tokenize(prompt, hf_model, add_bos=True)
        except Exception as e:
            print(f"  SKIP: tokenizer error: {e}")
            continue

        print(f"  Tokens: {len(token_ids)} prompt, generating {args.tokens}")

        # Run dflash-robot
        t0 = time.time()
        dflash_tokens = run_dflash_target(args.model, token_ids, args.tokens)
        t1 = time.time()
        if dflash_tokens is None:
            failed += 1
            continue

        # Detokenize output for comparison
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        dflash_text = tok.decode(dflash_tokens, skip_special_tokens=True)

        print(f"  dflash-robot: {t1-t0:.1f}s → {len(dflash_tokens)} tokens")
        if args.verbose:
            print(f"    text: {repr(dflash_text[:100])}")

        # Run llama.cpp (text-in, text-out comparison)
        t0 = time.time()
        llama_text = run_llama_cpp(args.model, prompt, args.tokens)
        t1 = time.time()
        if llama_text is None:
            failed += 1
            continue

        print(f"  llama.cpp:    {t1-t0:.1f}s → text")

        # Compare text output
        if dflash_text.strip() == llama_text.strip():
            print(f"  PASS — exact text match")
            passed += 1
        else:
            # Show diff
            dflash_short = dflash_text[:100].replace('\n', '\\n')
            llama_short = llama_text[:100].replace('\n', '\\n')
            print(f"  FAIL — text mismatch")
            print(f"    dflash: {repr(dflash_short)}")
            print(f"    llama:  {repr(llama_short)}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed == 0:
        print("ALL PASS — dflash-robot target matches llama.cpp oracle")
    else:
        print("FAILURES DETECTED — architecture bugs remain")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
