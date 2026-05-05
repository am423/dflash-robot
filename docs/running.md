# Running dflash-robot

## One-shot Generation

```bash
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf \
DFLASH_DRAFT=models/draft/model.safetensors \
.venv/bin/python scripts/run.py --prompt "Write a function that sorts a list." --n-gen 256
```

Options:
- `--prompt TEXT` — inline prompt
- `--n-gen N` — max tokens to generate (default 256)
- `--raw` — skip chat template, use raw text
- `--budget N` — DDTree budget (default 22)
- `--kv-q4` — Q4_0 KV cache for long context
- `--kv-tq3` — TQ3_0 KV cache (3.5 bpv, near-lossless)
- `--fa-window N` — sliding FA window (default 2048)
- `--max-ctx N` — override max KV context

## Multi-turn Chat

```bash
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf \
DFLASH_DRAFT=models/draft/model.safetensors \
.venv/bin/python examples/chat.py
```

## Benchmarking

```bash
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf \
DFLASH_DRAFT=models/draft/model.safetensors \
DFLASH_BIN=build/test_dflash \
DFLASH_BIN_AR=build/test_generate \
DFLASH_TOKENIZER=Qwen/Qwen3.5-27B \
.venv/bin/python scripts/bench_llm.py
```

Results saved to /tmp/dflash_bench/bench_llm_results.json

## Inspecting Models

```bash
./build/dflash_inspect --target model.gguf --json
./build/dflash_inspect --target model.gguf --draft model.safetensors --json
```
