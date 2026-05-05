# Upstream Sources

This project began as a standalone import of Luce DFlash from:
https://github.com/Luce-Org/lucebox-hub/tree/main/dflash

Imported commit:
82b99b0bb68b3aa4a7bcaf9ca94270250d92c9b9

Luce DFlash is the first GGUF port of DFlash speculative decoding,
enabling consumer GPU inference (RTX 3090) of Qwen3.5/3.6 27B models.

DFlash algorithm reference:
https://arxiv.org/html/2602.06036v1

Original z-lab DFlash implementation:
https://github.com/z-lab/DFlash
Inspected commit:
44947fbf71114e241c96de194f4b382b5dd330d0

This repository preserves upstream license notices. dflash-robot adds
a GGUF-native adapter roadmap, compatibility tooling, benchmarking,
and model-generalization work on top of Luce DFlash.

## Upstream Submodules

- llama.cpp (Luce fork, luce-dflash branch):
  https://github.com/Luce-Org/llama.cpp-dflash-ggml.git
  Commit: ce3919b4afaa91e8bd0a02eac32c82d2dd8a4de0

- Block-Sparse-Attention (BSA):
  https://github.com/mit-han-lab/Block-Sparse-Attention.git
  Commit: a75b4ac483166189a45290783cb0a18af5ff0ea5

- CUTLASS (via BSA submodule):
  Commit: 49d6c39e4dc0303442cda3bb758b3925d4399c49
