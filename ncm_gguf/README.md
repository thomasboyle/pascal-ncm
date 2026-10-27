# code_ncm.gguf — byte-level NCM code model

Author-trained **Next-Concept Modeling** (NCM) experiment on `D:\C++\LocalAI\code_corpus.txt`
(GitHub code). Architecture from arXiv 2610.14162 / github.com/HazyResearch/ncm.

## What this file is
A single GGUF container holding the **whole** NCM system, because the two parts are
coupled and neither runs alone:

| stage | tensors | what it does |
|-------|---------|--------------|
| `tokenizer.*` (0.36M) | VQ-VAE encoder + 6 multiscale-residual codebooks + decoder | maps bytes <-> discrete codes and back |
| `generator.*` (0.86M) | coarse-to-fine transformer (n_embd 128, 4 layers) | predicts code levels 1..5 in parallel, conditioned on earlier levels |

- Levels (codes per level, coarse->fine): **[1, 4, 8, 16, 32, 64]**, codebook size **256** each.
- Byte-level vocab 258 (0-255 + eos 256 + pad 257); sequence length 64.
- Total params: **1,217,578**. File size: **2.46 MB** (matrices F16, codebooks/norms F32).

## Runtime caveat (important)
llama.cpp has **no NCM op**, so this GGUF is a faithful *container*, **not** a stock
llama.cpp-runnable LM. The generator's output heads are tied to the tokenizer
codebooks (a codebook-distance head) and generation is coarse-to-fine, then decoded
by the VQ-VAE decoder — a custom loop. Run it with the Python runtime:

```
.venv/Scripts/python.exe -W ignore run_code_ncm.py     # generates from code_artifacts/*.pt
```
The GGUF is provided for portability / inspection (verified with `llama-gguf.exe r`
and the `gguf` Python reader; tensors match the checkpoints to F16 precision).

## Provenance
- Env: `D:\C++\LocalAI\.venv` (torch 2.8.0+cu128 — the cu128 build, since cu129 dropped Pascal sm_61).
- Train: `train_code_ncm_v2.py` (VQ 8000 steps, generator 12000 steps, GPU).
  - tokenizer recon val cross-entropy = **0.004** (near-perfect byte reconstruction)
  - generator code cross-entropy 5.5 -> **~2.4**
- Export: `export_gguf.py`  |  Verify: `verify_gguf.py`

## Quality note
This is a *toy* (L=64, 1.2M params). The tokenizer reconstructs well, but the
generator is only partially trained — sampled output shows faint code-like structure
but is not coherent. Scaling up (longer L, bigger batch/model) is needed for
coherent generation.
