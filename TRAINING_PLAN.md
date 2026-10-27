# Training plan

8-hour run for a local NCM coding agent on the GTX 1070 Ti. The model writes one 256-byte action. Intelligence is the prefix’s right edge deciding that action: continue the file, or emit a short plan and one tool call whose arguments are copied from the prefix.

This run has not started. `train_fast.py` only measured step speed.

## Hardware

- GPU: NVIDIA GTX 1070 Ti, 8192 MiB, compute capability 6.1. No tensor cores.
- PyTorch: `D:\C++\LocalAI\.venv`, 2.8.0+cu128. cu129 dropped Pascal.
- FP32 math. FP16 is slower on this GPU, and `torch.compile` cannot run: Triton needs compute capability 7.0.
- No bitsandbytes. The Windows wheel has no Pascal 8-bit Adam kernel.
- No CPU offload. The 1B run kept Adam in system RAM and moved weights every step, which is why it sat near 130 code tokens/s.

## Locked model

Measured fastest point that still finishes a useful amount of data:

| Setting | Value |
|---|---|
| Generator | 87.4M parameters, width 768, 12 layers, 12 heads |
| Precision | FP32 weights, gradients, and Adam |
| Optimizer | fused AdamW, weight decay 0.01, grad clip 1.0 |
| Batch | 8 |
| Dropout | 0 |
| Attention | block diagonal |
| Heads | tied to the frozen codebooks |
| RoPE base | 100000, QK norm on |
| Coarse level | `predict_level0` on, `query_level0` off |

`predict_level0` reads the last 4 prefix positions, so `prefix_len` must be at least 4. The prefix is raw bytes, left-padded, so the task sits on that right edge.

Speed on the current tokenizer, prefix length 1, batch 8: **3765 code tokens/s**, about **1.05 s/step**, **4.7 GB**. Batch 12 was 3718 tokens/s at 6.2 GB. Batch 16 spilled and fell to 2382 tokens/s. A 205M model topped out near 1900 tokens/s. Stay at 87M and batch 8.

The prefix-length fit below can slow this. Keep a prefix only if the step stays under about 6.5 GB and at least about 3000 code tokens/s.

## Tokenizer

Use the frozen tokenizer at `code_artifacts_scaled/vqvae.pt`.

- Levels, coarse to fine: `[4, 16, 32, 64, 128, 256]`
- Codebook size 1024, embed 512
- Byte length 256, EOS 256, pad 257
- Reconstruction validation cross-entropy 0.0012

Do not retrain it. A 512-byte tokenizer would lengthen every generator step. A short think plus one tool call fits in 256 bytes. Encode targets to code indices once, before the training loop, and keep the tokenizer frozen. The generator loss is cross-entropy over those codes. Chance for a uniform code is `ln(1024) ≈ 6.93`.

## Data

Training text is `D:\C++\LocalAI\code_corpus.txt`.

- Original source: 61.4 MB, 2,662 files, each introduced by a `# FILE:` header. It starts with Liquid Glass Dock (`CMakeLists.txt`, `src\DockApp.cpp`, and the rest) and continues through other C++, TypeScript, Python, C#, and smaller languages.
- Appendix already appended: 28.4 MB, 2,139 files. The corpus is now **89.9 MB**.

The appendix is licensed material for correctness, speed, and security:

- C++ Core Guidelines, including the performance rules. Personal or internal use only. Do not publish this corpus.
- Google C++ Style Guide and Google Python Style Guide (CC BY 3.0).
- SEI CERT C++ and SEI CERT C rules, with noncompliant and compliant examples (CC BY 4.0, examples MIT).
- OpenSSF Secure Coding Guide for Python, with `compliant` and `noncompliant` pairs (CC BY 4.0, examples MIT).
- PEP 8 (public domain).
- OWASP cheat sheets for Django, Django REST Framework, and FastAPI (CC BY-SA 4.0).
- Scientific Python optimizing lecture and its three example programs (CC BY 4.0).
- CPython `Lib` sources, test directories omitted (PSF license).
- Abseil C++ sources, test and benchmark files omitted (Apache 2.0).

Not in the corpus, because those texts are not licensed for this copy: *Effective Modern C++*, *C++ Coding Standards*, *Secure Coding in C and C++*, *Effective Python*, *High Performance Python*, Agner Fog’s optimization manual, Drepper’s memory paper, the Real Python security checklist, and the Abseil performance-hints essay.

There are still no tool traces in the file. Build them before training.

### Splits

Hold out whole files, about 5% of `# FILE:` entries, for validation. Do not cut a validation file into the training windows. Pack the rest on `# FILE:` boundaries so a window does not start mid-file.

### Continuation examples

- Prefix: the previous bytes of the same file, right-aligned in the prefix window.
- Target: the next 256 bytes.
- This is the code the model learns, including the guides above.

### Tool examples

Prefix: one task line, the path, and a small window of real lines from that file, right-aligned.

Target, always this shape, then pad:

```text
<think>
one or two sentences that name the file and the line
</think>
<tool>
{"name":"read","path":"CMakeLists.txt"}
</tool>
```

Names are `read`, `edit`, `run`, and `stop`.

- `edit` `old` and `new` are spans that appear in that example’s prefix.
- `run` is a command that already appears in the repo, such as the CMake build. No free-form shell.
- A second-turn example puts the tool result back into the prefix. The target is one follow-up call, or `{"name":"stop"}`.
- Pad bytes after `</tool>` are masked out of the loss.

Every tool argument has to be a copy from that example’s prefix. That is what makes a call depend on the file in front of the model.

Build these turns from the real files in the corpus. Do not invent paths or snippets.

## Objective

Cross-entropy on every code. Weight the 4 coarse codes by 4 so the plan is not lost under the 496 finer codes.

- Warmup: 500 steps.
- Then hold the learning rate flat.
- Cosine decay over the last 20% of training steps.
- Weight decay 0.01. Grad clip 1.0.

Choose the learning rate on continuation batches, 500 steps each, at `3e-4`, `1e-3`, and `3e-3`. Keep the rate with the lowest held-out file loss. On a single pass, `3e-4` under-trains.

Watch held-out file loss, not training loss. Training loss looks better because the same files repeat.

## Schedule

| Time | Work |
|---|---|
| 0:00–0:20 | Fit `prefix_len` 64, then 128, at batch 8. Keep the longest one that stays under about 6.5 GB and at least about 3000 code tokens/s. |
| 0:20–0:50 | Pack continuation windows on file boundaries. Build tool turns from real files. Freeze the 20 check prompts from the held-out files. |
| 0:50–1:15 | Learning-rate probe. |
| 1:15–6:15 | Continuation pretrain. |
| 6:15–7:45 | Mixed finetune: half continuation, half tool turns. |
| 7:45–8:00 | Export and the checks below. |

Save a checkpoint about every 30 minutes. At 3765 code tokens/s the training window is about 6.5 hours and about 88 million code tokens. The enlarged corpus is about 1.5 times the old 61 MB file, so this is a partial pass, not a memorization run. Loss should move below 6.93. It will not collapse.

If the prefix fit lands near 3000 tokens/s, the same window sees less text. Do not shrink the continuation block to make room for a longer prefix.

## Checkpoint to export

Score the mixed-finetune checkpoints on the 20 prompts and on held-out continuation loss.

Export the checkpoint with the best tool-parse rate whose continuation loss is still clearly under 6.93. If mixing raises held-out continuation loss by more than about 0.15, export the last pretrain checkpoint that still parses a call.

## Checks

Twenty frozen prompts, scored before export:

- Exactly one `<think>` block, non-empty, before the call, naming the file or the line in the prefix.
- Exactly one `<tool>` object. `name` is `read`, `edit`, `run`, or `stop`.
- For `edit`, `old` is a substring of the prefix. For `read` and `edit`, `path` is a real file in the corpus.
- One follow-up: run `read` in the harness, put the returned lines into the next prefix, and score whether the next sample is one call or `stop`.

A good prompt about `CMakeLists.txt` should produce a short think that names `project()` and one edit that quotes `VERSION 1.1.85`.

## Inference and export

Draw 4 drafts at temperature 0.3 and keep the first that passes the parser. The harness runs `read` by returning a file slice, `edit` only when `old` matches, and `run` only for the allowlisted build command. Then it stops, or feeds that result back for one more turn.

Write a new Q8_0 GGUF at `ncm_gguf/code_ncm_q8.gguf`.

- Tokenizer and generator only. No Adam state.
- Two-dimensional weights and codebooks stored as GGML Q8_0.
- One-dimensional norms and biases stay FP32.
- Do not overwrite `ncm_gguf/code_ncm.gguf` or `ncm_gguf/code_ncm_1b.gguf`.
- llama.cpp cannot run it. The file is a container for the Python NCM runtime.

## What this run will and will not do

After the loss is below chance, the model should stop emitting walls of `e`, follow this corpus’s style, and often emit one parseable think plus one tool call. It will not plan a multi-file change, and it will not recover from a bad call on its own.
