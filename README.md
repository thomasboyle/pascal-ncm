# pascal-ncm

Next-Concept Modeling experiments trained locally on a GTX 1070 Ti (Pascal, sm_61). The architecture in `ncm/` is the MIT-licensed implementation from [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (arXiv 2610.14162). llama.cpp has no NCM operator, so the GGUF files are containers for the Python runtime.

## Layout

- `ncm/` — tokenizer and coarse-to-fine generator.
- `train_code_ncm.py`, `train_code_ncm_v2.py` — the 1.2M toy run. Weights are in `code_artifacts/`, and `ncm_gguf/code_ncm.gguf` is the 2.5 MB export.
- `train_scaled.py`, `bench_ncm_speed.py` — the 989M run. The frozen tokenizer is `code_artifacts_scaled/vqvae.pt`. Inference on that model, FP16, batch 1, 256 tokens, was 662.4 tokens/s (`code_artifacts_scaled/speed_report.txt`).
- `train_fast.py`, `train_agent.py`, `TRAINING_PLAN.md` — the 87M on-GPU agent run. Logs and the prompt-check report are in `code_artifacts_agent/`.
- `export_gguf.py`, `export_scaled_gguf.py`, `verify_gguf.py`, `run_ncm.py`, `run_code_ncm.py` — export and generation.

Scripts still point at `D:\C++\LocalAI`. This repo is private. `code_corpus.txt` includes the C++ Core Guidelines, which are for personal or internal use only.

## Large weights

Files over GitHub's 100 MB git limit are release assets on the tag `training-weights`. On the other machine, from the clone:

```
gh release download training-weights -D code_artifacts_agent -p "data_p128.npz" -p "cont_best.pt"
gh release download training-weights -D ncm_gguf -p "code_ncm_q8_cont.gguf"
```

Continue from `code_artifacts_agent/cont_best.pt` with `data_p128.npz`. The frozen tokenizer is already in `code_artifacts_scaled/vqvae.pt`. `code_ncm_q8_cont.gguf` is the Q8 export of that checkpoint.
