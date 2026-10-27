"""Wall-clock generation comparison for the trained ~1B NCM code model.

Three schedules, same GPU, FP16 weights, batch 1, 256 output tokens:

  ncm        trained coarse-to-fine generator (one forward per code level)
  ar         matched-width autoregressive stack, one new token per step, KV cache
  iterative  same stack run over the full 256 tokens, repeated like a
             full-sequence diffusion step. 128 repeats is the NFE budget the
             NCM paper uses for its diffusion baselines (LM1B / Sudoku).

The iterative and AR models are speed baselines, not trained language models.
"""
import os, sys, json, time, types
import torch
import torch.nn as nn

sys.path.insert(0, "D:/C++/LocalAI")
sys.path.insert(0, "D:/C++/LocalAI/ncm")
from models.common import Block, LayerNorm
from train_scaled import build_tokenizer, build_generator

SRC = "D:/C++/LocalAI/code_artifacts_scaled"
OUT = os.path.join(SRC, "speed_report.txt")
DEV = "cuda"


def sync():
    torch.cuda.synchronize()


def time_call(fn, repeats):
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeats


def load_ncm(cfg):
    vqvae = build_tokenizer(
        cfg["levels"], cfg["tok_embed"], cfg["codebook"], cfg["tok_layers"], cfg["dec_layers"]
    ).to(DEV)
    vqvae.load_state_dict(torch.load(os.path.join(SRC, "vqvae.pt"), map_location=DEV))
    vqvae.eval()
    net = build_generator(vqvae, cfg["gen_embd"], cfg["gen_layers"], cfg["gen_heads"])
    blob = torch.load(os.path.join(SRC, "generator.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict(blob["model"], strict=True)
    net = net.to(DEV).half().eval()
    return vqvae, net, blob.get("step")


def bench_ncm(vqvae, net, repeats):
    L0 = int(vqvae.vq.levels[0])
    eos = 256

    def once():
        l0 = torch.randint(0, vqvae.vq.vqvae_vocab_sizes[0], (1, L0), device=DEV)
        prefix = torch.full((1, 1), eos, device=DEV)
        codes = net.generate(l0, prefix, temperature=0.0, top_k=0, top_p=1.0,
                             use_kv_cache=False, cfg_scale=0.0)
        z = vqvae.vq.idx_to_z_q(codes)
        return vqvae.decoder(z).argmax(-1)

    with torch.no_grad():
        once()
        dt = time_call(once, repeats)
    return dt


class AR(nn.Module):
    def __init__(self, n_embd, n_layer, n_head, vocab, seq):
        super().__init__()
        cfg = types.SimpleNamespace(
            n_embd=n_embd, n_head=n_head, bias=True, dropout=0.0,
            use_qk_norm=True, use_swiglu=False,
        )
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(seq, n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layer)])
        self.ln = LayerNorm(n_embd, bias=True)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.seq = seq

    def full_forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def fix_ar_generate(model, n):
    """Token-by-token with KV cache. Positions are added inside the loop."""
    device = model.wte.weight.device
    for blk in model.blocks:
        blk.attn.kv_caching(True)
    token = torch.zeros(1, 1, dtype=torch.long, device=device)
    try:
        for pos in range(n):
            x = model.wte(token) + model.wpe.weight[pos].view(1, 1, -1)
            for blk in model.blocks:
                x = blk(x)
            token = model.head(model.ln(x))[:, -1].argmax(-1, keepdim=True)
    finally:
        for blk in model.blocks:
            blk.attn.kv_caching(False)


def bench_baselines(cfg, repeats, diff_nfe):
    n_embd, n_layer, n_head = cfg["gen_embd"], cfg["gen_layers"], cfg["gen_heads"]
    L = cfg["seq_len"]
    ar = AR(n_embd, n_layer, n_head, vocab=258, seq=L).to(DEV).half().eval()
    nparam = sum(p.numel() for p in ar.parameters())
    idx = torch.randint(0, 258, (1, L), device=DEV)
    with torch.no_grad():
        ar.full_forward(idx)
        fix_ar_generate(ar, 4)
        full_dt = time_call(lambda: ar.full_forward(idx), repeats)
        ar_dt = time_call(lambda: fix_ar_generate(ar, L), max(1, repeats // 2))
    return nparam, full_dt, ar_dt


def main():
    with open(os.path.join(SRC, "scaled_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    repeats = 4
    diff_nfe = 128
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit(f"device: {torch.cuda.get_device_name(0)}")
    emit(f"levels: {cfg['levels']}  seq_len: {cfg['seq_len']}  "
         f"generator: {cfg['gen_embd']} x {cfg['gen_layers']} x {cfg['gen_heads']}")
    vqvae, net, step = load_ncm(cfg)
    n_gen = sum(p.numel() for p in net.parameters())
    emit(f"ncm generator params: {n_gen/1e6:.1f}M  trained steps: {step}")
    nfe = len(cfg["levels"]) - 1
    with torch.no_grad():
        ncm_dt = bench_ncm(vqvae, net, repeats)
    emit(f"ncm: {ncm_dt*1000:.1f} ms/seq  nfe={nfe}  "
         f"tokens/s={cfg['seq_len']/ncm_dt:.1f}")
    del net, vqvae
    torch.cuda.empty_cache()

    nparam, full_dt, ar_dt = bench_baselines(cfg, repeats, diff_nfe)
    emit(f"ar matched stack params: {nparam/1e6:.1f}M")
    emit(f"ar: {ar_dt*1000:.1f} ms/seq  nfe={cfg['seq_len']}  "
         f"tokens/s={cfg['seq_len']/ar_dt:.1f}  "
         f"speedup of ncm vs ar: {ar_dt/ncm_dt:.2f}x")
    iter_dt = full_dt * diff_nfe
    emit(f"full-sequence step: {full_dt*1000:.1f} ms  "
         f"(one forward over all {cfg['seq_len']} tokens)")
    emit(f"iterative x{diff_nfe}: {iter_dt*1000:.1f} ms/seq  nfe={diff_nfe}  "
         f"tokens/s={cfg['seq_len']/iter_dt:.2f}  "
         f"speedup of ncm vs iterative: {iter_dt/ncm_dt:.2f}x")
    emit("iterative x128 is measured per full-sequence forward, times 128. "
         "That is the diffusion NFE budget cited for LM1B and Sudoku in arXiv 2610.14162. "
         "It is a cost model of full-length refinement, not a trained diffusion model.")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[bench] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
