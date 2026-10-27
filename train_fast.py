"""GPU-resident FP32 NCM trainer.

The 1B run kept Adam in system RAM and moved weights and gradients across PCIe
every step. This keeps weights, gradients, and Adam on the GPU, encodes the
corpus to codes once, and times the generator step.
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, "D:/C++/LocalAI")
sys.path.insert(0, "D:/C++/LocalAI/ncm")
from train_scaled import build_tokenizer, build_generator, make_sequences, count_params

DEV = "cuda"
CORPUS = "D:/C++/LocalAI/code_corpus.txt"
TOK_CKPT = "D:/C++/LocalAI/code_artifacts_scaled/vqvae.pt"
LEVELS = [4, 16, 32, 64, 128, 256]
EOS = 256


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe"], default="probe")
    p.add_argument("--gen-embd", type=int, default=1024)
    p.add_argument("--gen-layers", type=int, default=16)
    p.add_argument("--gen-heads", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--encode", type=int, default=2048)
    return p.parse_args()


def load_tokenizer():
    vq = build_tokenizer(LEVELS, 512, 1024, 8, 2).to(DEV).eval()
    vq.load_state_dict(torch.load(TOK_CKPT, map_location=DEV))
    for p in vq.parameters():
        p.requires_grad_(False)
    return vq


@torch.no_grad()
def encode_codes(vq, seqs, n, bs=32):
    n = min(n, len(seqs))
    chunks = []
    for i in range(0, n, bs):
        x = seqs[i:i + bs].to(DEV)
        chunks.append(vq.encode(x).cpu())
    return torch.cat(chunks, dim=0)


def bench_batch(net, vq, opt, codes, batch, steps, warmup):
    L0 = int(vq.vq.levels[0])
    npos = int(sum(vq.vq.levels[1:]))
    eos = torch.full((batch, 1), EOS, device=DEV)

    def step():
        sel = torch.randint(0, len(codes), (batch,))
        idx = codes[sel].to(DEV, non_blocking=True)
        cond = vq.vq.idx_to_ncm_input(idx)
        target = idx[:, L0:]
        x = torch.cat([vq.encoder.wte(eos), cond], dim=1)
        logits = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return loss

    for i in range(warmup):
        t = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        print(f"[warm] batch={batch} {i} {time.perf_counter()-t:.2f}s "
              f"loss={float(loss):.3f} MiB={torch.cuda.max_memory_allocated()/1e6:.0f}", flush=True)
        if time.perf_counter() - t > 20:
            print(f"[paging] batch={batch} step over 20s, skipping", flush=True)
            return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = steps * batch * npos
    print(f"[probe] batch={batch} loss={float(loss):.4f} "
          f"peakMiB={torch.cuda.max_memory_allocated()/1e6:.0f} "
          f"{steps/dt:.2f} steps/s {toks/dt:.0f} code-tok/s", flush=True)


def main():
    args = parse()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    vq = load_tokenizer()
    L = max(LEVELS)
    seqs = make_sequences(CORPUS, L)
    t_enc = time.perf_counter()
    codes = encode_codes(vq, seqs, args.encode)
    torch.cuda.synchronize()
    print(f"[codes] {len(codes)} seqs in {time.perf_counter() - t_enc:.1f}s "
          f"shape={tuple(codes.shape)}", flush=True)

    net = build_generator(vq, args.gen_embd, args.gen_layers, args.gen_heads).to(DEV)
    nparam = count_params(net)
    if args.compile:
        net = torch.compile(net)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)
    print(f"[gen] {nparam/1e6:.1f}M fused_adam=True compile={args.compile}", flush=True)
    bench_batch(net, vq, opt, codes, args.batch, args.steps, args.warmup)


if __name__ == "__main__":
    main()
