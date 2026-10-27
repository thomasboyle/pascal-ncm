"""Scaled NCM trainer: big VQ-VAE tokenizer (native FP16) + ~1B coarse-to-fine
generator trained with FP16 forward/backward on GPU and CPU-offloaded Adam.

Why CPU offload: a 1B model's Adam state (FP32 master + moments) is ~13GB, which
does not fit in 8GB VRAM, but fits in system RAM. The GPU still runs every
forward/backward matmul (the actual compute); only the optimizer step runs on CPU.

bnb 8-bit Adam was tried but its Windows wheel has no Pascal (sm_61) kernel.

Modes:
  --mode probe   build model, run a few steps, report peak VRAM + steps/sec
  --mode train   full run, logs to code_artifacts/scaled_train_log.txt
"""
import os, sys, time, argparse, types, math, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, "D:/C++/LocalAI/ncm")
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

DEVICE = "cuda"
OUT = "D:/C++/LocalAI/code_artifacts_scaled"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "scaled_train_log.txt")
CORPUS = "D:/C++/LocalAI/code_corpus.txt"


def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ---------------- FP16 + CPU-offloaded Adam trainer ----------------
class OffloadFP16:
    def __init__(self, model, lr, weight_decay, device=DEVICE, init_scale=1.0, max_scale=1.0):
        self.device = device
        self.model = model.to(device).half()
        for p in self.model.parameters():
            p.requires_grad_(True)
        # FP32 master weights live in CPU RAM.
        self.master = [p.detach().float().cpu().clone() for p in self.model.parameters()]
        for p in self.master:
            p.requires_grad_(True)
        self.opt = torch.optim.AdamW(self.master, lr=lr, weight_decay=weight_decay,
                                     betas=(0.9, 0.95), eps=1e-8)
        # FP16 grads sum over the sequence. A large loss scale overflows them to inf
        # (the previous probe used 2048 and skipped every step). Keep the scale at 1.
        self.loss_scale = init_scale
        self.max_scale = max_scale
        self.skipped = 0
        self.last_real = None
        self.last_skip = None
        self.pre_bwd = None
        self._tgt = None
        self._names = [n for n, _ in self.model.named_parameters()]

    def _sync_down(self):
        with torch.no_grad():
            for mp, gp in zip(self.master, self.model.parameters()):
                gp.data.copy_(mp.to(self.device, dtype=torch.float16))

    @torch.no_grad()
    def _has_inf(self):
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def step(self, fwd_loss):
        """fwd_loss(model, x) -> scalar (unscaled). Returns real loss or None if skipped."""
        self.opt.zero_grad(set_to_none=True)
        self._sync_down()

        def _fwd(x):
            return fwd_loss(self.model, x) * self.loss_scale

        try:
            l = checkpoint(_fwd, self._x, use_reentrant=False)
        except RuntimeError:
            self.model.zero_grad(set_to_none=True)
            l = fwd_loss(self.model, self._x) * self.loss_scale
        self.pre_bwd = float(l.detach().float().item())
        l.backward()

        real = l.detach().float().item() / self.loss_scale
        self.last_real = real
        if not math.isfinite(real):
            self.last_skip = "nonfinite-loss"
        elif self._has_inf():
            self.last_skip = "nonfinite-grad"
        else:
            self.last_skip = None
        if self.last_skip is not None:
            self.model.zero_grad(set_to_none=True)
            self.opt.zero_grad(set_to_none=True)
            self.skipped += 1
            return None

        with torch.no_grad():
            for mp, gp in zip(self.master, self.model.parameters()):
                if gp.grad is not None:
                    mp.grad = (gp.grad.detach().float() / self.loss_scale).to("cpu")
                else:
                    mp.grad = None
            self.model.zero_grad(set_to_none=True)
            torch.nn.utils.clip_grad_norm_(self.master, 1.0)
            self.opt.step()
        if self.loss_scale < self.max_scale:
            self.loss_scale = min(self.max_scale, self.loss_scale * 1.002)
        return real

    def named_state_dict(self):
        """FP32 weights aligned with model.state_dict(), plus persistent buffers."""
        sd = self.model.state_dict()
        masters = dict(zip(self._names, self.master))
        for k in list(sd.keys()):
            if k in masters:
                sd[k] = masters[k].detach().float().cpu().clone()
            else:
                sd[k] = sd[k].detach().float().cpu().clone()
        return sd

    def load_named(self, sd):
        with torch.no_grad():
            for name, mp in zip(self._names, self.master):
                if name in sd:
                    mp.copy_(sd[name].detach().float().cpu())


# ---------------- model builders (config-driven, reuse ncm classes) ----------------
def build_tokenizer(levels, embed_dim, codebook, n_layers, dec_layers):
    vq_cfg = OmegaConf.create({
        "clss": "multiscale_residual_vector_quantizer",
        "decay": 0.99, "commitment_cost": 0.25, "epsilon": 1e-5,
        "learned_l1_sampling": True, "learned_l1_cascade": True,
        "learned_l1_cascade_base": 4, "learned_all_sampling": False,
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
        "aux": {"fine_drop": {"prob": 0.3, "min_keep": 1}},
        "levels": {"use_manual_levels": True, "manual_levels": levels},
    })
    return VQVAE(
        in_vocab_size=258, context_length=max(levels), embed_dim=embed_dim,
        vqvae_vocab_size=[codebook] * len(levels), n_layers=n_layers,
        vector_quantizer_config=vq_cfg, compression_factor=1, dropout=0.1,
        pre_quant_groupnorm=8, pre_quant_dropout=0.0, use_qk_norm=True,
        enc_layers=0, dec_layers=dec_layers, pre_quant_norm_type="group",
    )


def build_generator(vqvae, n_embd, n_layer, n_head):
    K = len(vqvae.vq.levels)
    cfg = types.SimpleNamespace(
        n_embd=n_embd, n_layer=n_layer, n_head=n_head, bias=True, dropout=0.0,
        vocab_size=[vqvae.vq.vqvae_vocab_sizes[i] for i in range(1, K)],
        attn_pattern="block_diagonal", prefix_len=1, raw_token_prefix=False,
        prefix_pad_token_id=257, use_positional_encoding=True, use_level_encoding=True,
        use_rope=True, rope_base=100000, use_qk_norm=True,
        ctx_encoder=OmegaConf.create({"enabled": False}),
        use_flex_attention=False, query_level0=False,
        tie_heads_to_codebook=True, tie_heads_learn_scale=True,
        predict_level0=False,
        post_upsample_conv=OmegaConf.create({"enabled": False}),
        shared_output_head=False, use_swiglu=False,
    )
    return NextConceptTransformer(vqvae_local=vqvae, config=cfg)


def make_sequences(path, L, n=None):
    with open(path, "rb") as f:
        data = f.read()
    arr = list(data)
    seqs = []
    i = 0
    while i + L <= len(arr):
        seqs.append(arr[i:i + L])
        i += L
        if n and len(seqs) >= n:
            break
    return torch.tensor(seqs, dtype=torch.long)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "train"], default="probe")
    p.add_argument("--levels", type=int, nargs="+", default=[4, 16, 32, 64, 128, 256])
    p.add_argument("--tok-embed", type=int, default=512)
    p.add_argument("--codebook", type=int, default=1024)
    p.add_argument("--tok-layers", type=int, default=8)
    p.add_argument("--gen-embd", type=int, default=1600)
    p.add_argument("--gen-layers", type=int, default=32)
    p.add_argument("--gen-heads", type=int, default=16)
    p.add_argument("--tok-steps", type=int, default=8000)
    p.add_argument("--gen-steps", type=int, default=20000)
    p.add_argument("--gen-bs", type=int, default=2)
    p.add_argument("--tok-bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--probe-steps", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200)
    return p.parse_args()


def main():
    args = parse()
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    L = max(args.levels)

    log(f"[cfg] levels={args.levels} L={L} tok_embed={args.tok_embed} codebook={args.codebook} "
        f"gen {args.gen_embd}x{args.gen_layers}x{args.gen_heads} lr={args.lr} mode={args.mode}")

    # Sequences stay on CPU. Only the batch moves to the GPU, so the 1B
    # generator's activations are not competing with the whole corpus.
    vqvae = build_tokenizer(args.levels, args.tok_embed, args.codebook, args.tok_layers, dec_layers=2).to(dev)
    log(f"[tok] params = {count_params(vqvae)/1e6:.2f}M")
    seqs = make_sequences(CORPUS, L)
    ntrain = int(0.95 * len(seqs))
    tr, va = seqs[:ntrain], seqs[ntrain:]
    log(f"[data] {len(seqs)} seqs of L={L} (train {len(tr)}, val {len(va)})")

    VOCAB = 258
    EOS = 256

    def tok_batch(src, bs):
        idx = torch.randint(0, len(src), (bs,))
        return src[idx].to(dev)

    if args.mode == "probe":
        # Do not step the tokenizer here. An unclipped update writes inf into the
        # codebook, and the generator's tied heads copy that codebook.
        vqvae.eval()
        with torch.no_grad():
            x = tok_batch(tr, args.tok_bs)
            logits, vq_loss, *_ = vqvae(x)
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
        log(f"[tok-probe] ce {ce.item():.4f} vq {float(vq_loss):.4f}")
        del x
    else:
        opt_tok = torch.optim.AdamW(vqvae.parameters(), lr=3e-4, weight_decay=0.01)
        t1 = time.time()
        for step in range(args.tok_steps):
            x = tok_batch(tr, args.tok_bs)
            logits, vq_loss, *_ = vqvae(x)
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
            loss = ce + vq_loss
            opt_tok.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
            opt_tok.step()
            if step % 200 == 0:
                cb = max(float(e.weight.detach().abs().max()) for e in vqvae.vq.embeddings)
                log(f"[vq] step {step}/{args.tok_steps} ce={ce.item():.4f} "
                    f"vq={float(vq_loss):.4f} codebook_absmax={cb:.3f} ({time.time()-t1:.1f}s)")
                if not math.isfinite(cb):
                    raise SystemExit("tokenizer codebook became non-finite")
        vqvae.eval()
        with torch.no_grad():
            xval = va[:4].to(dev)
            logits, vq_loss, *_ = vqvae(xval)
            recon = F.cross_entropy(logits.reshape(-1, VOCAB), xval.reshape(-1)).item()
        log(f"[vq] recon val ce={recon:.4f}")
        torch.save(vqvae.state_dict(), os.path.join(OUT, "vqvae.pt"))
        log("[vq] saved vqvae.pt")
        del opt_tok
        torch.cuda.empty_cache()

    # ---- generator (FP16 forward/backward on GPU, AdamW master on CPU) ----
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    net = build_generator(vqvae, args.gen_embd, args.gen_layers, args.gen_heads)
    ngen = count_params(net)
    log(f"[gen] params = {ngen/1e6:.1f}M")

    def fwd_loss(model, x):
        logits = model(x)
        tgt = trainer._tgt
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

    def gen_batch():
        xb = tok_batch(tr, args.gen_bs)
        with torch.no_grad():
            cond, target = vqvae.get_ncm_inputs(xb)
        z_prefix = vqvae.encoder.wte(torch.full((args.gen_bs, 1), EOS, device=dev))
        trainer._x = torch.cat([z_prefix, cond], dim=1).half()
        trainer._tgt = target
        return trainer.step(fwd_loss)

    if args.mode == "probe":
        trainer = OffloadFP16(net, lr=args.lr, weight_decay=0.1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        losses = []
        for _ in range(args.probe_steps):
            r = gen_batch()
            if r is not None:
                losses.append(r)
        torch.cuda.synchronize()
        dt = time.time() - t0
        avg = (sum(losses) / len(losses)) if losses else float("nan")
        log(f"[gen-probe] peak VRAM MiB {round(torch.cuda.max_memory_allocated()/1e6,1)} | "
            f"{args.probe_steps} steps in {dt:.2f}s = {args.probe_steps/max(dt,1e-6):.3f} steps/s | "
            f"loss {avg:.4f} | finite={len(losses)} skip={trainer.skipped} "
            f"last={trainer.last_skip} pre_bwd={trainer.pre_bwd} last_real={trainer.last_real} "
            f"scale={trainer.loss_scale:.1f}")
        return

    trainer = OffloadFP16(net, lr=args.lr, weight_decay=0.1)
    ckpt_path = os.path.join(OUT, "generator.pt")
    meta = {
        "levels": list(args.levels),
        "tok_embed": args.tok_embed,
        "codebook": args.codebook,
        "tok_layers": args.tok_layers,
        "dec_layers": 2,
        "gen_embd": args.gen_embd,
        "gen_layers": args.gen_layers,
        "gen_heads": args.gen_heads,
        "seq_len": L,
        "gen_params": ngen,
        "gen_steps": args.gen_steps,
        "tok_steps": args.tok_steps,
    }
    with open(os.path.join(OUT, "scaled_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    def save_gen(step):
        trainer._sync_down()
        blob = {}
        for k, v in trainer.model.state_dict().items():
            t = v.detach()
            blob[k] = t.half().cpu() if t.is_floating_point() else t.cpu()
        torch.save({"model": blob, "step": step, "skipped": trainer.skipped}, ckpt_path)
        log(f"[gen] saved {ckpt_path} @ step {step}")

    t0 = time.time()
    for step in range(args.gen_steps):
        r = gen_batch()
        if step % args.log_every == 0 or step + 1 == args.gen_steps:
            peak = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            loss_s = "skip" if r is None else f"{r:.4f}"
            log(f"[gen] step {step}/{args.gen_steps} loss={loss_s} peakMiB={peak} "
                f"scale={trainer.loss_scale:.1f} skips={trainer.skipped} "
                f"({time.time()-t0:.1f}s)")
        if args.save_every and step > 0 and step % args.save_every == 0:
            save_gen(step)
    save_gen(args.gen_steps)
    if not os.path.exists(os.path.join(OUT, "vqvae.pt")):
        torch.save(vqvae.state_dict(), os.path.join(OUT, "vqvae.pt"))
    log(f"[done] {time.time()-t0:.1f}s params={ngen/1e6:.1f}M")


if __name__ == "__main__":
    main()
