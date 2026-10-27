"""Train the byte-level NCM code model on GPU (VQ-VAE tokenizer then generator).

Picks up the pipeline from train_code_ncm.py with:
  - configurable step counts / batch / seed via env or argv
  - loss logging to stdout AND code_artifacts/train_log.txt
  - a reconstruction sanity check after the tokenizer stage
"""
import os, sys, time, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "D:/C++/LocalAI/ncm")
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

DEVICE = "cuda"
OUT = "D:/C++/LocalAI/code_artifacts"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "train_log.txt")

VOCAB = 258  # 0-255 bytes + 256 eos + 257 pad
EOS = 256
PAD = 257
L = 64
LEVELS = [1, 4, 8, 16, 32, 64]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--vq-steps", type=int, default=4000)
    p.add_argument("--gen-steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--gen-bs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_sequences(path, L):
    with open(path, "rb") as f:
        data = f.read()
    arr = list(data)
    seqs = []
    i = 0
    n = len(arr)
    while i + L <= n:
        seqs.append(arr[i:i + L])
        i += L
    return torch.tensor(seqs, dtype=torch.long)


def build_vqvae():
    vq_cfg = OmegaConf.create({
        "clss": "multiscale_residual_vector_quantizer",
        "decay": 0.99, "commitment_cost": 0.25, "epsilon": 1e-5,
        "learned_l1_sampling": True, "learned_l1_cascade": True,
        "learned_l1_cascade_base": 4, "learned_all_sampling": False,
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
        "aux": {"fine_drop": {"prob": 0.3, "min_keep": 1}},
        "levels": {"use_manual_levels": True, "manual_levels": LEVELS},
    })
    return VQVAE(
        in_vocab_size=VOCAB, context_length=L, embed_dim=64,
        vqvae_vocab_size=[256] * len(LEVELS), n_layers=4,
        vector_quantizer_config=vq_cfg, compression_factor=1, dropout=0.1,
        pre_quant_groupnorm=8, pre_quant_dropout=0.0, use_qk_norm=True,
        enc_layers=0, dec_layers=1, pre_quant_norm_type="group",
    ).to(DEVICE)


def build_generator(vqvae):
    cfg = types.SimpleNamespace(
        n_embd=128, n_layer=4, n_head=4, bias=True, dropout=0.1,
        vocab_size=[256] * (len(LEVELS) - 1), attn_pattern="block_diagonal",
        prefix_len=1, raw_token_prefix=False, prefix_pad_token_id=PAD,
        use_positional_encoding=True, use_level_encoding=True, use_rope=True,
        rope_base=100000, use_qk_norm=True,
        ctx_encoder=OmegaConf.create({"enabled": False}),
        use_flex_attention=False, query_level0=False,
        tie_heads_to_codebook=True, tie_heads_learn_scale=True,
        predict_level0=False,
        post_upsample_conv=OmegaConf.create({"enabled": False}),
        shared_output_head=False, use_swiglu=False,
    )
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg).to(DEVICE)
    return net


import types


def main():
    args = parse()
    torch.manual_seed(args.seed)
    dev = torch.device(DEVICE)
    t0 = time.time()

    log(f"[train] loading corpus + building sequences (L={L})")
    seqs = make_sequences("D:/C++/LocalAI/code_corpus.txt", L)
    ntrain = int(0.95 * len(seqs))
    train_seqs = seqs[:ntrain].to(dev)
    val_seqs = seqs[ntrain:].to(dev)
    log(f"[train] {len(seqs)} seqs | train {len(train_seqs)} | val {len(val_seqs)} | {time.time()-t0:.1f}s")

    # ---------------- Stage 1: VQ-VAE tokenizer ----------------
    vqvae = build_vqvae()
    opt_vq = torch.optim.AdamW(vqvae.parameters(), lr=3e-4, weight_decay=0.01)
    t1 = time.time()
    for step in range(args.vq_steps):
        idx = torch.randint(0, len(train_seqs), (args.bs,))
        x = train_seqs[idx]
        logits, vq_loss, _, _, _ = vqvae(x)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
        loss = ce + vq_loss
        opt_vq.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
        opt_vq.step()
        if step % 200 == 0:
            log(f"[vq] step {step}/{args.vq_steps} ce={ce.item():.4f} vq={vq_loss.item():.4f} loss={loss.item():.4f} ({time.time()-t1:.1f}s)")
    # reconstruction check on val
    vqvae.eval()
    with torch.no_grad():
        xval = val_seqs[:4]
        logits, vq_loss, _, _, _ = vqvae(xval)
        recon_ce = F.cross_entropy(logits.reshape(-1, VOCAB), xval.reshape(-1)).item()
    log(f"[vq] recon val ce={recon_ce:.4f} vq={vq_loss.item():.4f}")
    torch.save(vqvae.state_dict(), os.path.join(OUT, "vqvae.pt"))
    log("[vq] saved vqvae.pt")

    # ---------------- Stage 2: generator ----------------
    vqvae.eval()
    net = build_generator(vqvae)
    opt_g = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.1)
    L0 = vqvae.vq.levels[0]
    t2 = time.time()
    for step in range(args.gen_steps):
        idx = torch.randint(0, len(train_seqs), (args.gen_bs,))
        xtok = train_seqs[idx]
        with torch.no_grad():
            cond, target = vqvae.get_ncm_inputs(xtok)
        prefix_tokens = torch.full((xtok.shape[0], 1), EOS, device=dev)
        z_prefix = vqvae.encoder.wte(prefix_tokens)
        x = torch.cat([z_prefix, cond], dim=1)
        logits = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        opt_g.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt_g.step()
        if step % 200 == 0:
            log(f"[gen] step {step}/{args.gen_steps} loss={loss.item():.4f} ({time.time()-t2:.1f}s)")
    torch.save({"model": net.state_dict()}, os.path.join(OUT, "generator.pt"))
    log("[gen] saved generator.pt")
    log(f"[train] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
