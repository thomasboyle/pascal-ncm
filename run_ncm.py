"""Run the Next-Concept Modeling (NCM) generator on GPU.

Usage:
    python D:/C++/LocalAI/run_ncm.py [--task tinystories] [--n 4] [--checkpoint-dir ...]
"""
import argparse, os, sys

ROOT = "D:/C++/LocalAI/ncm"
sys.path.insert(0, ROOT)

import torch
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tinystories")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(ROOT, "config", "eval", f"{args.task}.yaml")
    ckpt_dir = args.checkpoint_dir or os.path.join("D:/C++/LocalAI/artifacts", args.task)
    CFG = OmegaConf.load(cfg_path)

    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, CFG.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if CFG.checkpoints.get(name):
            CFG.tokenizer.pre_tokenizer[key] = os.path.join(ckpt_dir, CFG.checkpoints[name])

    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
    sd = torch.load(os.path.join(ckpt_dir, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    torch.manual_seed(args.seed)
    L0 = tk.vqvae.vq.levels[0]
    L0_P = None
    if CFG.generation.get("level0_unigram"):
        spec = str(CFG.generation.level0_unigram)
        path, ref = spec.split(":") if ":" in spec else (spec, None)
        obj = torch.load(os.path.join(ckpt_dir, path), map_location="cuda", weights_only=False)
        if ref:
            for k in ref.split("."):
                obj = obj[k]
        L0_P = obj.float()
    P = int(CFG.model.prefix_len)
    EOS = tk.pre_tokenizer.eos_token_id
    PAD = tk.pre_tokenizer.pad_token_id

    n = args.n
    bs = args.batch_size
    out_texts = []
    with torch.no_grad():
        for s in range(0, n, bs):
            B = min(bs, n - s)
            l0 = None
            if L0_P is not None:
                l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).cuda()
            x = torch.full((B, 1), EOS, device="cuda")
            g = net.generate(l0, x, temperature=float(CFG.generation.temperature),
                             top_k=int(CFG.generation.top_k), top_p=float(CFG.generation.top_p),
                             cfg_scale=float(CFG.generation.cfg_scale), cfg_null_eos=EOS)
            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
            for r in ids:
                r = r.tolist()
                if EOS in r:
                    r = r[:r.index(EOS)]
                out_texts.append(tk.pre_tokenizer.decode([t for t in r if t not in (EOS, PAD)], skip_special_tokens=True))

    for i, t in enumerate(out_texts):
        print(f"=== sample {i} ===")
        print(t)
        print()


if __name__ == "__main__":
    main()
