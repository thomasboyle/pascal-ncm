"""Generate code with the small NCM model trained on Thomas Boyle's GitHub code."""
import os, sys, torch
sys.path.insert(0, "D:/C++/LocalAI/ncm")
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

DEVICE = "cuda"
OUT = "D:/C++/LocalAI/code_artifacts"
VOCAB = 258; EOS = 256; PAD = 257; L = 64
levels = [1, 4, 8, 16, 32, 64]

vq_cfg = OmegaConf.create({
    "clss": "multiscale_residual_vector_quantizer",
    "decay": 0.99, "commitment_cost": 0.25, "epsilon": 1e-5,
    "learned_l1_sampling": True, "learned_l1_cascade": True,
    "learned_l1_cascade_base": 4, "learned_all_sampling": False,
    "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
    "aux": {"fine_drop": {"prob": 0.3, "min_keep": 1}},
    "levels": {"use_manual_levels": True, "manual_levels": levels},
})
vqvae = VQVAE(in_vocab_size=VOCAB, context_length=L, embed_dim=64,
    vqvae_vocab_size=[256]*len(levels), n_layers=4,
    vector_quantizer_config=vq_cfg, compression_factor=1, dropout=0.1,
    pre_quant_groupnorm=8, pre_quant_dropout=0.0, use_qk_norm=True,
    enc_layers=0, dec_layers=1, pre_quant_norm_type="group").to(DEVICE)
vqvae.load_state_dict(torch.load(os.path.join(OUT, "vqvae.pt"), map_location=DEVICE))
vqvae.eval()

class Cfg: pass
cfg = Cfg()
cfg.n_embd=128; cfg.n_layer=4; cfg.n_head=4; cfg.bias=True; cfg.dropout=0.1
cfg.vocab_size=[256]*(len(levels)-1); cfg.attn_pattern="block_diagonal"
cfg.prefix_len=1; cfg.raw_token_prefix=False; cfg.prefix_pad_token_id=PAD
cfg.use_positional_encoding=True; cfg.use_level_encoding=True; cfg.use_rope=True
cfg.rope_base=100000; cfg.use_qk_norm=True
cfg.ctx_encoder=OmegaConf.create({"enabled": False})
cfg.use_flex_attention=False; cfg.query_level0=False
cfg.tie_heads_to_codebook=True; cfg.tie_heads_learn_scale=True
cfg.predict_level0=False
cfg.post_upsample_conv=OmegaConf.create({"enabled": False})
cfg.shared_output_head=False; cfg.use_swiglu=False

net = NextConceptTransformer(vqvae_local=vqvae, config=cfg).to(DEVICE)
sd = torch.load(os.path.join(OUT, "generator.pt"), map_location=DEVICE, weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()

torch.manual_seed(0)
L0 = vqvae.vq.levels[0]
# level0 unigram from codebook? just uniform
L0_P = torch.ones(256) / 256
n=4; bs=2
out=[]
with torch.no_grad():
    for s in range(0, n, bs):
        B=min(bs,n-s)
        l0 = torch.multinomial(L0_P, B*L0, replacement=True).view(B, L0).to(DEVICE)
        x = torch.full((B,1), EOS, device=DEVICE)
        g = net.generate(l0, x, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
        _, zl = vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        ids = vqvae.decoder(zl[-1]).argmax(-1)
        for r in ids:
            r = r.tolist()
            if EOS in r: r = r[:r.index(EOS)]
            out.append(bytes([t for t in r if t < 256]).decode("utf-8", errors="replace"))

for i,t in enumerate(out):
    print(f"=== sample {i} ===")
    print(t[:500])
    print()
PY = None
