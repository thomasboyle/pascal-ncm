import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "D:/C++/LocalAI/ncm")
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

DEVICE = "cuda"
OUT = "D:/C++/LocalAI/code_artifacts"
os.makedirs(OUT, exist_ok=True)

# Byte-level tokenizer params
VOCAB = 258  # 0-255 bytes + 256 eos + 257 pad
EOS = 256
PAD = 257
L = 64

# ---- data ----
def make_sequences(path, L):
    with open(path, "rb") as f:
        data = f.read()
    # keep ascii-ish
    arr = list(data)
    # map bytes to 0-255 (already), build sequences
    seqs = []
    i = 0
    n = len(arr)
    while i + L <= n:
        seqs.append(arr[i:i+L])
        i += L
    # convert to tensor
    t = torch.tensor(seqs, dtype=torch.long)
    return t

corpus_path = "D:/C++/LocalAI/code_corpus.txt"
seqs = make_sequences(corpus_path, L)
print("num seqs", seqs.shape)
# split train
ntrain = int(0.95 * len(seqs))
train_seqs = seqs[:ntrain]
val_seqs = seqs[ntrain:]

# ---- VQ-VAE ----
levels = [1, 4, 8, 16, 32, 64]
vq_cfg = OmegaConf.create({
    "clss": "multiscale_residual_vector_quantizer",
    "decay": 0.99,
    "commitment_cost": 0.25,
    "epsilon": 1e-5,
    "learned_l1_sampling": True,
    "learned_l1_cascade": True,
    "learned_l1_cascade_base": 4,
    "learned_all_sampling": False,
    "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
    "aux": {"fine_drop": {"prob": 0.3, "min_keep": 1}},
    "levels": {"use_manual_levels": True, "manual_levels": levels},
})
vqvae = VQVAE(
    in_vocab_size=VOCAB,
    context_length=L,
    embed_dim=64,
    vqvae_vocab_size=[256]*len(levels),
    n_layers=4,  # head count for decoder block
    vector_quantizer_config=vq_cfg,
    compression_factor=1,
    dropout=0.1,
    pre_quant_groupnorm=8,
    pre_quant_dropout=0.0,
    use_qk_norm=True,
    enc_layers=0,
    dec_layers=1,
    pre_quant_norm_type="group",
).to(DEVICE)

opt_vq = torch.optim.AdamW(vqvae.parameters(), lr=3e-4, weight_decay=0.01)
bs = 16
steps = 2000
for step in range(steps):
    idx = torch.randint(0, len(train_seqs), (bs,))
    x = train_seqs[idx].to(DEVICE)
    logits, vq_loss, _, _, _ = vqvae(x)
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
    loss = ce + vq_loss
    opt_vq.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
    opt_vq.step()
    if step % 100 == 0:
        print(f"vqvae step {step} ce={ce.item():.4f} vq={vq_loss.item():.4f}", flush=True)

torch.save(vqvae.state_dict(), os.path.join(OUT, "vqvae.pt"))
print("saved vqvae")

# ---- Generator ----
# Build config-like object
class Cfg: pass
cfg = Cfg()
cfg.n_embd = 128
cfg.n_layer = 4
cfg.n_head = 4
cfg.bias = True
cfg.dropout = 0.1
cfg.vocab_size = [256]* (len(levels)-1)  # levels 1..K-1
cfg.attn_pattern = "block_diagonal"
cfg.prefix_len = 1
cfg.raw_token_prefix = False
cfg.prefix_pad_token_id = PAD
cfg.use_positional_encoding = True
cfg.use_level_encoding = True
cfg.use_rope = True
cfg.rope_base = 100000
cfg.use_qk_norm = True
cfg.ctx_encoder = OmegaConf.create({"enabled": False})
cfg.use_flex_attention = False
cfg.query_level0 = False
cfg.tie_heads_to_codebook = True
cfg.tie_heads_learn_scale = True
cfg.predict_level0 = False
cfg.post_upsample_conv = OmegaConf.create({"enabled": False})
cfg.shared_output_head = False
cfg.use_swiglu = False

vqvae.eval()
net = NextConceptTransformer(vqvae_local=vqvae, config=cfg).to(DEVICE)
opt_g = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.1)

# Precompute ncm inputs for a subset? We'll do online.
gsteps = 2000
for step in range(gsteps):
    idx = torch.randint(0, len(train_seqs), (bs//2,))  # smaller batch
    xtok = train_seqs[idx].to(DEVICE)
    with torch.no_grad():
        cond, target = vqvae.get_ncm_inputs(xtok)  # cond: (B, L-L0, d), target: (B, L-L0)
    # prefix in vq space: use eos embedding
    prefix_tokens = torch.full((xtok.shape[0], 1), EOS, device=DEVICE, dtype=torch.long)
    z_prefix = vqvae.encoder.wte(prefix_tokens)  # (B,1,d)
    x = torch.cat([z_prefix, cond], dim=1)  # (B, 1+L-L0, d)
    logits = net(x)  # (B, L-L0, maxvocab)
    # target shape (B, L-L0)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    opt_g.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt_g.step()
    if step % 100 == 0:
        print(f"gen step {step} loss={loss.item():.4f}", flush=True)

torch.save({"model": net.state_dict()}, os.path.join(OUT, "generator.pt"))
print("saved generator")

# Save config yaml for run_ncm compatibility? We'll just note.
print("DONE")
PY = None
