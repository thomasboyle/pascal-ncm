"""Export the trained byte-level NCM code model to a single GGUF container.

The NCM system is two coupled parts that MUST travel together:
  * tokenizer  : frozen multiscale-residual VQ-VAE (encoder + codebooks + decoder)
  * generator  : NextConceptTransformer whose output heads are TIED to the
                 tokenizer codebooks (a codebook-distance head).
llama.cpp has no NCM coarse-to-fine op, so this GGUF is a faithful *container*
of the whole system (verifiable with the gguf reader), not a stock-llama.cpp
runnable LM. Tensors from both state dicts are written with NCM metadata.
"""
import os, sys, types
import numpy as np
import torch

sys.path.insert(0, "D:/C++/LocalAI/ncm")
import gguf
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

SRC = "D:/C++/LocalAI/code_artifacts"
OUT_DIR = "D:/C++/LocalAI/ncm_gguf"
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "code_ncm.gguf")

VOCAB = 258
EOS, PAD = 256, 257
L = 64
LEVELS = [1, 4, 8, 16, 32, 64]


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
    )


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
    return NextConceptTransformer(vqvae_local=vqvae, config=cfg)


def to_array(t):
    # F16 for sizable weight matrices; F32 for codebooks/norms/biases/scalars.
    a = t.detach().cpu().numpy() if hasattr(t, "numpy") else np.asarray(t)
    if a.ndim >= 2 and a.size >= 256:
        return a.astype(np.float16)
    return a.astype(np.float32)


def main():
    dev = "cpu"
    print("[export] building model", flush=True)
    vqvae = build_vqvae().to(dev)
    vqvae.load_state_dict(torch.load(os.path.join(SRC, "vqvae.pt"), map_location=dev))
    vqvae.eval()

    net = build_generator(vqvae).to(dev)
    sd = torch.load(os.path.join(SRC, "generator.pt"), map_location=dev, weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    print("[export] checkpoints loaded OK", flush=True)

    # Collect tensors (detached, numpy). Prefix by stage; keep original submodule path.
    tensors = {}
    for k, v in vqvae.state_dict().items():
        tensors[f"tokenizer.{k}"] = v.detach().float().cpu()
    for k, v in net.state_dict().items():
        tensors[f"generator.{k}"] = v.detach().float().cpu()

    # Drop purely transient / cluster-tracking buffers that aren't needed to run.
    drop_prefix = ("tokenizer.vq.ema_", "tokenizer.vq.codebook_hit_")
    tensors = {k: v for k, v in tensors.items()
               if not k.startswith(drop_prefix)}

    print(f"[export] {len(tensors)} tensors", flush=True)

    writer = gguf.GGUFWriter(OUT, "ncm", use_temp_file=False)
    writer.add_name("code_ncm_byte64_l6")
    writer.add_string("description",
                      "Byte-level NCM (Next-Concept Modeling) code model: "
                      "multiscale residual VQ-VAE tokenizer + coarse-to-fine generator. "
                      "Author-trained experiment.")
    # Architecture metadata
    writer.add_uint32("levels_count", len(LEVELS))
    try:
        writer.add_array("levels", list(LEVELS))
    except Exception:
        for i, l in enumerate(LEVELS):
            writer.add_uint32(f"level.{i}", l)
    writer.add_array("codebook_sizes", [256] * len(LEVELS))
    writer.add_uint32("tokenizer.embed_dim", 64)
    writer.add_uint32("tokenizer.in_vocab_size", VOCAB)
    writer.add_uint32("tokenizer.n_layers", 4)
    writer.add_uint32("generator.n_embd", 128)
    writer.add_uint32("generator.n_layer", 4)
    writer.add_uint32("generator.n_head", 4)
    writer.add_uint32("generator.rope_base", 100000)
    writer.add_uint32("generator.n_levels_predicted", len(LEVELS) - 1)
    writer.add_uint32("seq_len", L)
    writer.add_uint32("eos_token_id", EOS)
    writer.add_uint32("pad_token_id", PAD)
    writer.add_uint32("attn_pattern", 0)  # 0 = block_diagonal
    # Counts
    n_tok = sum(v.numel() for k, v in tensors.items() if k.startswith("tokenizer."))
    n_gen = sum(v.numel() for k, v in tensors.items() if k.startswith("generator."))
    writer.add_uint32("tokenizer.param_count", n_tok)
    writer.add_uint32("generator.param_count", n_gen)
    writer.add_uint32("total_param_count", n_tok + n_gen)

    # Write tensor info + data
    for name, t in tensors.items():
        writer.add_tensor(name, to_array(t))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"[export] wrote {OUT}  ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
