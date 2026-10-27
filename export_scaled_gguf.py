"""Export the scaled (~1B) NCM code model to a GGUF container.

llama.cpp has no coarse-to-fine NCM op, so this file is the tokenizer plus the
generator in one container, same layout as ncm_gguf/code_ncm.gguf.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, "D:/C++/LocalAI")
sys.path.insert(0, "D:/C++/LocalAI/ncm")
import gguf
from train_scaled import build_tokenizer, build_generator

SRC = "D:/C++/LocalAI/code_artifacts_scaled"
OUT_DIR = "D:/C++/LocalAI/ncm_gguf"
OUT = os.path.join(OUT_DIR, "code_ncm_1b.gguf")
VOCAB = 258
EOS, PAD = 256, 257


def to_array(t):
    a = t.detach().cpu().numpy()
    # F16 for sizable weight matrices. FP16 rounding exceeds the 2e-2 fidelity
    # bound once |x| >= 64, so codebooks and other large or non-finite tensors
    # stay F32.
    if (
        a.ndim >= 2
        and a.size >= 256
        and np.isfinite(a).all()
        and float(np.max(np.abs(a))) < 64.0
    ):
        return np.ascontiguousarray(a.astype(np.float16))
    return np.ascontiguousarray(a.astype(np.float32))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(SRC, "scaled_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    levels = list(cfg["levels"])
    print("[export] building tokenizer", flush=True)
    vqvae = build_tokenizer(
        levels, cfg["tok_embed"], cfg["codebook"], cfg["tok_layers"], cfg["dec_layers"])
    vqvae.load_state_dict(torch.load(os.path.join(SRC, "vqvae.pt"), map_location="cpu"))
    vqvae.eval()

    print("[export] building generator", flush=True)
    net = build_generator(vqvae, cfg["gen_embd"], cfg["gen_layers"], cfg["gen_heads"])
    blob = torch.load(os.path.join(SRC, "generator.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict(blob["model"], strict=True)
    net.eval()
    print(f"[export] checkpoints loaded (train step {blob.get('step')})", flush=True)

    tensors = {}
    for k, v in vqvae.state_dict().items():
        tensors["tokenizer." + k] = v.detach().float().cpu()
    for k, v in net.state_dict().items():
        tensors["generator." + k] = v.detach().float().cpu()
    del net
    drop = ("tokenizer.vq.ema_", "tokenizer.vq.codebook_hit_")
    tensors = {k: v for k, v in tensors.items() if not k.startswith(drop)}
    print(f"[export] {len(tensors)} tensors", flush=True)

    n_tok = sum(v.numel() for k, v in tensors.items() if k.startswith("tokenizer."))
    n_gen = sum(v.numel() for k, v in tensors.items() if k.startswith("generator."))

    writer = gguf.GGUFWriter(OUT, "ncm", use_temp_file=False)
    writer.add_name("code_ncm_1b")
    writer.add_string(
        "description",
        "Byte-level NCM code model, ~1B generator. Multiscale residual VQ-VAE "
        "tokenizer plus coarse-to-fine generator. Container only: llama.cpp has no NCM op.")
    writer.add_uint32("levels_count", len(levels))
    for i, lv in enumerate(levels):
        writer.add_uint32(f"level.{i}", int(lv))
    writer.add_array("codebook_sizes", [int(cfg["codebook"])] * len(levels))
    writer.add_uint32("tokenizer.embed_dim", int(cfg["tok_embed"]))
    writer.add_uint32("tokenizer.in_vocab_size", VOCAB)
    writer.add_uint32("tokenizer.n_layers", int(cfg["tok_layers"]))
    writer.add_uint32("generator.n_embd", int(cfg["gen_embd"]))
    writer.add_uint32("generator.n_layer", int(cfg["gen_layers"]))
    writer.add_uint32("generator.n_head", int(cfg["gen_heads"]))
    writer.add_uint32("generator.rope_base", 100000)
    writer.add_uint32("generator.n_levels_predicted", len(levels) - 1)
    writer.add_uint32("seq_len", int(cfg["seq_len"]))
    writer.add_uint32("eos_token_id", EOS)
    writer.add_uint32("pad_token_id", PAD)
    writer.add_uint32("attn_pattern", 0)
    writer.add_uint64("tokenizer.param_count", int(n_tok))
    writer.add_uint64("generator.param_count", int(n_gen))
    writer.add_uint64("total_param_count", int(n_tok + n_gen))
    writer.add_uint32("train.gen_steps", int(blob.get("step") or 0))

    for name, t in tensors.items():
        writer.add_tensor(name, to_array(t))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[export] wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

    # Round-trip a handful of tensors. Full 1B compare is the same check, sampled.
    print("[export] verifying", flush=True)
    reader = gguf.GGUFReader(OUT)
    ref = {}
    for k, v in tensors.items():
        ref[k] = v.float().numpy().reshape(-1)
    worst = 0.0
    worst_name = ""
    checked = 0
    for t in reader.tensors:
        if t.name not in ref:
            continue
        got = np.asarray(t.data).reshape(-1).astype(np.float32)
        src = ref[t.name]
        if src.shape[0] != got.shape[0]:
            raise SystemExit(f"GGUF fidelity check failed: length mismatch on {t.name}")
        n = min(src.shape[0], 4096)
        delta = src[:n] - got[:n]
        same_inf = np.isinf(src[:n]) & np.isinf(got[:n]) & (np.signbit(src[:n]) == np.signbit(got[:n]))
        delta = np.where(same_inf, 0.0, delta)
        if not np.isfinite(delta).all():
            raise SystemExit(f"GGUF fidelity check failed: non-finite mismatch on {t.name}")
        err = float(np.max(np.abs(delta))) if n else 0.0
        if err > worst:
            worst = err
            worst_name = t.name
        checked += 1
    print(
        f"[export] compared {checked} tensors, max abs diff on first 4096 = {worst:.3e} ({worst_name})",
        flush=True,
    )
    if checked != len(ref) or worst > 2e-2:
        raise SystemExit("GGUF fidelity check failed")


if __name__ == "__main__":
    main()
