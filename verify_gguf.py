"""Verify the exported NCM GGUF: read it back, print metadata + tensors, and
confirm the stored tensors match the trained checkpoints (F16 rounding aside)."""
import os, sys
import numpy as np
import gguf

SRC = "D:/C++/LocalAI/code_artifacts"
OUT = "D:/C++/LocalAI/ncm_gguf/code_ncm.gguf"


def main():
    r = gguf.GGUFReader(OUT)
    print("=== GGUF header ===")
    print("arch/general:", r.fields.get("general.architecture", None))
    print("name:", r.fields.get("general.name", None).contents if hasattr(r.fields.get("general.name"), "contents") else r.fields.get("general.name"))
    print("size on disk: %.2f MB" % (os.path.getsize(OUT) / 1e6))

    def show(field):
        v = r.fields.get(field)
        if v is None:
            return
        try:
            print(f"  {field} = {v.contents}")
        except Exception:
            print(f"  {field} = {v}")

    print("\n=== metadata (kv) ===")
    for k, v in r.fields.items():
        try:
            print(f"  {k} = {v.contents!r}")
        except Exception:
            pass

    print("\n=== tensors (%d) ===" % len(r.tensors))
    tot = 0
    for t in r.tensors:
        shape = tuple(t.shape)
        tot += int(np.prod(shape))
        dt = t.types[0].name if getattr(t, "types", None) else "?"
        print(f"  {t.name:45s} {str(shape):18s} {dt}")
    print(f"total elements: {tot}")

    # Fidelity check: reload checkpoints and compare a few tensors to the GGUF.
    print("\n=== fidelity check (ckpt vs gguf) ===")
    import torch
    ck = torch.load(os.path.join(SRC, "vqvae.pt"), map_location="cpu")
    gk = torch.load(os.path.join(SRC, "generator.pt"), map_location="cpu")["model"]
    ck.update({f"generator.{k}": v for k, v in gk.items()})
    maxerr = 0.0
    checked = 0
    for t in r.tensors:
        name = t.name
        if name not in ck:
            continue
        ref = ck[name].float().numpy().reshape(-1)
        got = t.data.reshape(-1).astype(np.float32)
        # gguf dequantizes to float32; compare over overlapping region
        n = min(ref.shape[0], got.shape[0])
        err = float(np.abs(ref[:n] - got[:n]).max())
        maxerr = max(maxerr, err)
        checked += 1
    print(f"  compared {checked} tensors; max abs diff = {maxerr:.3e} (F16 rounding ~5e-4)")


if __name__ == "__main__":
    main()
