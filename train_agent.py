"""8-hour NCM agent trainer.

87M FP32 generator, fused Adam on the GPU, frozen 256-byte tokenizer.
Continuation pretrain, then a mix of continuation and think/tool turns.
Writes ncm_gguf/code_ncm_q8.gguf. Does not touch the older GGUF files.
"""
import json, math, os, random, sys, time, traceback, types, zlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, "D:/C++/LocalAI")
sys.path.insert(0, "D:/C++/LocalAI/ncm")
from train_scaled import build_tokenizer
from models.next_concept import NextConceptTransformer

CORPUS = "D:/C++/LocalAI/code_corpus.txt"
TOK_CKPT = "D:/C++/LocalAI/code_artifacts_scaled/vqvae.pt"
OUT = "D:/C++/LocalAI/code_artifacts_agent"
GGUF_PATH = "D:/C++/LocalAI/ncm_gguf/code_ncm_q8.gguf"
LEVELS = [4, 16, 32, 64, 128, 256]
EOS, PAD = 256, 257
BATCH = 8
EMBD, LAYERS, HEADS = 768, 12, 12
N_CODES = sum(LEVELS)
DEV = "cuda"
BUDGET_S = 8 * 3600
EXPORT_TAIL_S = 20 * 60
FINETUNE_S = 90 * 60
PROBE_STEPS = 500
WARMUP = 500


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "train_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state_path():
    return os.path.join(OUT, "state.json")


def load_state():
    if os.path.isfile(state_path()):
        with open(state_path(), encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    tmp = state_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_path())


def build_generator(vqvae, prefix_len):
    k = len(vqvae.vq.levels)
    cfg = types.SimpleNamespace(
        n_embd=EMBD, n_layer=LAYERS, n_head=HEADS, bias=True, dropout=0.0,
        vocab_size=[vqvae.vq.vqvae_vocab_sizes[i] for i in range(1, k)],
        attn_pattern="block_diagonal", prefix_len=int(prefix_len), raw_token_prefix=True,
        prefix_pad_token_id=PAD, use_positional_encoding=True, use_level_encoding=True,
        use_rope=True, rope_base=100000, use_qk_norm=True,
        ctx_encoder=OmegaConf.create({"enabled": False}),
        use_flex_attention=False, query_level0=False,
        tie_heads_to_codebook=True, tie_heads_learn_scale=True,
        predict_level0=True,
        post_upsample_conv=OmegaConf.create({"enabled": False}),
        shared_output_head=False, use_swiglu=False,
    )
    return NextConceptTransformer(vqvae_local=vqvae, config=cfg)


def load_vq():
    vq = build_tokenizer(LEVELS, 512, 1024, 8, 2).to(DEV).eval()
    vq.load_state_dict(torch.load(TOK_CKPT, map_location=DEV))
    for p in vq.parameters():
        p.requires_grad_(False)
    return vq


def make_opt(net, lr):
    return torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95), fused=True)


def code_weight():
    w = torch.ones(N_CODES, device=DEV)
    w[: LEVELS[0]] = 4.0
    return w


def parse_files(path):
    raw = open(path, "rb").read()
    files = []
    for block in raw.split(b"# FILE: "):
        nl = block.find(b"\n")
        if nl <= 0:
            continue
        name = block[:nl].decode("utf-8", "replace").strip()
        body = block[nl + 1 :]
        if name and body:
            files.append((name, body))
    return files


def left_pad(raw, width):
    raw = raw[-width:]
    out = np.full(width, PAD, dtype=np.int16)
    if len(raw):
        out[-len(raw):] = np.frombuffer(raw, dtype=np.uint8)
    return out


def code_mask_from_valid(valid):
    parts = []
    n = valid.shape[0]
    for lv in LEVELS:
        span = 256 // lv
        parts.append(valid.reshape(n, lv, span).any(axis=2))
    return np.concatenate(parts, axis=1)


def json_tool(obj):
    return json.dumps(obj, ensure_ascii=True)


def pack_target(text):
    raw = text.encode("utf-8")
    if len(raw) > 256 or len(raw) < 16:
        return None
    valid = np.zeros(256, dtype=np.uint8)
    valid[: len(raw)] = 1
    target = np.full(256, PAD, dtype=np.int16)
    target[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return target, valid


def make_tool_rows(train_files, prefix_len, limit=12000):
    rows = []
    commands = (b"cmake", b"pytest", b"python", b"msbuild", b"pip ")
    for name, body in train_files:
        if len(rows) >= limit:
            break
        if len(body) < 40:
            continue
        window = body[: min(len(body), max(40, prefix_len - 40))]
        base = name.replace("\\", "/").split("/")[-1]
        task_read = f"Read {base}.\n{name}\n".encode("utf-8", "replace")
        prefix = left_pad(window + b"\n" + task_read, prefix_len)
        line = b""
        for ln in window.splitlines():
            if 8 <= len(ln) <= 70 and all(32 <= c < 127 for c in ln):
                line = ln
                break
        shown = line.decode("ascii", "replace")[:48] if line else base
        read_txt = f"<think>\n{base}: {shown}\n</think>\n<tool>\n{json_tool({'name':'read','path':name})}\n</tool>\n"
        packed = pack_target(read_txt)
        if packed is not None:
            rows.append((prefix.copy(), packed[0], packed[1], 1))
        lines = [ln for ln in window.splitlines() if 8 <= len(ln) <= 60 and all(32 <= c < 127 for c in ln)]
        if len(lines) >= 2:
            old, new = lines[0].decode("ascii"), lines[1].decode("ascii")
            task_edit = f"Edit {base}.\n{name}\n".encode("utf-8", "replace")
            edit_txt = (
                f"<think>\n{base}: replace a shown line.\n</think>\n<tool>\n"
                f"{json_tool({'name':'edit','path':name,'old':old,'new':new})}\n</tool>\n"
            )
            packed = pack_target(edit_txt)
            if packed is not None and old.encode("ascii") in window and new.encode("ascii") in window:
                pref = left_pad(window + b"\n" + task_edit, prefix_len)
                rows.append((pref, packed[0], packed[1], 1))
        for cmd in commands:
            if cmd in window:
                task_run = f"Run the build for {base}.\n{name}\n".encode("utf-8", "replace")
                cmd_s = cmd.decode("ascii").strip()
                run_txt = (
                    f"<think>\n{base}: run {cmd_s}.\n</think>\n<tool>\n"
                    f"{json_tool({'name':'run','cmd':cmd_s})}\n</tool>\n"
                )
                packed = pack_target(run_txt)
                if packed is not None:
                    pref = left_pad(window + b"\n" + task_run, prefix_len)
                    rows.append((pref, packed[0], packed[1], 1))
                break
        result = window[:80]
        task_stop = f"Stop if the read is enough.\n{name}\n".encode("utf-8", "replace")
        stop_txt = f"<think>\n{base}: done.\n</think>\n<tool>\n{json_tool({'name':'stop'})}\n</tool>\n"
        packed = pack_target(stop_txt)
        if packed is not None:
            blob = b"result:\n" + result + b"\n" + task_stop
            pref = left_pad(blob, prefix_len)
            rows.append((pref, packed[0], packed[1], 1))
    return rows


def build_arrays(files, prefix_len):
    train_files, val_files = [], []
    for name, body in files:
        if (zlib_crc(name) % 20) == 0:
            val_files.append((name, body))
        else:
            train_files.append((name, body))
    log(f"[data] files train={len(train_files)} val={len(val_files)}")

    def count_windows(group):
        n = 0
        for _, body in group:
            if len(body) >= 64:
                n += (len(body) - 64) // 256 + 1
        return n

    n_train = count_windows(train_files)
    n_val = count_windows(val_files)
    tools = make_tool_rows(train_files, prefix_len)
    log(f"[data] continuation train={n_train} val={n_val} tools={len(tools)}")
    n_all = n_train + len(tools)

    def alloc(n):
        return {
            "prefix": np.empty((n, prefix_len), dtype=np.int16),
            "target": np.empty((n, 256), dtype=np.int16),
            "valid": np.empty((n, 256), dtype=np.uint8),
            "kind": np.zeros(n, dtype=np.uint8),
        }

    train = alloc(n_all)
    val = alloc(max(n_val, 1))

    def fill_fast(group, dest, start):
        i = start
        for _, body in group:
            pos = 0
            while pos < len(body):
                chunk = body[pos : pos + 256]
                if len(chunk) < 64:
                    break
                dest["target"][i, :] = PAD
                dest["valid"][i, :] = 0
                dest["target"][i, : len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
                dest["valid"][i, : len(chunk)] = 1
                pre = body[max(0, pos - prefix_len) : pos]
                dest["prefix"][i] = left_pad(pre, prefix_len)
                dest["kind"][i] = 0
                i += 1
                pos += 256
                if i >= dest["target"].shape[0]:
                    return i
        return i

    filled = fill_fast(train_files, train, 0)
    for prefix, target, valid, kind in tools:
        if filled >= n_all:
            break
        train["prefix"][filled] = prefix
        train["target"][filled] = target
        train["valid"][filled] = valid
        train["kind"][filled] = kind
        filled += 1
    train = {k: v[:filled] for k, v in train.items()}
    vfilled = fill_fast(val_files, val, 0)
    val = {k: v[:vfilled] for k, v in val.items()}
    if vfilled == 0:
        val = {k: v[:64].copy() for k, v in train.items()}
    prompts = make_prompts(val_files, prefix_len)
    all_names = sorted({n.replace("\\", "/") for n, _ in files})
    return train, val, prompts, all_names


def zlib_crc(name):
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def make_prompts(val_files, prefix_len):
    prompts = []
    for name, body in val_files:
        if len(prompts) >= 20 or len(body) < 80:
            continue
        window = body[:180]
        base = name.replace("\\", "/").split("/")[-1]
        if len(prompts) % 2 == 0:
            task = f"Read {base}.\n{name}\n".encode("utf-8", "replace")
        else:
            task = f"Edit {base}.\n{name}\n".encode("utf-8", "replace")
        prefix = left_pad(window + b"\n" + task, prefix_len)
        prompts.append({
            "path": name,
            "prefix": prefix.tolist(),
            "result": list(body[:120]),
        })
    return prompts


@torch.no_grad()
def encode_targets(vq, targets):
    codes = np.empty((targets.shape[0], N_CODES), dtype=np.int16)
    bs = 128
    t0 = time.perf_counter()
    for i in range(0, targets.shape[0], bs):
        batch = torch.from_numpy(targets[i : i + bs].astype(np.int64)).to(DEV)
        idx = vq.encode(batch)
        codes[i : i + batch.shape[0]] = idx.detach().cpu().numpy().astype(np.int16)
        if i % (bs * 40) == 0:
            rate = (i + batch.shape[0]) / max(1e-6, time.perf_counter() - t0)
            log(f"[encode] {i + batch.shape[0]}/{targets.shape[0]} {rate:.0f} seq/s")
    return codes


def step_batch(net, opt, vq, prefix, codes, mask, weight, train):
    cond = vq.vq.idx_to_ncm_input(codes)
    z_pre = vq.encoder.wte(prefix)
    x = torch.cat([z_pre, cond], 1)
    logits = net(x, prefix_pad_mask=(prefix == PAD))
    if logits.shape[1] != codes.shape[1]:
        raise RuntimeError(f"logit length {logits.shape[1]} != codes {codes.shape[1]}")
    flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), codes.reshape(-1), reduction="none")
    w = mask * weight
    denom = w.sum().clamp(min=1.0)
    loss = (flat * w.reshape(-1)).sum() / denom
    if train:
        opt.zero_grad(set_to_none=True)
        if bool(torch.isfinite(loss).item()):
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if bool(torch.isfinite(grad_norm).item()):
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
        loss = loss.detach()
    return loss


def take(arr_idx, prefix_np, codes_np, mask_np, n):
    sel = arr_idx[torch.randint(0, len(arr_idx), (n,)).numpy()]
    prefix = torch.from_numpy(prefix_np[sel].astype(np.int64)).to(DEV)
    codes = torch.from_numpy(codes_np[sel].astype(np.int64)).to(DEV)
    mask = torch.from_numpy(mask_np[sel].astype(np.float32)).to(DEV)
    return prefix, codes, mask


def fit_prefix(vq):
    chosen = None
    for plen in (64, 128):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net = build_generator(vq, plen).to(DEV)
        opt = make_opt(net, 1e-3)
        weight = code_weight()
        try:
            times = []
            loss_v = None
            for i in range(6):
                prefix = torch.randint(0, 256, (BATCH, plen), device=DEV)
                codes = torch.randint(0, 1024, (BATCH, N_CODES), device=DEV)
                mask = torch.ones(BATCH, N_CODES, device=DEV)
                t0 = time.perf_counter()
                loss_v = step_batch(net, opt, vq, prefix, codes, mask, weight, True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if i >= 2:
                    times.append(dt)
            dt = sum(times) / len(times)
            mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            tps = BATCH * N_CODES / dt
            log(f"[fit] prefix={plen} {dt:.2f}s {tps:.0f} code-tok/s {mem:.0f} MiB loss={float(loss_v):.3f}")
            with torch.no_grad():
                probe = torch.randint(0, 256, (1, plen), device=DEV)
                idx = net.generate(None, probe, temperature=0, top_k=1, top_p=1, use_kv_cache=False)
            log(f"[fit] generate {tuple(idx.shape)}")
            ok = math.isfinite(float(loss_v)) and mem < 6500 and tps >= 3000
            if ok:
                chosen = plen
        except RuntimeError as exc:
            log(f"[fit] prefix={plen} failed {exc.__class__.__name__}")
        del net, opt
        torch.cuda.empty_cache()
    if chosen is None:
        chosen = 64
        log("[fit] no prefix cleared the bar, using 64")
    log(f"[fit] chosen prefix_len={chosen}")
    return chosen


def current_lr(step, elapsed, span, base):
    if step < WARMUP:
        return base * (step + 1) / WARMUP
    if span <= 0 or elapsed <= 0.8 * span:
        return base
    u = min(1.0, (elapsed - 0.8 * span) / max(1e-6, 0.2 * span))
    return base * 0.5 * (1.0 + math.cos(math.pi * u))


def set_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


@torch.no_grad()
def val_loss(net, vq, val, weight):
    net.eval()
    total, n = 0.0, 0
    idx = np.arange(len(val["kind"]))
    for _ in range(4):
        prefix, codes, mask = take(idx, val["prefix"], val["codes"], val["mask"], BATCH)
        loss = step_batch(net, None, vq, prefix, codes, mask, weight, False)
        if torch.isfinite(loss):
            total += float(loss)
            n += 1
    net.train()
    return total / max(1, n)


def save_ckpt(name, net, opt, step, phase, lr):
    path = os.path.join(OUT, name)
    torch.save({
        "model": net.state_dict(),
        "opt": None if opt is None else opt.state_dict(),
        "step": step,
        "phase": phase,
        "lr": lr,
    }, path)
    log(f"[ckpt] {name} step={step} phase={phase}")


def load_model(vq, prefix_len, lr, ckpt_name, with_opt):
    net = build_generator(vq, prefix_len).to(DEV)
    opt = make_opt(net, lr) if with_opt else None
    path = os.path.join(OUT, ckpt_name)
    if os.path.isfile(path):
        blob = torch.load(path, map_location=DEV, weights_only=False)
        net.load_state_dict(blob["model"], strict=True)
        if with_opt and opt is not None and blob.get("opt") is not None:
            opt.load_state_dict(blob["opt"])
        return net, opt, int(blob.get("step") or 0)
    return net, opt, 0


def decode_bytes(vq, idx):
    logits = vq.decode(idx)
    ids = logits.argmax(dim=-1)[0].tolist()
    out = []
    for b in ids:
        if b == PAD:
            break
        out.append(b if b < 256 else 32)
    return bytes(out).decode("utf-8", "replace")


def parse_sample(text):
    think_a = text.find("<think>")
    think_b = text.find("</think>")
    tool_a = text.find("<tool>")
    tool_b = text.find("</tool>")
    if min(think_a, think_b, tool_a, tool_b) < 0:
        return None
    if not (think_a < think_b < tool_a < tool_b):
        return None
    if text.count("<think>") != 1 or text.count("<tool>") != 1:
        return None
    think = text[think_a + 7 : think_b].strip()
    if not think:
        return None
    try:
        obj = json.loads(text[tool_a + 6 : tool_b])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("name") not in {"read", "edit", "run", "stop"}:
        return None
    return think, obj


def score_prompts(net, vq, prompts, names, drafts):
    net.eval()
    name_set = {n.replace("\\", "/") for n in names}
    parsed = 0
    copied = 0
    think_ok = 0
    samples = []
    for prompt in prompts:
        prefix = torch.tensor([prompt["prefix"]], dtype=torch.long, device=DEV)
        raw = bytes(b for b in prompt["prefix"] if b != PAD)
        hit = None
        for _ in range(drafts):
            idx = net.generate(None, prefix, temperature=0.3, top_k=40, top_p=0.9, use_kv_cache=False)
            text = decode_bytes(vq, idx)
            got = parse_sample(text)
            if got is not None:
                hit = (text, got)
                break
        if hit is None:
            samples.append("")
            continue
        text, (think, obj) = hit
        samples.append(text)
        parsed += 1
        base = prompt["path"].replace("\\", "/").split("/")[-1]
        if base.lower() in think.lower():
            think_ok += 1
        path = str(obj.get("path", "")).replace("\\", "/")
        old = str(obj.get("old", ""))
        ok_copy = True
        if obj["name"] in {"read", "edit"}:
            ok_copy = path in name_set
        if obj["name"] == "edit":
            ok_copy = ok_copy and old != "" and old in raw.decode("utf-8", "replace")
        if obj["name"] == "run":
            ok_copy = str(obj.get("cmd", "")) in raw.decode("utf-8", "replace")
        if ok_copy:
            copied += 1
    net.train()
    n = max(1, len(prompts))
    return {"parsed": parsed, "n": len(prompts), "copied": copied, "think": think_ok,
            "parse_rate": parsed / n, "sample": samples[0] if samples else ""}


def export_q8(vq, net, step, prefix_len, dest=None):
    import gguf
    dest = dest or GGUF_PATH
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tensors = {}
    for k, v in vq.state_dict().items():
        tensors["tokenizer." + k] = v.detach().float().cpu()
    for k, v in net.state_dict().items():
        tensors["generator." + k] = v.detach().float().cpu()
    tensors = {k: v for k, v in tensors.items()
               if not k.startswith(("tokenizer.vq.ema_", "tokenizer.vq.codebook_hit_"))}
    writer = gguf.GGUFWriter(dest, "ncm", use_temp_file=False)
    writer.add_name("code_ncm_q8")
    writer.add_string("description", "NCM agent generator, Q8_0 inference weights. Python runtime only.")
    writer.add_uint32("levels_count", len(LEVELS))
    for i, lv in enumerate(LEVELS):
        writer.add_uint32(f"level.{i}", int(lv))
    writer.add_array("codebook_sizes", [1024] * len(LEVELS))
    writer.add_uint32("tokenizer.embed_dim", 512)
    writer.add_uint32("tokenizer.in_vocab_size", 258)
    writer.add_uint32("generator.n_embd", EMBD)
    writer.add_uint32("generator.n_layer", LAYERS)
    writer.add_uint32("generator.n_head", HEADS)
    writer.add_uint32("generator.prefix_len", int(prefix_len))
    writer.add_uint32("generator.predict_level0", 1)
    writer.add_uint32("seq_len", 256)
    writer.add_uint32("eos_token_id", EOS)
    writer.add_uint32("pad_token_id", PAD)
    writer.add_uint32("train.gen_steps", int(step))
    worst = 0.0
    for name, tensor in tensors.items():
        arr = np.ascontiguousarray(tensor.numpy().astype(np.float32))
        q8 = arr.ndim >= 2 and arr.shape[-1] % 32 == 0 and np.isfinite(arr).all()
        if q8:
            q = gguf.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            back = gguf.dequantize(q, gguf.GGMLQuantizationType.Q8_0).astype(np.float32)
            err = float(np.max(np.abs(arr.reshape(-1)[:4096] - back.reshape(-1)[:4096])))
            worst = max(worst, err)
        else:
            writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log(f"[export] {dest} {os.path.getsize(dest)/1e6:.1f} MB max_abs_sample={worst:.4f}")


def prepare_data(vq, prefix_len):
    data_path = os.path.join(OUT, f"data_p{prefix_len}.npz")
    prompt_path = os.path.join(OUT, "prompts.json")
    if os.path.isfile(data_path) and os.path.isfile(prompt_path):
        log(f"[data] loading {data_path}")
        blob = np.load(data_path)
        with open(prompt_path, encoding="utf-8") as f:
            meta = json.load(f)
        train = {k: blob[k] for k in ("prefix", "codes", "mask", "kind")}
        val = {k: blob["val_" + k] for k in ("prefix", "codes", "mask", "kind")}
        return train, val, meta["prompts"], meta["names"]
    files = parse_files(CORPUS)
    train_raw, val_raw, prompts, names = build_arrays(files, prefix_len)
    log("[data] encoding train targets")
    train_codes = encode_targets(vq, train_raw["target"])
    log("[data] encoding val targets")
    val_codes = encode_targets(vq, val_raw["target"])
    train_mask = code_mask_from_valid(train_raw["valid"]).astype(np.uint8)
    val_mask = code_mask_from_valid(val_raw["valid"]).astype(np.uint8)
    np.savez(
        data_path,
        prefix=train_raw["prefix"], codes=train_codes, mask=train_mask, kind=train_raw["kind"],
        val_prefix=val_raw["prefix"], val_codes=val_codes, val_mask=val_mask, val_kind=val_raw["kind"],
    )
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts, "names": names}, f)
    train = {"prefix": train_raw["prefix"], "codes": train_codes, "mask": train_mask, "kind": train_raw["kind"]}
    val = {"prefix": val_raw["prefix"], "codes": val_codes, "mask": val_mask, "kind": val_raw["kind"]}
    log(f"[data] saved {data_path} train={len(train['kind'])} val={len(val['kind'])}")
    return train, val, prompts, names


def probe_lr(vq, prefix_len, train, val):
    weight = code_weight()
    cont = np.flatnonzero(train["kind"] == 0)
    best_lr, best_val = 1e-3, 1e9
    for lr in (3e-4, 1e-3, 3e-3):
        torch.cuda.empty_cache()
        net = build_generator(vq, prefix_len).to(DEV).train()
        opt = make_opt(net, lr)
        last = None
        t0 = time.perf_counter()
        for step in range(PROBE_STEPS):
            prefix, codes, mask = take(cont, train["prefix"], train["codes"], train["mask"], BATCH)
            last = step_batch(net, opt, vq, prefix, codes, mask, weight, True)
            if step % 100 == 0:
                log(f"[probe] lr={lr:g} step={step} loss={float(last):.4f}")
            if not torch.isfinite(last):
                log(f"[probe] lr={lr:g} non-finite at {step}")
                last = None
                break
        score = 1e9
        if last is not None and torch.isfinite(last):
            score = val_loss(net, vq, val, weight)
        log(f"[probe] lr={lr:g} val={score:.4f} {(time.perf_counter()-t0):.0f}s")
        if score < best_val:
            best_lr, best_val = lr, score
        del net, opt
        torch.cuda.empty_cache()
    log(f"[probe] chosen lr={best_lr:g} val={best_val:.4f}")
    return best_lr


def train_loop(vq, prefix_len, base_lr, train, val, prompts, names, state, deadline):
    export_at = deadline - EXPORT_TAIL_S
    finetune_at = export_at - FINETUNE_S
    weight = code_weight()
    cont = np.flatnonzero(train["kind"] == 0)
    tool = np.flatnonzero(train["kind"] == 1)
    phase = state.get("phase", "pretrain")
    if phase not in {"pretrain", "finetune"}:
        phase = "pretrain"
    net, opt, step = load_model(vq, prefix_len, base_lr, "last.pt", True)
    if step == 0 and os.path.isfile(os.path.join(OUT, "last.pt")):
        step = int(state.get("step") or 0)
    train_t0 = state.get("train_t0") or time.time()
    state["train_t0"] = train_t0
    state["phase"] = phase
    save_state(state)
    net.train()
    best_val = float(state.get("best_val") or 1e9)
    best_parse = float(state.get("best_parse") or -1)
    last_side = time.time()
    log(f"[train] phase={phase} step={step} lr={base_lr:g} until export {export_at-time.time():.0f}s")
    while time.time() < export_at:
        if phase == "pretrain" and time.time() >= finetune_at:
            phase = "finetune"
            state["phase"] = phase
            save_ckpt("pretrain_end.pt", net, None, step, phase, base_lr)
            log("[train] switching to mixed finetune")
        use_tool = phase == "finetune" and len(tool) > 0 and random.random() < 0.5
        pool = tool if use_tool else cont
        elapsed = time.time() - train_t0
        span = max(1.0, export_at - train_t0)
        lr = current_lr(step, elapsed, span, base_lr)
        set_lr(opt, lr)
        prefix, codes, mask = take(pool, train["prefix"], train["codes"], train["mask"], BATCH)
        loss = step_batch(net, opt, vq, prefix, codes, mask, weight, True)
        step += 1
        if step % 20 == 0:
            log(f"[{phase}] step={step} loss={float(loss):.4f} lr={lr:.2e}")
        if not torch.isfinite(loss):
            log(f"[{phase}] non-finite loss at step {step}")
            del loss
            torch.cuda.empty_cache()
            continue
        now = time.time()
        if now - last_side >= 1800:
            v = val_loss(net, vq, val, weight)
            log(f"[{phase}] val={v:.4f}")
            if phase == "pretrain" and v < best_val:
                best_val = v
                state["best_val"] = best_val
                save_ckpt("best_pretrain.pt", net, None, step, phase, lr)
            if phase == "finetune":
                scored = score_prompts(net, vq, prompts, names, 1)
                log(f"[finetune] parse={scored['parsed']}/{scored['n']} val={v:.4f}")
                better = scored["parse_rate"] > best_parse and v < 6.93
                if better:
                    best_parse = scored["parse_rate"]
                    state["best_parse"] = best_parse
                    state["best_parse_val"] = v
                    save_ckpt("best_tool.pt", net, None, step, phase, lr)
            save_ckpt("last.pt", net, opt, step, phase, lr)
            state["step"] = step
            state["phase"] = phase
            save_state(state)
            last_side = now
    save_ckpt("last.pt", net, opt, step, phase, base_lr)
    state["step"] = step
    state["phase"] = "export"
    save_state(state)
    return net, step


def pick_export_net(vq, prefix_len, state):
    best_val = float(state.get("best_val") or 1e9)
    tool_path = os.path.join(OUT, "best_tool.pt")
    pre_path = os.path.join(OUT, "best_pretrain.pt")
    choice = "last.pt"
    if os.path.isfile(tool_path):
        tool_val = float(state.get("best_parse_val") or 1e9)
        if tool_val < 6.93 and tool_val <= best_val + 0.15:
            choice = "best_tool.pt"
    if choice == "last.pt" and os.path.isfile(pre_path):
        choice = "best_pretrain.pt"
    log(f"[export] checkpoint {choice}")
    net, _, step = load_model(vq, prefix_len, 1e-3, choice, False)
    return net, step


def run_checks(net, vq, prompts, names):
    scored = score_prompts(net, vq, prompts, names, 4)
    follow = "no read to follow"
    net.eval()
    for prompt in prompts:
        prefix = torch.tensor([prompt["prefix"]], dtype=torch.long, device=DEV)
        idx = net.generate(None, prefix, temperature=0.3, top_k=40, top_p=0.9, use_kv_cache=False)
        got = parse_sample(decode_bytes(vq, idx))
        if got is None or got[1].get("name") != "read":
            continue
        nxt = left_pad(bytes(prompt["result"]) + b"\nStop if the read is enough.\n", len(prompt["prefix"]))
        prefix2 = torch.tensor(nxt.reshape(1, -1), dtype=torch.long, device=DEV)
        idx2 = net.generate(None, prefix2, temperature=0.3, top_k=40, top_p=0.9, use_kv_cache=False)
        text2 = decode_bytes(vq, idx2)
        got2 = parse_sample(text2)
        follow = text2 if got2 is None else f"{got2[1].get('name')}: {text2}"
        break
    report = (
        f"parsed {scored['parsed']}/{scored['n']}\n"
        f"think_names_file {scored['think']}/{scored['n']}\n"
        f"copied_args {scored['copied']}/{scored['n']}\n"
        f"sample:\n{scored['sample']}\n"
        f"followup:\n{follow}\n"
    )
    with open(os.path.join(OUT, "check_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    log("[check]\n" + report)


def long_tokens(text):
    found, cur = [], []
    for ch in text:
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if len(cur) >= 6:
                found.append("".join(cur))
            cur = []
    if len(cur) >= 6:
        found.append("".join(cur))
    return found


def log_greedy(net, vq, prompts):
    net.eval()
    hits = 0
    lines = []
    with torch.no_grad():
        for prompt in prompts[:4]:
            prefix = torch.tensor([prompt["prefix"]], dtype=torch.long, device=DEV)
            idx = net.generate(None, prefix, temperature=0, top_k=1, top_p=1, use_kv_cache=False)
            text = decode_bytes(vq, idx).replace("\n", "\\n")
            raw = bytes(b for b in prompt["prefix"] if b != PAD).decode("utf-8", "replace")
            hit = any(tok.lower() in text.lower() for tok in long_tokens(raw))
            hits += int(hit)
            lines.append(f"  {prompt['path']} hit={int(hit)} {text[:220]}")
    net.train()
    log(f"[cont] samples {hits}/4\n" + "\n".join(lines))
    return hits


def cont_lr(session_step, elapsed, span, base):
    if session_step < 200:
        return base * (session_step + 1) / 200
    if elapsed <= 0.8 * span:
        return base
    u = min(1.0, (elapsed - 0.8 * span) / max(1e-6, 0.2 * span))
    return base * 0.5 * (1.0 + math.cos(math.pi * u))


def continue_training(hours=3.0):
    os.makedirs(OUT, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.backends.cudnn.benchmark = True
    base_lr = 3e-4
    meta_path = os.path.join(OUT, "cont_state.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    if not meta.get("deadline") or meta.get("phase") == "done":
        meta = {"deadline": time.time() + hours * 3600, "phase": "cont", "best_val": 1.816153198480606, "session_step": 0}
    deadline = float(meta["deadline"])
    log(f"[cont] budget {deadline - time.time():.0f}s from best_pretrain lr={base_lr:g}")
    vq = load_vq()
    prefix_len = 128
    train, val, prompts, _names = prepare_data(vq, prefix_len)
    cont = np.flatnonzero(train["kind"] == 0)
    weight = code_weight()
    ckpt = "cont_last.pt" if os.path.isfile(os.path.join(OUT, "cont_last.pt")) and meta.get("phase") == "cont" else "best_pretrain.pt"
    net, opt, step = load_model(vq, prefix_len, base_lr, ckpt, True)
    if ckpt == "best_pretrain.pt":
        opt = make_opt(net, base_lr)
        meta["session_step"] = 0
        meta["origin_step"] = step
    session_step = int(meta.get("session_step") or 0)
    origin = int(meta.get("origin_step") or step)
    best_val = float(meta.get("best_val") or 1.816153198480606)
    t0 = time.time() - (session_step * 1.1)
    span = max(1.0, deadline - (time.time() - session_step * 1.1))
    if ckpt == "best_pretrain.pt":
        t0 = time.time()
        span = max(1.0, deadline - t0)
    net.train()
    log(f"[cont] loaded {ckpt} step={step} session_step={session_step}")
    last_side = time.time()
    log_greedy(net, vq, prompts)
    while time.time() < deadline:
        elapsed = time.time() - t0
        lr = cont_lr(session_step, elapsed, span, base_lr)
        set_lr(opt, lr)
        prefix, codes, mask = take(cont, train["prefix"], train["codes"], train["mask"], BATCH)
        loss = step_batch(net, opt, vq, prefix, codes, mask, weight, True)
        step += 1
        session_step += 1
        if session_step % 20 == 0:
            log(f"[cont] step={step} loss={float(loss):.4f} lr={lr:.2e}")
        if not torch.isfinite(loss):
            log(f"[cont] non-finite loss at step {step}")
            del loss
            torch.cuda.empty_cache()
            continue
        if time.time() - last_side >= 1800:
            v = val_loss(net, vq, val, weight)
            hits = log_greedy(net, vq, prompts)
            log(f"[cont] val={v:.4f} id_hits={hits}/4")
            if v < best_val:
                best_val = v
                save_ckpt("cont_best.pt", net, None, step, "cont", lr)
            save_ckpt("cont_last.pt", net, opt, step, "cont", lr)
            meta.update({"phase": "cont", "step": step, "session_step": session_step,
                         "best_val": best_val, "origin_step": origin, "deadline": deadline})
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            last_side = time.time()
    save_ckpt("cont_last.pt", net, opt, step, "cont", base_lr)
    winner = "cont_best.pt" if os.path.isfile(os.path.join(OUT, "cont_best.pt")) else "best_pretrain.pt"
    log(f"[cont] export {winner} best_val={best_val:.4f}")
    net, _, step = load_model(vq, prefix_len, base_lr, winner, False)
    net.eval()
    export_q8(vq, net, step, prefix_len, dest=os.path.join(os.path.dirname(GGUF_PATH), "code_ncm_q8_cont.gguf"))
    log_greedy(net, vq, prompts)
    meta["phase"] = "done"
    meta["best_val"] = best_val
    meta["step"] = step
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    log("[cont] done")


def main():
    os.makedirs(OUT, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    state = load_state()
    if "deadline" not in state:
        state["deadline"] = time.time() + BUDGET_S
        save_state(state)
    deadline = float(state["deadline"])
    log(f"[start] budget {deadline - time.time():.0f}s left")
    vq = load_vq()
    if "prefix_len" not in state:
        state["prefix_len"] = fit_prefix(vq)
        save_state(state)
    prefix_len = int(state["prefix_len"])
    train, val, prompts, names = prepare_data(vq, prefix_len)
    if "lr" not in state:
        if time.time() > deadline - EXPORT_TAIL_S - FINETUNE_S:
            state["lr"] = 1e-3
            log("[probe] skipped, using 1e-3")
        else:
            state["lr"] = probe_lr(vq, prefix_len, train, val)
        save_state(state)
    if state.get("phase") != "done":
        net, step = train_loop(vq, prefix_len, float(state["lr"]), train, val, prompts, names, state, deadline)
        del net
        torch.cuda.empty_cache()
        net, step = pick_export_net(vq, prefix_len, state)
        net.eval()
        export_q8(vq, net, step, prefix_len)
        run_checks(net, vq, prompts, names)
        state["phase"] = "done"
        state["step"] = step
        save_state(state)
    log("[done]")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "continue":
            continue_training(3.0)
        else:
            main()
    except Exception:
        traceback.print_exc()
        raise
