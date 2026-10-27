"""Native FP16 throughput/util probe: does a model that FITS in VRAM (no CPU
offload) actually saturate the GTX 1070 Ti, unlike the ~25% we saw for the
offloaded 1B? Builds a plain ~400M GPT-2 stack, FP16 mixed-precision AdamW."""
import torch, torch.nn as nn, time

dev = "cuda"
torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, bias=bias, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x, tok_emb):
        # causal mask handled by attention; use relative via simple additive
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))
        return x


n_embd, n_layer, n_head = 1280, 24, 16
L, B = 256, 8
vocab = 258
model = nn.ModuleDict(dict(
    wte=nn.Embedding(vocab, n_embd),
    wpe=nn.Embedding(L, n_embd),
    h=nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]),
    ln_f=nn.LayerNorm(n_embd),
)).to(dev).half()

n_params = sum(p.numel() for p in model.parameters())
print(f"params = {n_params/1e6:.1f}M", flush=True)

# causal mask
mask = torch.tril(torch.ones(L, L, device=dev, dtype=torch.bool)).view(1, 1, L, L)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

x = torch.randint(0, vocab, (B, L), device=dev)
target = torch.randint(0, vocab, (B, L), device=dev)

# warmup
for _ in range(3):
    em = model.wte(x) + model.wpe(torch.arange(L, device=dev))
    for blk in model.h:
        em = blk(em, em)
    logits = model.ln_f(em)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
print("warmup done", flush=True)

iters = 2000
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for _ in range(iters):
    em = model.wte(x) + model.wpe(torch.arange(L, device=dev))
    for blk in model.h:
        em = blk(em, em)
    logits = model.ln_f(em)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"native FP16: {iters} iters in {dt:.2f}s = {iters/dt:.1f} steps/s")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
