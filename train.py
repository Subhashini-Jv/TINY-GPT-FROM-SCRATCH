"""Train the tiny GPT.

    python train.py                       # 'cpu' preset: ~12 minutes on a laptop
    python train.py --preset mps          # Apple Silicon GPU
    python train.py --preset gpu          # NVIDIA GPU, the full 10M-parameter model
    python train.py --max-iters 500 --n-layer 2   # any setting can be overridden

Writes into out/<run-name>/:
    ckpt.pt        best checkpoint by validation loss
    log.json       loss, perplexity and learning rate over time
    samples.jsonl  text the model wrote at each evaluation (watch it learn)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tinygpt import GPT, GPTConfig, CharTokenizer, get_device  # noqa: E402

PRESETS = {
    # ~0.8M params. Runs on any laptop CPU in roughly 10-15 minutes (val loss ~1.58).
    "cpu": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1,
                batch_size=32, max_iters=3000, lr=2e-3, warmup_iters=100, eval_interval=250),
    # ~3M params. Apple Silicon (M1/M2/M3) GPU, roughly 10-15 minutes.
    "mps": dict(n_layer=6, n_head=6, n_embd=192, block_size=256, dropout=0.15,
                batch_size=32, max_iters=4000, lr=1.5e-3, warmup_iters=200, eval_interval=250),
    # ~10.6M params. A single NVIDIA GPU, a few minutes. Reaches val loss ~1.47.
    "gpu": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2,
                batch_size=64, max_iters=5000, lr=1e-3, warmup_iters=200, eval_interval=500),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="cpu")
    ap.add_argument("--data", default=str(ROOT / "data" / "shakespeare"))
    ap.add_argument("--out", default=None, help="output folder (default out/<preset>)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true", help="continue from out/.../ckpt.pt")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    ap.add_argument("--prompt", default="ROMEO:", help="prompt for the progress samples")
    # Everything below overrides the preset when given.
    for name, typ in [("n_layer", int), ("n_head", int), ("n_embd", int), ("block_size", int),
                      ("dropout", float), ("batch_size", int), ("max_iters", int), ("lr", float),
                      ("warmup_iters", int), ("eval_interval", int)]:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="final lr as a fraction of peak")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-iters", type=int, default=100)
    ap.add_argument("--sample-chars", type=int, default=300)
    args = ap.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


class Batcher:
    """Picks random windows of block_size+1 characters from a token file.
    x is the window minus its last character, y is the window minus its first,
    so at every position the target is simply the next character."""

    def __init__(self, path: Path, block_size: int, batch_size: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.bs, self.B, self.device = block_size, batch_size, device

    def get(self):
        ix = np.random.randint(0, len(self.data) - self.bs - 1, size=self.B)
        x = np.stack([self.data[i:i + self.bs] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.bs] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)


def lr_at(it: int, args) -> float:
    """Linear warmup, then cosine decay down to min_lr."""
    min_lr = args.lr * args.min_lr_frac
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = min(1.0, (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - min_lr)


@torch.no_grad()
def estimate_loss(model, batchers, eval_iters, ctx):
    model.eval()
    out = {}
    for split, b in batchers.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = b.get()
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    data = Path(args.data)
    if not (data / "train.bin").exists():
        sys.exit(f"No data in {data}. Run:  python data/prepare.py")
    tok = CharTokenizer.load(data / "vocab.json")
    out = Path(args.out or ROOT / "out" / args.preset)
    out.mkdir(parents=True, exist_ok=True)

    # bfloat16 autocast on CUDA roughly doubles speed; CPU and MPS stay in float32.
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(args.lr, args.weight_decay)

    start_iter, best_val, log = 0, float("inf"), []
    if args.resume and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_iter, best_val = ck["iter"] + 1, ck["best_val"]
        if (out / "log.json").exists():
            log = json.loads((out / "log.json").read_text())["history"]
        print(f"resumed from iter {ck['iter']} (best val {best_val:.4f})")
    elif (out / "samples.jsonl").exists():
        (out / "samples.jsonl").unlink()

    train_model = torch.compile(model) if args.compile else model
    batchers = {"train": Batcher(data / "train.bin", cfg.block_size, args.batch_size, device),
                "val": Batcher(data / "val.bin", cfg.block_size, args.batch_size, device)}
    prompt_ids = torch.tensor([tok.encode(args.prompt)], device=device)

    print(f"device {device} | {model.num_params() / 1e6:.2f}M params | preset {args.preset} | "
          f"{args.max_iters} iters x {args.batch_size} x {cfg.block_size} chars")
    print(f"a model that guesses uniformly would score loss ln({tok.vocab_size}) = "
          f"{math.log(tok.vocab_size):.2f}")

    def save(it):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "model_config": cfg.to_dict(), "chars": tok.chars, "iter": it,
                    "best_val": best_val, "args": vars(args)}, out / "ckpt.pt")

    t0 = time.time()
    x, y = batchers["train"].get()
    for it in range(start_iter, args.max_iters + 1):
        lr = lr_at(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(train_model, batchers, args.eval_iters, ctx)
            ppl = math.exp(losses["val"])
            sample = tok.decode(model.generate(prompt_ids, args.sample_chars, temperature=0.8,
                                               top_k=None)[0].tolist())
            elapsed = time.time() - t0
            print(f"\nstep {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"val ppl {ppl:6.2f} | lr {lr:.2e} | {elapsed:6.1f}s")
            print("-" * 60 + "\n" + sample.strip()[:200] + "\n" + "-" * 60)
            log.append({"step": it, "train": losses["train"], "val": losses["val"],
                        "val_ppl": ppl, "lr": lr, "time": elapsed})
            (out / "log.json").write_text(json.dumps({"config": cfg.to_dict(), "history": log}, indent=1))
            with open(out / "samples.jsonl", "a") as f:
                f.write(json.dumps({"step": it, "val": losses["val"], "text": sample}) + "\n")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(it)
            if it == args.max_iters:
                break

        with ctx:
            _, loss = train_model(x, y)
        x, y = batchers["train"].get()  # fetch the next batch while the GPU works
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if it % 50 == 0:
            print(f"step {it:5d} | loss {loss.item():.4f}", end="\r", flush=True)

    print(f"\ndone in {time.time() - t0:.0f}s. best val loss {best_val:.4f} "
          f"(perplexity {math.exp(best_val):.2f}). checkpoint: {out / 'ckpt.pt'}")
    print(f"next:  python sample.py --ckpt {out / 'ckpt.pt'}   |   python visualize.py --run {out}")


if __name__ == "__main__":
    main()
