"""Draw what the model learned.

    python visualize.py --run out/cpu

Writes three images into the run folder:
    loss.png          training and validation loss, with perplexity on the right axis
    attention.png     attention weights of the last layer for a sentence, one panel per head
    progress.png      the text the model wrote at different training steps
"""

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tinygpt import load_checkpoint  # noqa: E402

BG, FG, GRID = "#111418", "#e6e6e6", "#2a2f36"
TRAIN_C, VAL_C = "#7aa2f7", "#2dd4bf"


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def plot_loss(run: Path):
    hist = json.loads((run / "log.json").read_text())["history"]
    steps = [h["step"] for h in hist]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    style(ax)
    ax.plot(steps, [h["train"] for h in hist], color=TRAIN_C, lw=2, label="train")
    ax.plot(steps, [h["val"] for h in hist], color=VAL_C, lw=2, label="validation")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats per character)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_title("Loss goes down, the text gets real")
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(math.exp(lo), math.exp(hi))
    ax2.set_yscale("log")
    ax2.set_ylabel("perplexity = e^loss", color=FG)
    ax2.tick_params(colors=FG, which="both")
    leg = ax.legend(facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)
    best = min(hist, key=lambda h: h["val"])
    ax.annotate(f"best val {best['val']:.3f}\nppl {math.exp(best['val']):.2f}",
                (best["step"], best["val"]), textcoords="offset points", xytext=(-90, 30),
                color=FG, arrowprops=dict(arrowstyle="->", color=FG))
    fig.tight_layout()
    fig.savefig(run / "loss.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_attention(run: Path, text: str, layer: int):
    model, tok, _ = load_checkpoint(run / "ckpt.pt", "cpu")
    text = "".join(c for c in text if c in tok.stoi)[: model.cfg.block_size]
    idx = torch.tensor([tok.encode(text)])
    model.set_attention_capture(True)
    model(idx, targets=idx)  # targets given so the whole sequence is processed
    att = model.blocks[layer].attn.last_attn[0]  # (heads, T, T)
    model.set_attention_capture(False)

    H = att.size(0)
    cols = 2 if H == 4 else min(H, 3)
    rows = math.ceil(H / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), facecolor=BG, squeeze=False)
    labels = [c if c not in " \n" else ("·" if c == " " else "⏎") for c in text]
    for h in range(rows * cols):
        ax = axes[h // cols][h % cols]
        if h >= H:
            ax.axis("off")
            continue
        style(ax)
        # sqrt makes faint attention visible without hiding the strong spots
        ax.imshow(att[h].sqrt().numpy(), cmap="mako" if "mako" in plt.colormaps() else "viridis")
        ax.set_title(f"head {h}", fontsize=10)
        ax.set_xticks(range(len(text)))
        ax.set_yticks(range(len(text)))
        ax.set_xticklabels(labels, fontsize=6, family="monospace")
        ax.set_yticklabels(labels, fontsize=6, family="monospace")
    fig.suptitle(f"Attention weights, layer {layer if layer >= 0 else len(model.blocks) + layer} "
                 f"(row = character doing the looking, column = what it looks at)",
                 color=FG, fontsize=11)
    fig.tight_layout()
    fig.savefig(run / "attention.png", dpi=150)
    plt.close(fig)


def plot_progress(run: Path, n: int = 4):
    lines = [json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines() if l.strip()]
    if len(lines) > n:  # first, last, and evenly spaced in between
        pick = [round(i * (len(lines) - 1) / (n - 1)) for i in range(n)]
        lines = [lines[i] for i in pick]
    fig, axes = plt.subplots(1, len(lines), figsize=(3.6 * len(lines), 4.2), facecolor=BG)
    if len(lines) == 1:
        axes = [axes]
    for ax, s in zip(axes, lines):
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(VAL_C if s is lines[-1] else GRID)
        body = "\n".join(textwrap.fill(l, 30) for l in s["text"].splitlines()[:12])[:420]
        ax.text(0.04, 0.96, f"step {s['step']}  (val {s['val']:.2f})", color=VAL_C,
                fontsize=9, va="top", transform=ax.transAxes, weight="bold")
        ax.text(0.04, 0.88, body, color=FG, fontsize=7, va="top", family="monospace",
                transform=ax.transAxes)
    fig.suptitle("From random letters to fake Shakespeare", color=FG)
    fig.tight_layout()
    fig.savefig(run / "progress.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "out" / "cpu"))
    ap.add_argument("--text", default="The king is dead, long live the king.")
    ap.add_argument("--layer", type=int, default=-1, help="which layer's attention to draw")
    args = ap.parse_args()
    run = Path(args.run)
    plot_loss(run)
    plot_attention(run, args.text, args.layer)
    plot_progress(run)
    print(f"wrote {run / 'loss.png'}, {run / 'attention.png'}, {run / 'progress.png'}")


if __name__ == "__main__":
    main()
