"""Generate text from a trained checkpoint.

    python sample.py                                   # 500 chars after "ROMEO:"
    python sample.py --prompt "JULIET:" --temperature 0.6 --top-k 20
    python sample.py --interactive                     # type prompts, get continuations
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tinygpt import get_device, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "out" / "cpu" / "ckpt.pt"))
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--chars", type=int, default=500, help="how many characters to generate")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = always pick the top char")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cache", action="store_true", help="disable the KV cache (slower)")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = get_device(args.device)
    if not Path(args.ckpt).exists():
        sys.exit(f"No checkpoint at {args.ckpt}. Train one first:  python train.py")
    model, tok, ck = load_checkpoint(args.ckpt, device)
    print(f"loaded {args.ckpt} (step {ck['iter']}, val loss {ck['best_val']:.3f})\n")

    def run(prompt: str):
        missing = sorted({c for c in prompt if c not in tok.stoi})
        if missing:
            print(f"(skipping characters the model has never seen: {missing})")
            prompt = "".join(c for c in prompt if c in tok.stoi)
        prompt = prompt or "\n"
        for i in range(args.num_samples):
            idx = torch.tensor([tok.encode(prompt)], device=device)
            print(prompt, end="", flush=True)
            model.generate(idx, args.chars, temperature=args.temperature, top_k=args.top_k,
                           top_p=args.top_p, use_cache=not args.no_cache,
                           stream=lambda t: print(tok.itos[t], end="", flush=True))
            print("\n" + ("=" * 60 if i < args.num_samples - 1 else ""))

    if args.interactive:
        print("Type a prompt and press Enter (Ctrl-C to quit). Try a character name like 'HAMLET:'.")
        while True:
            try:
                p = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            run(p.replace("\\n", "\n") + ("\n" if p.isupper() and p.endswith(":") else ""))
    else:
        run(args.prompt.replace("\\n", "\n"))


if __name__ == "__main__":
    main()
