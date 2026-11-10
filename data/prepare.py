"""Download Tiny Shakespeare (~1.1M characters), build the character vocabulary,
and write train.bin / val.bin as arrays of uint16 token ids.

    python data/prepare.py                 # Tiny Shakespeare
    python data/prepare.py --input my.txt  # any plain-text file of your own
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tinygpt.tokenizer import CharTokenizer  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="use this text file instead of downloading Shakespeare")
    ap.add_argument("--out", default=str(ROOT / "data" / "shakespeare"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        raw = out / "input.txt"
        if not raw.exists():
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, raw)
        text = raw.read_text(encoding="utf-8")

    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.uint16)
    n_val = int(len(ids) * args.val_frac)
    # The last 10% of the text is held out, so validation is on unseen passages.
    ids[:-n_val].tofile(out / "train.bin")
    ids[-n_val:].tofile(out / "val.bin")
    tok.save(out / "vocab.json")

    print(f"{len(text):,} characters, vocabulary of {tok.vocab_size}: {''.join(tok.chars)!r}")
    print(f"train: {len(ids) - n_val:,} tokens   val: {n_val:,} tokens   -> {out}")


if __name__ == "__main__":
    main()
