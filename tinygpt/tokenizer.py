"""Character-level tokenizer: one integer per distinct character in the text."""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        unknown = sorted({c for c in s if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in the vocabulary: {unknown!r}")
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text())["chars"])
