"""Minimal trainer stub for Afrisyl-tokenizer."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

from .utils import save_vocab


class AfriSylTrainer:
    """
    Train a syllable-aware vocab from a corpus.
    Very minimal implementation: extracts CV/CVC syllables via regex
    and selects most frequent.
    """

    def __init__(self, language: str = "shona", vocab_size: int = 8000):
        self.language = language
        self.vocab_size = vocab_size

    def train(self, corpus_path: str | Path, output_path: Optional[str | Path] = None) -> Dict:
        corpus_path = Path(corpus_path)
        text = corpus_path.read_text(encoding="utf-8", errors="ignore").lower()
        # Simple syllable regex: CV pattern
        pattern = r"(?:[bcdfghjklmnpqrstvwxyz]+[aeiou]+n?|n|m|[aeiou])"
        tokens = re.findall(pattern, text)
        freq = Counter(tokens)
        specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        vocab = specials + [tok for tok, _ in freq.most_common(self.vocab_size - len(specials))]
        data = {
            "language": self.language,
            "version": "0.1.0",
            "vocab": vocab,
            "token_to_id": {tok: i for i, tok in enumerate(vocab)},
            "id_to_token": {str(i): tok for i, tok in enumerate(vocab)},
            "vocab_size": len(vocab),
            "special_tokens": {"pad": "<pad>", "unk": "<unk>", "bos": "<bos>", "eos": "<eos>"},
        }
        if output_path:
            save_vocab(data, output_path)
        return data
