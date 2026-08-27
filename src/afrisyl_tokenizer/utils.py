"""Utils for Afrisyl-tokenizer."""

import json
from pathlib import Path
from typing import Dict, List


def load_corpus(path: str | Path) -> List[str]:
    p = Path(path)
    return [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]


def save_vocab(data: Dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
