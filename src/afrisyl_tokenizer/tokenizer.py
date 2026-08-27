import json
from pathlib import Path
from typing import List, Dict, Union, Optional

class AfriSylTokenizer:
    """
    AfriSyl: A Syllable-Aware Tokenizer for African Languages

    Splits text into CV/CVC/CCV/CCVV syllables based on a language-specific vocab.
    Works for any language if you provide a syllable vocab json.

    Example:
        >>> tok = AfriSylTokenizer(language="shona")
        >>> tok.encode("nyika yakatanga")
        [2, 139, 82, 184, 82, 168, 124, 3]
        >>> tok.decode([2, 139, 82, 184, 82, 168, 124, 3])
        'nyikayakatanga'
    """

    def __init__(self, language: str = "shona", vocab_path: Optional[Union[str, Path]] = None):
        """
        Args:
            language: Language code. Used to load vocabs/{language}_vocab.json
            vocab_path: Custom path to vocab json. If None, loads from package.
        """
        if vocab_path is None:
            vocab_path = Path(__file__).parent / "vocabs" / f"{language}_vocab.json"

        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)

        self.vocab: List[str] = data["vocab"]
        self.token_to_id: Dict[str, int] = data["token_to_id"]
        self.id_to_token: Dict[int, str] = {int(k): v for k, v in data["id_to_token"].items()}
        self.vocab_size: int = data["vocab_size"]
        self.language: str = data.get("language", language)
        self.version: str = data.get("version", "0.1.0")

        # Special tokens
        specials = data["special_tokens"]
        self.PAD_TOKEN = specials["pad"]
        self.UNK_TOKEN = specials["unk"]
        self.BOS_TOKEN = specials["bos"]
        self.EOS_TOKEN = specials["eos"]

        self.PAD_ID = self.token_to_id[self.PAD_TOKEN]
        self.UNK_ID = self.token_to_id[self.UNK_TOKEN]
        self.BOS_ID = self.token_to_id[self.BOS_TOKEN]
        self.EOS_ID = self.token_to_id[self.EOS_TOKEN]

        # Tokenization rules
        self._punctuation = ",.!?;:()'[]-"
        self._space_chars = " \t\n\r"

        # Build syllable list: longest first for greedy matching
        syllables = [t for t in self.vocab if not t.startswith("<")]
        self._syllables = sorted(syllables + list(self._punctuation), key=len, reverse=True)

    def _is_skip_char(self, ch: str) -> bool:
        return ch in self._space_chars

    def tokenize(self, text: str) -> List[str]:
        """Convert text to list of syllable tokens."""
        text = text.lower().strip()
        tokens: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if self._is_skip_char(ch):
                i += 1
                continue

            # Greedy match longest syllable first
            matched = False
            for tok in self._syllables:
                if text.startswith(tok, i):
                    tokens.append(tok)
                    i += len(tok)
                    matched = True
                    break

            if not matched:
                tokens.append(self.UNK_TOKEN)
                i += 1
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert text to list of token IDs."""
        ids = [self.token_to_id.get(t, self.UNK_ID) for t in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert list of token IDs back to text."""
        tokens = [self.id_to_token.get(i, self.UNK_TOKEN) for i in ids]
        if skip_special:
            tokens = [t for t in tokens if not t.startswith("<")]
        return "".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
        padding: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Batch encode for PyTorch/TensorFlow training."""
        all_ids = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        seq_len = max_len or max(len(ids) for ids in all_ids)

        input_ids, attention_mask = [], []
        for ids in all_ids:
            ids = ids[:seq_len]
            mask = [1] * len(ids)
            if padding and len(ids) < seq_len:
                pad_len = seq_len - len(ids)
                ids += [self.PAD_ID] * pad_len
                mask += [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, batch_ids: List[List[int]], skip_special: bool = True) -> List[str]:
        """Batch decode list of ID sequences."""
        return [self.decode(ids, skip_special) for ids in batch_ids]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"AfriSylTokenizer(lang={self.language}, vocab_size={self.vocab_size}, version={self.version})"
