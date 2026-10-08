import json
from pathlib import Path
from typing import List, Dict, Union, Optional

class AfriSylTokenizer:
    """
    AfriSyl: A Syllable-Aware Tokenizer for African Languages

    Splits text into CV/CVC/CCV/CCVV syllables based on a language-specific vocab.
    Works for any language if you provide a syllable vocab json.

    Whitespace is tokenized the way o200k (GPT-4o's tokenizer) treats it: a
    space is baked onto the front of the following token (e.g. " ya") rather
    than being dropped or emitted as its own token between every word. This
    makes encode/decode fully reversible.

    Example:
        >>> tok = AfriSylTokenizer(language="shona")
        >>> ids = tok.encode("nyika yakatanga")
        >>> ids
        [133, 198, 112, 477, 112, 175, 133, 86]
        >>> tok.decode(ids)
        'nyika yakatanga'
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
        self.version: str = data.get("version", "0.1.2")

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
        self._special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}

        # Byte-fallback table: "<0x00>".."<0xFF>", one token per raw byte.
        # Anything not covered by a real vocab entry (emoji, accented
        # letters, other scripts, ...) falls back to its UTF-8 bytes instead
        # of a lossy <unk>, the same way byte-level BPE tokenizers like
        # o200k always have full coverage. This means <unk> should now be
        # essentially unreachable for real text.
        self._byte_token_value: Dict[str, int] = {}
        for b in range(256):
            bt = f"<0x{b:02X}>"
            if bt in self.token_to_id:
                self._byte_token_value[bt] = b
        self._has_byte_fallback = len(self._byte_token_value) == 256

        # Build match list: every real vocab entry (syllables, fallback letters,
        # digits, punctuation, whitespace atoms, and their space-prefixed
        # siblings), longest first so greedy matching prefers the longest
        # known token -- this is what lets a token like " ba" (space + "ba")
        # win over matching " " and "ba" separately, the same way o200k bakes
        # a leading space into the following token instead of emitting a
        # standalone space token between every pair of words.
        self._match_tokens = sorted(
            (t for t in self.vocab if not t.startswith("<")),
            key=len,
            reverse=True,
        )
        # Fast first-character dispatch so we don't scan the whole list for
        # every position.
        self._by_first_char: Dict[str, List[str]] = {}
        for tok in self._match_tokens:
            self._by_first_char.setdefault(tok[0], []).append(tok)

    def _byte_fallback(self, ch: str) -> Optional[List[str]]:
        """Represent one character as its UTF-8 byte tokens, if the vocab
        has the byte-fallback table loaded."""
        if not self._has_byte_fallback:
            return None
        return [f"<0x{b:02X}>" for b in ch.encode("utf-8")]

    def tokenize(self, text: str) -> List[str]:
        """Convert text to list of tokens.

        Whitespace is treated the way o200k (and GPT-2/GPT-4 style BPE
        tokenizers) treat it: a space is not stripped or skipped, it is
        matched as part of the vocabulary like any other character, so a
        single leading space naturally fuses onto the token that follows it
        (e.g. " mba" rather than " " + "mba"). This keeps tokenize/encode
        fully reversible -- decode just concatenates the tokens back
        together and the original spacing reappears.

        Any character outside the vocab (emoji, accented letters, other
        scripts, ...) falls back to its raw UTF-8 bytes rather than a lossy
        <unk>, so tokenize/decode stays reversible for arbitrary text too.
        """
        text = text.lower()
        tokens: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            candidates = self._by_first_char.get(ch)

            matched = False
            if candidates:
                for tok in candidates:  # already longest-first
                    if text.startswith(tok, i):
                        tokens.append(tok)
                        i += len(tok)
                        matched = True
                        break

            if not matched:
                byte_toks = self._byte_fallback(ch)
                if byte_toks:
                    tokens.extend(byte_toks)
                else:
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
        """Convert list of token IDs back to text.

        Runs of byte-fallback tokens (see tokenize/_byte_fallback) are
        regrouped and UTF-8 decoded back into the original character(s) --
        this also correctly reconstructs multi-codepoint emoji sequences
        (ZWJ emoji, flags, skin-tone modifiers), since UTF-8 byte
        concatenation is associative across codepoints.
        """
        tokens = [
            self.id_to_token.get(i, self.UNK_TOKEN)
            for i in ids
            if not (skip_special and i in self._special_ids)
        ]

        pieces: List[str] = []
        byte_buf = bytearray()
        for t in tokens:
            bval = self._byte_token_value.get(t)
            if bval is not None:
                byte_buf.append(bval)
                continue
            if byte_buf:
                pieces.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf = bytearray()
            pieces.append(t)
        if byte_buf:
            pieces.append(byte_buf.decode("utf-8", errors="replace"))

        return "".join(pieces)

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
