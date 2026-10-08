# Afrisyl-tokenizer

**A syllable-aware tokenizer for African languages**

Afrisyl-tokenizer is built to fix a core problem: BPE and WordPiece tokenizers break African languages.
Instead of splitting "ndinoda" into ['n', 'din', 'oda'], we respect the natural CV syllable structure of Bantu languages.

Built as part of MSc research by Nkosilomusa Ncube, Alumni WeThinkCode.

## Key Features
- **Syllable-based**: Uses CV, CVC, V patterns found in Shona, Ndebele, Zulu, Swahili, etc
- **Low fertility**: 30% fewer tokens than BPE on Shona text
- **Fast**: Pure Python + regex. No dependencies for inference
- **HF Compatible**: Drop-in replacement for HuggingFace tokenizers
- **Trainable**: Train your own tokenizer on any African language corpus
- **Small vocab**: 8k-16k vocab covers 95%+ of tokens
- **Open Source**: MIT Licensed for research and social impact

## Installation
```bash
pip install afrisyl-tokenizer
```

## Quick Usage

### 1. Load Tokenizer
```python
from afrisyl_tokenizer import AfriSylTokenizer

# Load default shona vocab from package
tok = AfriSylTokenizer(language="shona")

# Or load custom vocab
# tok = AfriSylTokenizer(vocab_path="path/to/ndebele_vocab.json")

# Batch processing
texts = ["ndinoda rubatsiro", "mhoroi shamwari"]

batch = tok.batch_encode(
    texts, 
    add_bos=True, 
    add_eos=True, 
    max_len=32, 
    padding=True
)
# batch['input_ids'] -> [[1, 12, 45, ...], [1, 8, 22, ...]]
# batch['attention_mask'] -> [[1, 1, 1, ...], [1, 1, 1, ...]]

decoded_batch = tok.batch_decode(batch["input_ids"])
print(decoded_batch)
# ['ndinoda rubatsiro', 'mhoroi shamwari']
```

## Whitespace handling
Spaces are tokenized the same way o200k (GPT-4o's tokenizer) handles them:
a space is fused onto the front of the token that follows it (e.g. `" ya"`)
instead of being dropped or turned into its own token between every word.
`decode(encode(text))` is fully reversible, spacing included:

```python
tok = AfriSylTokenizer(language="shona")
ids = tok.encode("nyika yakatanga")
tok.decode(ids) == "nyika yakatanga"  # True
```

Every syllable, fallback letter, digit and punctuation mark also has a
space-prefixed sibling in the vocab (`"ba"` and `" ba"` are both real
tokens), and multi-space/newline runs get their own tokens too -- the same
trick o200k uses so whitespace doesn't need special-casing at decode time.

## Emoji & other scripts
Anything outside the vocab -- emoji, accented letters, other scripts --
falls back to its raw UTF-8 bytes (256 `<0xXX>` byte tokens) instead of a
lossy `<unk>`, the same way byte-level BPE tokenizers like o200k always have
full coverage. This also correctly round-trips multi-codepoint emoji
sequences (ZWJ family emoji, flags, skin-tone modifiers):

```python
tok = AfriSylTokenizer(language="shona")
ids = tok.encode("mhoroi shamwari 😀🇿🇼")
tok.decode(ids) == "mhoroi shamwari 😀🇿🇼"  # True
```
