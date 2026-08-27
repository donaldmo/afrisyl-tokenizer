README.md

```
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

###Batch processing
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
# ['ndinodarubatsiro', 'mhoroshamwari']