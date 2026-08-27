"""
Afrisyl-tokenizer: A syllable-aware tokenizer for African languages

Author: Nkosilomusa Ncube
License: MIT
"""

__version__ = "0.1.1"
__author__ = "Nkosilomusa Ncube"
__email__ = "nkosilomusa955@gmail.com"
__license__ = "MIT"

from .tokenizer import AfriSylTokenizer

try:
    from .trainer import AfriSylTrainer
except ImportError:  # pragma: no cover
    AfriSylTrainer = None  # type: ignore

try:
    from .utils import load_corpus, save_vocab
except ImportError:  # pragma: no cover
    load_corpus = None  # type: ignore
    save_vocab = None  # type: ignore

__all__ = ["AfriSylTokenizer", "AfriSylTrainer", "load_corpus", "save_vocab"]
