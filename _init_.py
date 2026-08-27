"""
Afrisyl-tokenizer: A syllable-aware tokenizer for African languages

Author: Nkosilomusa Ncube
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Nkosilomusa Ncube"
__email__ = "nkosilomusa955@gmail.com"
__license__ = "MIT"

from .tokenizer import AfriSylTokenizer
from .trainer import AfriSylTrainer
from .utils import load_corpus, save_vocab