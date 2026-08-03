"""
utils/text_processing.py
─────────────────────────────────────────────────────────────────────────────
Text cleaning, tokenization, and vocabulary building.
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


def download_nltk_resources() -> None:
    resources = [
        "stopwords", "wordnet", "punkt", "punkt_tab",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    ]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass


download_nltk_resources()

_lemmatizer  = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))
_custom_stop = {"paper", "research", "study", "approach", "propose", "result",
                 "method", "using", "based", "show", "work", "present", "model"}
_ABBREVS     = {
    "e.g.": "for example", "i.e.": "that is",
    "etc.": "and so on",   "vs.":  "versus",
}


def clean_str(text: str, nouns_only: bool = True) -> str:
    """
    Clean and normalise a raw string.
    Returns a space-joined, lower-cased string of lemmatised (noun) tokens.
    """
    if not isinstance(text, str) or text.strip().lower() in ("nan", "none", ""):
        return ""

    for abbr, repl in _ABBREVS.items():
        text = text.replace(abbr, repl)

    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"[0-9]+", " ", text)             # digits
    text = re.sub(r"-", " ", text)                  # hyphens
    text = re.sub(r"[^A-Za-z\s]", " ", text)        # punctuation
    text = re.sub(r"\s{2,}", " ", text)              # extra spaces
    # Expand contractions
    for pat, repl in [("'s", " is"), ("'ve", " have"), ("n't", " not"),
                      ("'re", " are"), ("'d", " would"), ("'ll", " will")]:
        text = text.replace(pat, repl)

    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens
              if t.lower() not in _stop_words
              and t.lower() not in _custom_stop
              and t.isalpha()]

    if nouns_only:
        tagged  = nltk.pos_tag(tokens)
        tokens  = [t for t, pos in tagged if pos.startswith("NN")]

    tokens = [_lemmatizer.lemmatize(t, pos="n") for t in tokens]
    return " ".join(tokens)


def build_vocabulary(
    sentences: List[str],
    freq_limit: int = 3,
) -> tuple[Dict[str, int], Dict[str, int]]:
    """
    Build word→id mapping from cleaned sentences.
    Returns (word_id_map, word_freq).
    """
    word_freq: Dict[str, int] = {}
    for s in sentences:
        for w in clean_str(s).split():
            word_freq[w] = word_freq.get(w, 0) + 1

    vocab = [w for w, c in word_freq.items() if c >= freq_limit]
    word_id_map = {w: i for i, w in enumerate(vocab)}
    return word_id_map, word_freq


def tokenize_corpus(
    sentences: List[str],
    word_id_map: Dict[str, int],
    word_freq: Dict[str, int],
    freq_limit: int = 3,
) -> List[List[str]]:
    """
    Convert raw sentences to lists of filtered, cleaned tokens.
    """
    tokenized = []
    for s in sentences:
        cleaned  = clean_str(s)
        filtered = [w for w in cleaned.split()
                    if w in word_id_map and word_freq.get(w, 0) >= freq_limit]
        tokenized.append(filtered)
    return tokenized
