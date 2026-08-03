"""
models/embedding_backend.py
─────────────────────────────────────────────────────────────────────────────
Unified text-embedding backend for CiteAware.

Two interchangeable implementations behind one interface:

  1. TransformerBackend  — the backbone actually described in the paper
     (SciBERT for CTC labeling, Llama-3.2-3B for node/LLM-branch features).
     This is what must be used to reproduce the numbers reported in the
     paper. Requires internet access to the HuggingFace Hub (or local
     cached weights) and, for Llama-3.2-3B, a HF access token + GPU.

  2. OfflineTfidfBackend — a network-free substitute used ONLY when the
     transformer backbone cannot be loaded (e.g. no internet access to
     huggingface.co). It mean-approximates "semantic embedding" with
     TF-IDF + TruncatedSVD projected to the same hidden size, so the rest
     of the pipeline (graph construction, GCN, fusion, metrics) is
     completely unchanged. Results produced with this backend are NOT the
     paper's reported numbers and must be clearly labeled as such in any
     write-up.

get_embedding_backend() auto-selects (1) and transparently falls back to
(2) with a loud warning if the transformer cannot be initialised. This
makes the SAME train.py deployable as-is on a GPU machine with HF access
(true CTC + true Llama features) and still runnable end-to-end for
pipeline validation in network-restricted environments.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    """Common interface every backend must implement."""

    name: str = "base"
    hidden_size: int = 0

    def encode(self, texts: List[str], batch_size: int = 32, max_len: int = 512) -> np.ndarray:
        raise NotImplementedError

    def encode_words(self, words: List[str], batch_size: int = 256) -> np.ndarray:
        raise NotImplementedError


class TransformerBackend(EmbeddingBackend):
    """
    Real backbone as specified in the paper. Attention-mask-aware mean
    pooling over the last hidden state (Eq. 8 / Eq. 17 in the paper).
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        logger.info(f"[TransformerBackend] loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.name = model_name

    def _mean_pool(self, texts: List[str], batch_size: int, max_len: int) -> np.ndarray:
        torch = self._torch
        out_arr = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            batch = [t if isinstance(t, str) and t.strip() else "[UNK]"
                     for t in texts[start:start + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (res.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            out_arr[start:start + len(batch)] = pooled.cpu().numpy()
        return out_arr

    def encode(self, texts: List[str], batch_size: int = 32, max_len: int = 512) -> np.ndarray:
        return self._mean_pool(texts, batch_size, max_len)

    def encode_words(self, words: List[str], batch_size: int = 256) -> np.ndarray:
        return self._mean_pool(words, batch_size, max_len=16)


class OfflineTfidfBackend(EmbeddingBackend):
    """
    Network-free substitute: TF-IDF (word 1-2 grams) -> TruncatedSVD to a
    fixed hidden size, L2-normalised. Deterministic, CPU-only, no downloads.

    NOT a substitute for SciBERT/Llama semantics — used only to validate
    the pipeline end-to-end and to produce non-fabricated numbers when the
    real backbones are unreachable.
    """

    def __init__(self, hidden_size: int = 256, seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._TfidfVectorizer = TfidfVectorizer
        self.hidden_size = hidden_size
        self.seed = seed
        self.name = f"offline-tfidf-svd{hidden_size}"
        self._fitted = False

    def _fit_if_needed(self, corpus: List[str]):
        from sklearn.decomposition import TruncatedSVD
        if self._fitted:
            return
        self._vec = self._TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        X = self._vec.fit_transform([t if isinstance(t, str) else "" for t in corpus])
        k = min(self.hidden_size, max(2, X.shape[1] - 1), max(2, X.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self._svd.fit(X)
        self._k = k
        self._fitted = True

    def encode(self, texts: List[str], batch_size: int = 32, max_len: int = 512) -> np.ndarray:
        self._fit_if_needed(texts)
        X = self._vec.transform([t if isinstance(t, str) else "" for t in texts])
        Z = self._svd.transform(X).astype(np.float32)
        if self._k < self.hidden_size:
            pad = np.zeros((Z.shape[0], self.hidden_size - self._k), dtype=np.float32)
            Z = np.hstack([Z, pad])
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return Z / norm

    def encode_words(self, words: List[str], batch_size: int = 256) -> np.ndarray:
        # represent each word by TF-IDF/SVD of itself as a 1-token document,
        # using the already-fitted vocabulary/SVD from encode()
        if not self._fitted:
            self._fit_if_needed(words)
        X = self._vec.transform(words)
        Z = self._svd.transform(X).astype(np.float32)
        if self._k < self.hidden_size:
            pad = np.zeros((Z.shape[0], self.hidden_size - self._k), dtype=np.float32)
            Z = np.hstack([Z, pad])
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return Z / norm


def get_embedding_backend(model_name: str, device: str = "cpu",
                           fallback_hidden_size: int = 256) -> EmbeddingBackend:
    """
    Try to load the real transformer backbone named `model_name`
    (e.g. "allenai/scibert_scivocab_uncased" or
    "meta-llama/Llama-3.2-3B-Instruct"). If unreachable (no internet to
    the HF Hub, no cached weights, missing gated-model token, etc.),
    fall back to the offline TF-IDF/SVD backend and log a clear warning.
    """
    try:
        return TransformerBackend(model_name, device=device)
    except Exception as e:
        logger.warning(
            f"Could not load transformer backbone '{model_name}' ({type(e).__name__}: {e}). "
            f"Falling back to OfflineTfidfBackend. Results produced with this run are NOT "
            f"the paper's SciBERT/Llama numbers — see supplementary methodology note."
        )
        return OfflineTfidfBackend(hidden_size=fallback_hidden_size)
