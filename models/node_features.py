"""
models/node_features.py
─────────────────────────────────────────────────────────────────────────────
Builds the initial node feature matrix X for the GCN.
Papers → SciBERT [CLS] embedding of their abstract (replaces averaged word embeddings)
Words  → SciBERT embedding of the word token (for RPCG-1)
Authors→ mean of their papers' SciBERT embeddings  (for RPCG-2)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch

from models.embedding_backend import EmbeddingBackend

logger = logging.getLogger(__name__)


def build_paper_embeddings(
    abstracts:  List[str],
    backend:    EmbeddingBackend,
    batch_size: int          = 32,
    max_len:    int          = 512,
) -> np.ndarray:
    """
    Returns (N, d) float32 array of mean-pooled paper embeddings using
    the frozen backbone described in Section 4.1.4 (Llama-3.2-3B in the
    paper; see models/embedding_backend.py for the real-vs-offline
    backend selection). One row per abstract; empty abstracts -> zero-ish
    vector via the "[UNK]" placeholder used inside the backend.
    """
    logger.info(f"Building paper embeddings with backend={backend.name} ...")
    embeds = backend.encode(abstracts, batch_size=batch_size, max_len=max_len)
    logger.info(f"Paper embeddings: {embeds.shape}")
    return embeds.astype(np.float32)


def build_word_embeddings(
    word_id_map: Dict[str, int],
    backend:     EmbeddingBackend,
    batch_size:  int          = 256,
) -> np.ndarray:
    """
    Returns (V, d) float32 array of word-level embeddings from the same
    backend used for paper embeddings (Eq. 5 in the paper).
    """
    words = list(word_id_map.keys())
    logger.info(f"Building word embeddings for {len(words)} vocab words ...")
    embeds = backend.encode_words(words, batch_size=batch_size)
    logger.info(f"Word embeddings: {embeds.shape}")
    return embeds.astype(np.float32)


def build_author_embeddings(
    paper_embeds:  np.ndarray,
    paper_authors: List[List[int]],
    num_authors:   int,
) -> np.ndarray:
    """
    Returns (Na, d) float32 — each author is the mean of their papers' embeddings.
    """
    d        = paper_embeds.shape[1]
    embeds   = np.zeros((num_authors, d), dtype=np.float32)
    counts   = np.zeros(num_authors, dtype=np.float32)

    for p_idx, a_list in enumerate(paper_authors):
        for a in a_list:
            embeds[a] += paper_embeds[p_idx]
            counts[a] += 1

    nonzero = counts > 0
    embeds[nonzero] = embeds[nonzero] / counts[nonzero, None]
    logger.info(f"Author embeddings: {embeds.shape}")
    return embeds


def combine_features_rpcg1(
    paper_embeds: np.ndarray,
    word_embeds:  np.ndarray,
) -> np.ndarray:
    """Stack paper and word embeddings: (N+V, d)"""
    return np.vstack([paper_embeds, word_embeds]).astype(np.float32)


def combine_features_rpcg2(
    paper_embeds:  np.ndarray,
    author_embeds: np.ndarray,
) -> np.ndarray:
    """Stack paper and author embeddings: (N+Na, d)"""
    return np.vstack([paper_embeds, author_embeds]).astype(np.float32)
