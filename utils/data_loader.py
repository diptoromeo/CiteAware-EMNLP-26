"""
utils/data_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads the four citation JSON datasets and builds:
  - RPCG-1: Paper-Word-Citation heterogeneous graph
  - RPCG-2: Paper-Author-Citation heterogeneous graph
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from utils.text_processing import clean_str, build_vocabulary, tokenize_corpus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  JSON Loader
# ─────────────────────────────────────────────

def load_dataset(json_path: str) -> List[dict]:
    """Load a citation dataset JSON and return list of paper dicts."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records from {json_path}")
    return data


def extract_corpus(
    scholar_data: List[dict],
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]], List[int]]:
    """
    Extract from raw JSON:
      - abstracts      : all abstract strings (main + citing), INTERLEAVED
                         as Main_0, Cite_0_0, Cite_0_1, ..., Main_1, Cite_1_0, ...
      - titles         : all title strings (same order)
      - node_ids       : unique string id for each abstract node (same order)
      - citation_edges : list of (citing_id, cited_id) directed pairs
      - main_positions : main_positions[i] = index of main paper i within
                         `abstracts`/`node_ids`/`titles` (NOT simply i,
                         because each main paper is followed immediately
                         by its own citing-paper nodes, so main papers do
                         NOT sit at positions 0..num_papers-1).

    Any code that builds per-main-paper features/labels/predictions MUST
    index through `main_positions`, not through raw range(num_papers).
    """
    abstracts: List[str]          = []
    titles:    List[str]          = []
    node_ids:  List[str]          = []
    citation_edges: List[Tuple]   = []
    main_positions: List[int]     = []

    for idx, entry in enumerate(scholar_data):
        main_abstract = entry.get("original_csv_abstract", "")
        main_title    = entry.get("original_csv_title", "")
        main_id       = f"Main_{idx}"

        if not isinstance(main_abstract, str) or main_abstract.strip().lower() in ("nan", "none", ""):
            main_abstract = ""

        main_positions.append(len(abstracts))   # position BEFORE appending
        abstracts.append(main_abstract)
        titles.append(main_title if isinstance(main_title, str) else "")
        node_ids.append(main_id)

        citing_articles = entry.get("citing_articles", []) or []
        for j, article in enumerate(citing_articles):
            cite_abstract = article.get("abstract", "")
            cite_title    = article.get("title", "")
            cite_id       = f"Cite_{idx}_{j}"

            if not isinstance(cite_abstract, str) or cite_abstract.strip().lower() in ("nan", "none", ""):
                cite_abstract = ""

            abstracts.append(cite_abstract)
            titles.append(cite_title if isinstance(cite_title, str) else "")
            node_ids.append(cite_id)
            citation_edges.append((cite_id, main_id))   # citing → cited

    logger.info(
        f"  Papers: {len(scholar_data)}  |  Total nodes: {len(abstracts)}  "
        f"|  Citation edges: {len(citation_edges)}"
    )
    return abstracts, titles, node_ids, citation_edges, main_positions


# ─────────────────────────────────────────────
#  PMI helpers
# ─────────────────────────────────────────────

def compute_pmi_edges(
    tokenized: List[List[str]],
    word_id_map: Dict[str, int],
    window_size: int = 20,
) -> Dict[Tuple[int, int], float]:
    """
    Compute positive PMI for word-word edges using a sliding window.
    Returns {(wi, wj): pmi_score, ...} only for positive PMI pairs.
    """
    vocab_size   = len(word_id_map)
    windows_total = 0
    word_count    = defaultdict(int)
    pair_count    = defaultdict(int)

    for tokens in tokenized:
        for i in range(len(tokens)):
            window = tokens[i: i + window_size]
            windows_total += 1
            unique = set(window)
            for w in unique:
                if w in word_id_map:
                    word_count[word_id_map[w]] += 1
            for a, b in [(word_id_map[x], word_id_map[y])
                         for x in unique for y in unique
                         if x != y and x in word_id_map and y in word_id_map]:
                pair_count[(min(a, b), max(a, b))] += 1

    pmi_edges: Dict[Tuple[int, int], float] = {}
    for (i, j), cnt in pair_count.items():
        p_ij = cnt / windows_total
        p_i  = word_count[i] / windows_total
        p_j  = word_count[j] / windows_total
        if p_i > 0 and p_j > 0:
            pmi = math.log(p_ij / (p_i * p_j))
            if pmi > 0:
                pmi_edges[(i, j)] = pmi
    return pmi_edges


# ─────────────────────────────────────────────
#  RPCG-1 — Paper-Word-Citation Graph
# ─────────────────────────────────────────────

def build_rpcg1(
    abstracts: List[str],
    node_ids: List[str],
    citation_edges: List[Tuple[str, str]],
    window_size: int = 20,
    freq_limit: int = 3,
) -> Tuple[sp.csr_matrix, np.ndarray, int]:
    """
    Build RPCG-1 adjacency matrix.
    Nodes = papers + words.
    Edges = paper-word (TF-IDF) + word-word (PPMI) + paper-paper (citation count).

    Returns:
      adj_norm   : normalised sparse adjacency (N+V, N+V)
      tfidf_mat  : (N, V) dense TF-IDF matrix for paper node initialisation
      num_papers : N
    """
    num_papers = len(abstracts)
    word_id_map, word_freq = build_vocabulary(abstracts, freq_limit)
    tokenized              = tokenize_corpus(abstracts, word_id_map, word_freq, freq_limit)
    vocab_size             = len(word_id_map)
    total_nodes            = num_papers + vocab_size

    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    logger.info(f"RPCG-1 | Papers={num_papers}  Words={vocab_size}  Total={total_nodes}")

    rows, cols, vals = [], [], []

    # ── Paper-Word TF-IDF edges ──────────────────────────────────────────
    from sklearn.feature_extraction.text import TfidfVectorizer
    clean_abstracts = [clean_str(a) for a in abstracts]
    tfidf_vec       = TfidfVectorizer(vocabulary=word_id_map, norm="l2")
    tfidf_mat_full  = tfidf_vec.fit_transform(clean_abstracts)   # (N, V) sparse

    cx = tfidf_mat_full.tocoo()
    for p_idx, w_local, v in zip(cx.row, cx.col, cx.data):
        if v > 0:
            w_global = num_papers + w_local
            rows += [p_idx, w_global];  cols += [w_global, p_idx];  vals += [v, v]

    # ── Word-Word PPMI edges ──────────────────────────────────────────────
    pmi_edges = compute_pmi_edges(tokenized, word_id_map, window_size)
    for (wi, wj), pmi in pmi_edges.items():
        wgi, wgj = num_papers + wi, num_papers + wj
        rows += [wgi, wgj];  cols += [wgj, wgi];  vals += [pmi, pmi]

    # ── Paper-Paper citation edges ────────────────────────────────────────
    cite_count: Dict[Tuple[int, int], int] = defaultdict(int)
    for (citing_id, cited_id) in citation_edges:
        if citing_id in node_id_to_idx and cited_id in node_id_to_idx:
            i, j = node_id_to_idx[citing_id], node_id_to_idx[cited_id]
            cite_count[(i, j)] += 1
    for (i, j), cnt in cite_count.items():
        rows += [i, j];  cols += [j, i];  vals += [cnt, cnt]

    # ── Self-loops ────────────────────────────────────────────────────────
    for i in range(total_nodes):
        rows.append(i);  cols.append(i);  vals.append(1.0)

    A = sp.csr_matrix(
        (vals, (rows, cols)), shape=(total_nodes, total_nodes), dtype=np.float32
    )
    adj_norm = _normalise(A)
    tfidf_dense = np.asarray(tfidf_mat_full.todense(), dtype=np.float32)
    return adj_norm, tfidf_dense, num_papers


# ─────────────────────────────────────────────
#  RPCG-2 — Paper-Author-Citation Graph
# ─────────────────────────────────────────────

def build_rpcg2(
    scholar_data: List[dict],
    abstracts: List[str],
    node_ids: List[str],
    citation_edges: List[Tuple[str, str]],
    main_positions: List[int] = None,
) -> Tuple[sp.csr_matrix, int]:
    """
    Build RPCG-2 adjacency matrix.
    Nodes = papers + authors.
    Edges = paper-author (authorship) + author-author (PPMI co-authorship)
            + paper-paper (citation count).

    Returns:
      adj_norm   : normalised sparse adjacency
      num_papers : N
    """
    num_papers     = len(abstracts)
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # ── Collect authors ───────────────────────────────────────────────────
    author_to_id: Dict[str, int] = {}
    paper_authors: List[List[int]] = [[] for _ in range(num_papers)]

    def _parse_authors(raw: str) -> List[str]:
        if not isinstance(raw, str):
            return []
        # Google Scholar format: "A Author, B Author… - Journal, Year"
        raw = re.split(r"[-–—]", raw)[0]
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        # Drop parts that look like journal names (start with uppercase + long)
        return [p for p in parts if p and len(p) < 40]

    import re
    if main_positions is None:
        # Backward-compatible default assumes no interleaving (WRONG for
        # data produced by extract_corpus whenever any paper has citing
        # articles — kept only so old call sites don't hard-crash).
        main_positions = list(range(len(scholar_data)))

    for idx, entry in enumerate(scholar_data):
        authors_raw = entry.get("scraped_main_authors", "") or ""
        authors     = _parse_authors(authors_raw)
        node_pos    = main_positions[idx]   # true position of this main paper

        for a in authors:
            if a not in author_to_id:
                author_to_id[a] = len(author_to_id)
            paper_authors[node_pos].append(author_to_id[a])

    num_authors  = len(author_to_id)
    total_nodes  = num_papers + num_authors
    logger.info(f"RPCG-2 | Papers={num_papers}  Authors={num_authors}  Total={total_nodes}")

    rows, cols, vals = [], [], []

    # ── Paper-Author edges ────────────────────────────────────────────────
    for p_idx, a_list in enumerate(paper_authors):
        for a_local in a_list:
            a_global = num_papers + a_local
            rows += [p_idx, a_global];  cols += [a_global, p_idx];  vals += [1.0, 1.0]

    # ── Author-Author PPMI (co-authorship) ───────────────────────────────
    author_paper_sets: Dict[int, set] = defaultdict(set)
    for p_idx, a_list in enumerate(paper_authors):
        for a in a_list:
            author_paper_sets[a].add(p_idx)

    total_papers = num_papers
    for ai in author_paper_sets:
        for aj in author_paper_sets:
            if aj <= ai:
                continue
            shared = len(author_paper_sets[ai] & author_paper_sets[aj])
            if shared == 0:
                continue
            p_ai  = len(author_paper_sets[ai]) / total_papers
            p_aj  = len(author_paper_sets[aj]) / total_papers
            p_ij  = shared / total_papers
            if p_ai > 0 and p_aj > 0:
                pmi = math.log(p_ij / (p_ai * p_aj) + 1e-9)
                if pmi > 0:
                    gi, gj = num_papers + ai, num_papers + aj
                    rows += [gi, gj];  cols += [gj, gi];  vals += [pmi, pmi]

    # ── Paper-Paper citation edges ────────────────────────────────────────
    cite_count: Dict[Tuple[int, int], int] = defaultdict(int)
    for (citing_id, cited_id) in citation_edges:
        if citing_id in node_id_to_idx and cited_id in node_id_to_idx:
            i, j = node_id_to_idx[citing_id], node_id_to_idx[cited_id]
            cite_count[(i, j)] += 1
    for (i, j), cnt in cite_count.items():
        rows += [i, j];  cols += [j, i];  vals += [cnt, cnt]

    # ── Self-loops ────────────────────────────────────────────────────────
    for i in range(total_nodes):
        rows.append(i);  cols.append(i);  vals.append(1.0)

    A = sp.csr_matrix(
        (vals, (rows, cols)), shape=(total_nodes, total_nodes), dtype=np.float32
    )
    adj_norm = _normalise(A)
    return adj_norm, num_papers


# ─────────────────────────────────────────────
#  Adjacency normalisation
# ─────────────────────────────────────────────

def _normalise(A: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} A D^{-1/2} symmetric normalisation."""
    deg   = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv = sp.diags(d_inv)
    return (D_inv @ A @ D_inv).tocsr().astype(np.float32)


def sparse_to_torch(A: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert scipy sparse CSR to torch sparse COO tensor."""
    A_coo  = A.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
    values  = torch.from_numpy(A_coo.data).float()
    shape   = torch.Size(A_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


# ─────────────────────────────────────────────
#  Train / Val / Test split
# ─────────────────────────────────────────────

def make_splits(
    num_papers: int,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (idx_train, idx_val, idx_test) as integer arrays."""
    indices = list(range(num_papers))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(num_papers * train_ratio)
    n_val   = int(num_papers * val_ratio)
    idx_train = np.array(indices[:n_train])
    idx_val   = np.array(indices[n_train: n_train + n_val])
    idx_test  = np.array(indices[n_train + n_val:])
    return idx_train, idx_val, idx_test
