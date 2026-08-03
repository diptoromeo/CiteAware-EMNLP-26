"""
utils/ctc_labeler.py
─────────────────────────────────────────────────────────────────────────────
Citing-Paper Topic Clustering (CTC) — implements the paper's Section 4.2
exactly (Eq. 8-13):

  1. For every target paper p_i, take its set of citing papers C(p_i).
  2. Encode each *unique* citing-paper abstract with a frozen encoder
     (SciBERT in the paper) via attention-mask-aware mean pooling  (Eq. 8).
  3. Stack all unique citing-paper embeddings into E_C in R^{N_C x d} (Eq. 9).
  4. Fit a single global K-Means (K=10) on E_C                      (Eq. 10).
  5. Assign each citing paper to its nearest centroid phi(e_c)      (Eq. 11).
  6. Each target paper receives a multi-hot label: Y_ik = 1 iff at least
     one of its citing papers falls in cluster k                    (Eq. 12).
  7. Stack into the supervision matrix Y in {0,1}^{Np x K}          (Eq. 13).

Crucially, labels are derived ONLY from citing-paper text — never from the
target paper's own abstract — which is what the paper claims removes label
leakage. This directly replaces utils/labeler.py (ZeroLeakageLabeler),
which incorrectly built labels from the target paper's own abstract via a
hand-written keyword taxonomy.

A frequency filter analogous to Appendix D.2 removes clusters that are
either near-empty (tau_min) or near-universal (tau_max), which are
uninformative for multi-label classification.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class CTCLabeler:
    def __init__(
        self,
        k_clusters: int = 10,
        tau_min_frac: float = 0.05,
        tau_max_frac: float = 0.60,
        seed: int = 42,
        n_init: int = 10,
    ):
        self.k_clusters = k_clusters
        self.tau_min_frac = tau_min_frac
        self.tau_max_frac = tau_max_frac
        self.seed = seed
        self.n_init = n_init

    def build_labels(
        self,
        citing_texts_per_paper: List[List[str]],
        backend,
        num_papers: int,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Args:
          citing_texts_per_paper : list of length Np; citing_texts_per_paper[i]
                                    is the list of citing-paper abstracts for
                                    target paper i (possibly empty).
          backend                : an EmbeddingBackend (SciBERT / Llama /
                                    offline fallback) used ONLY to embed
                                    citing-paper text (Eq. 8).
          num_papers              : Np

        Returns:
          Y_filtered  : (Np, K*) multi-hot label matrix after frequency
                        filtering (Eq. 13, Appendix D.2)
          info        : dict with cluster sizes, kept clusters, NC, etc.
                        (used for the CTC-geometry / topic-label appendix)
        """
        # ---- Eq. 8/9: unique citing-paper embeddings -----------------
        flat_texts: List[str] = []
        owner_index: List[int] = []          # which target paper each text belongs to
        for i, texts in enumerate(citing_texts_per_paper):
            for t in texts:
                flat_texts.append(t if isinstance(t, str) else "")
                owner_index.append(i)

        NC = len(flat_texts)
        if NC == 0:
            logger.warning("No citing-paper text found at all — returning all-zero labels.")
            return np.zeros((num_papers, self.k_clusters), dtype=np.float32), {
                "NC": 0, "kept_clusters": [], "cluster_sizes": {}
            }

        logger.info(f"CTC: encoding {NC} citing-paper abstracts with backend={backend.name} ...")
        E_C = backend.encode(flat_texts, batch_size=32, max_len=256)

        # ---- Eq. 10/11: global K-Means + nearest-centroid assignment --
        k = min(self.k_clusters, max(2, len(set(map(tuple, np.round(E_C, 3)))) ))
        k = min(k, NC) if NC >= 2 else 1
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=self.n_init)
        phi = km.fit_predict(E_C)   # cluster id per citing paper

        # ---- Eq. 12: multi-hot target-paper labels --------------------
        Y = np.zeros((num_papers, k), dtype=np.float32)
        for owner, cluster_id in zip(owner_index, phi):
            Y[owner, cluster_id] = 1.0

        cluster_sizes = {int(c): int((phi == c).sum()) for c in range(k)}

        # ---- Appendix D.2 style frequency filter -----------------------
        tau_min = int(np.floor(self.tau_min_frac * num_papers))
        tau_max = int(np.ceil(self.tau_max_frac * num_papers))
        col_freq = Y.sum(axis=0)
        keep = [j for j in range(k) if tau_min <= col_freq[j] <= tau_max]
        if len(keep) == 0:
            # never drop everything — keep the top-K by informativeness
            order = np.argsort(-np.abs(col_freq - num_papers / 2))
            keep = sorted(order[: min(k, self.k_clusters)].tolist())

        Y_filtered = Y[:, keep]

        info = {
            "NC": NC,
            "k_fit": k,
            "kept_clusters": keep,
            "cluster_sizes": cluster_sizes,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "avg_labels_per_paper": float(Y_filtered.sum(axis=1).mean()),
            "papers_with_zero_labels": int((Y_filtered.sum(axis=1) == 0).sum()),
        }
        logger.info(
            f"CTC: NC={NC} k_fit={k} kept={len(keep)}/{k} "
            f"avg_labels/paper={info['avg_labels_per_paper']:.3f} "
            f"zero-label papers={info['papers_with_zero_labels']}/{num_papers}"
        )
        return Y_filtered.astype(np.float32), info


def collect_citing_texts(scholar_data: List[dict]) -> List[List[str]]:
    """
    Pulls citing-paper abstracts (NOT the target paper's own abstract) per
    entry, straight from the raw JSON — this is the only text CTC is
    allowed to see.
    """
    out: List[List[str]] = []
    for entry in scholar_data:
        citing = entry.get("citing_articles", []) or []
        texts = []
        for art in citing:
            ab = art.get("abstract", "")
            if isinstance(ab, str) and ab.strip().lower() not in ("", "nan", "none"):
                texts.append(ab)
        out.append(texts)
    return out
