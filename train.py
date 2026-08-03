"""
train.py  —  CiteAware main entry point (CORRECTED to match the paper)
─────────────────────────────────────────────────────────────────────────────
Fixes applied relative to the originally-submitted repo:

  1. Labels now come from Citing-Paper Topic Clustering (CTC), Section 4.2,
     Eq. 8-13 — NOT from a keyword/semantic match against the target
     paper's own abstract (utils/ctc_labeler.py replaces utils/labeler.py
     as the default and only supported labeling path for reported results).
  2. Node / LLM-branch features use the backbone actually named in the
     paper (Llama-3.2-3B-Instruct, Section 4.1.4 / 4.3.2) via
     models/embedding_backend.py, which also embeds CTC's citing-paper
     text with SciBERT (Eq. 8) — two distinct backbones, as in the paper,
     instead of SciBERT being reused for everything.
  3. The embedding backend transparently falls back to an offline
     TF-IDF/SVD substitute ONLY when the real backbone cannot be
     downloaded, and this is loudly logged and recorded in the output
     JSON's "backend" field so results are never silently mislabeled.

Usage:
  python train.py --dataset arXiv --graph RPCG1
  python train.py --dataset PubMed --graph RPCG2 --seed 7
  python train.py --dataset DBLP --graph RPCG1 --ablation no_citation_edges
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.config import (
    DATA_DIR, OUTPUT_DIR, CKPT_DIR,
    CTC_NUM_CLUSTERS, CTC_TAU_MIN_FRAC, CTC_TAU_MAX_FRAC,
    CTC_MODEL_NAME, CTC_MAX_SEQ_LEN, NODE_LLM_MODEL_NAME,
    LLM_MAX_SEQ_LEN, LLM_BATCH_SIZE, LLM_LR, LLM_WEIGHT_DECAY,
    LLM_WARMUP_RATIO, LLM_FROZEN_LAYERS,
    GCN_HIDDEN_DIM, GCN_NUM_LAYERS, GCN_DROPOUT, GCN_LR, GCN_WEIGHT_DECAY,
    NUM_EPOCHS, EARLY_STOPPING, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, THRESHOLD,
    GRAPH_WINDOW_SIZE, REMOVE_FREQ_LIMIT,
)
from utils.data_loader import (
    load_dataset, extract_corpus, build_rpcg1, build_rpcg2,
    sparse_to_torch, make_splits,
)
from utils.ctc_labeler import CTCLabeler, collect_citing_texts
from models.embedding_backend import get_embedding_backend
from models.node_features import (
    build_paper_embeddings, build_word_embeddings, build_author_embeddings,
    combine_features_rpcg1, combine_features_rpcg2,
)
from models.gcn import GCN
from models.llm_encoder import ScientificTextEncoder
from utils.trainer import CiteAwareTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CiteAware — citation-graph classification")
    p.add_argument("--dataset",    default="arXiv", choices=["arXiv", "DBLP", "Elsevier", "PubMed"])
    p.add_argument("--graph",      default="RPCG1", choices=["RPCG1", "RPCG2"])
    p.add_argument("--data_dir",   default=DATA_DIR)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--seed",       type=int, default=RANDOM_SEED)
    p.add_argument("--ctc_model",  default=CTC_MODEL_NAME,
                   help="Backbone for CTC labeling (Eq. 8). Default: SciBERT, as in the paper.")
    p.add_argument("--node_model", default=NODE_LLM_MODEL_NAME,
                   help="Backbone for node/LLM-branch features (Sec. 4.1.4/4.3.2). "
                        "Default: Llama-3.2-3B-Instruct, as in the paper.")
    p.add_argument("--ablation", default="full",
                   choices=["full", "no_citation_edges", "gcn_only", "llm_only"],
                   help="Matches Table 3 / Table 4 ablations.")
    p.add_argument("--offline_hidden_size", type=int, default=256,
                   help="Hidden size for the offline TF-IDF/SVD fallback backend "
                        "(only used when the real backbone cannot be downloaded).")
    return p.parse_args()


def main() -> dict:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_epochs = args.epochs or NUM_EPOCHS
    batch_size = args.batch_size or LLM_BATCH_SIZE

    # ── Load dataset ────────────────────────────────────────────────────
    json_path = os.path.join(args.data_dir, f"{args.dataset}_citation_dataset.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    scholar_data = load_dataset(json_path)
    abstracts, titles, node_ids, citation_edges, main_positions = extract_corpus(scholar_data)
    num_papers = len(scholar_data)   # Np — number of MAIN (target) papers

    if args.ablation == "no_citation_edges":
        citation_edges = []   # Table 3 "w/o C_p2p" row

    # ── Embedding backends (real-or-fallback, auto-detected) ─────────────
    ctc_backend = get_embedding_backend(
        args.ctc_model, device=str(device), fallback_hidden_size=args.offline_hidden_size
    )
    node_backend = get_embedding_backend(
        args.node_model, device=str(device), fallback_hidden_size=args.offline_hidden_size
    )
    backend_info = {"ctc_backend": ctc_backend.name, "node_backend": node_backend.name}
    logger.info(f"Backends in use: {backend_info}")

    # ── CTC labels (Eq. 8-13) — built ONLY from citing-paper abstracts ───
    # (scholar_data is already ordered by main paper i = 0..Np-1, matching
    #  the Y matrix / train-val-test split indices, regardless of how
    #  main/citing nodes are interleaved inside the graph.)
    citing_texts_per_paper = collect_citing_texts(scholar_data)
    ctc = CTCLabeler(
        k_clusters=CTC_NUM_CLUSTERS,
        tau_min_frac=CTC_TAU_MIN_FRAC,
        tau_max_frac=CTC_TAU_MAX_FRAC,
        seed=args.seed,
    )
    Y, ctc_info = ctc.build_labels(citing_texts_per_paper, ctc_backend, num_papers)
    num_labels = Y.shape[1]
    labels_tensor = torch.tensor(Y, dtype=torch.float)
    logger.info(f"CTC label matrix: {Y.shape} | info={ctc_info}")

    # ── Train/Val/Test splits ─────────────────────────────────────────────
    idx_train, idx_val, idx_test = make_splits(num_papers, TRAIN_RATIO, VAL_RATIO, args.seed)
    logger.info(f"Split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    # ── Node-feature backbone (Llama in the paper) ─────────────────────────
    # IMPORTANT: the graph has Np + NC "paper-like" nodes (main papers AND
    # citing papers, per Appendix C.2 / Table 6: N1 = (Np+NC) + |V|). We
    # must embed the FULL node list, not just the Np main papers, or the
    # feature matrix will not match the adjacency matrix's node count.
    paper_embeds_full = build_paper_embeddings(
        abstracts, node_backend, batch_size=batch_size, max_len=LLM_MAX_SEQ_LEN
    )
    # Slice out just the Np main-paper rows for the LLM/adapter branch,
    # which per Eq. 17 only ever looks at target-paper abstracts.
    paper_embeds_main = paper_embeds_full[np.array(main_positions)]

    # ── Graph construction ─────────────────────────────────────────────────
    if args.graph == "RPCG1":
        adj_sp, tfidf_mat, _ = build_rpcg1(
            abstracts, node_ids, citation_edges,
            window_size=GRAPH_WINDOW_SIZE, freq_limit=REMOVE_FREQ_LIMIT,
        )
        from utils.text_processing import build_vocabulary
        word_id_map, _ = build_vocabulary(abstracts, REMOVE_FREQ_LIMIT)
        word_embeds = build_word_embeddings(word_id_map, node_backend)
        features_np = combine_features_rpcg1(paper_embeds_full, word_embeds)
    else:
        adj_sp, _ = build_rpcg2(scholar_data, abstracts, node_ids, citation_edges, main_positions)
        import re
        author_to_id, paper_authors = {}, [[] for _ in range(len(abstracts))]
        for idx, entry in enumerate(scholar_data):
            raw = entry.get("scraped_main_authors", "") or ""
            raw = re.split(r"[-\u2013\u2014]", raw)[0]
            authors = [p.strip() for p in raw.split(",") if p.strip() and len(p.strip()) < 40]
            node_pos = main_positions[idx]
            for a in authors:
                if a not in author_to_id:
                    author_to_id[a] = len(author_to_id)
                paper_authors[node_pos].append(author_to_id[a])
        author_embeds = build_author_embeddings(paper_embeds_full, paper_authors, len(author_to_id))
        features_np = combine_features_rpcg2(paper_embeds_full, author_embeds)

    adj_torch = sparse_to_torch(adj_sp, device)
    feat_tensor = torch.tensor(features_np, dtype=torch.float)
    logger.info(f"Feature matrix: {feat_tensor.shape} | Adj: {adj_sp.shape}")
    main_positions_t = torch.tensor(main_positions, dtype=torch.long)

    # ── Models ─────────────────────────────────────────────────────────
    feat_dim = feat_tensor.shape[1]
    gcn_model = GCN(nfeat=feat_dim, nhid=GCN_HIDDEN_DIM, nclass=num_labels,
                     dropout=GCN_DROPOUT, n_layers=GCN_NUM_LAYERS)

    use_real_text_encoder = args.node_model == node_backend.name  # True unless fallback fired
    if use_real_text_encoder:
        text_encoder = ScientificTextEncoder(
            num_labels=num_labels, model_name=args.node_model,
            frozen_layers=LLM_FROZEN_LAYERS, max_seq_len=LLM_MAX_SEQ_LEN,
        )
    else:
        # Offline fallback: a tiny MLP adapter over the SAME frozen
        # paper embeddings already computed for the node features,
        # mirroring the paper's "frozen encoder + trainable adapter"
        # design (Eq. 17-18) without requiring a downloadable transformer.
        text_encoder = _OfflineAdapterEncoder(paper_embeds_main, num_labels)

    trainer = CiteAwareTrainer(
        gcn_model=gcn_model, text_encoder=text_encoder, adj=adj_torch,
        features=feat_tensor, labels=labels_tensor,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        abstracts=[abstracts[p] for p in main_positions], device=device,
        gcn_lr=GCN_LR, llm_lr=LLM_LR, gcn_wd=GCN_WEIGHT_DECAY, llm_wd=LLM_WEIGHT_DECAY,
        num_epochs=num_epochs, early_stop=EARLY_STOPPING, batch_size=batch_size,
        threshold=THRESHOLD, checkpoint_dir=CKPT_DIR,
        dataset_name=args.dataset, graph_type=args.graph, warmup_ratio=LLM_WARMUP_RATIO,
        main_node_positions=main_positions_t,
        branch_mode=("gcn" if args.ablation == "gcn_only" else
                     "llm" if args.ablation == "llm_only" else "both"),
    )

    t0 = time.time()
    history = trainer.train()
    train_time = time.time() - t0
    test_metrics = trainer.evaluate_test()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "dataset": args.dataset, "graph_type": args.graph, "ablation": args.ablation,
        "seed": args.seed, "num_labels": num_labels, "backend_info": backend_info,
        "ctc_info": ctc_info, "test": test_metrics, "train_time_sec": round(train_time, 2),
        "history": {k: [round(v, 5) for v in vs] for k, vs in history.items()},
    }
    out_path = os.path.join(OUTPUT_DIR, f"{args.dataset}_{args.graph}_{args.ablation}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved -> {out_path}")

    print("\n" + "=" * 60)
    print(f"  FINAL TEST RESULTS — {args.dataset} ({args.graph}, {args.ablation})")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("=" * 60)
    return result


class _OfflineAdapterEncoder(torch.nn.Module):
    """
    Drop-in replacement for ScientificTextEncoder when the real transformer
    backbone (SciBERT/Llama) is unreachable. Reuses the frozen paper
    embeddings already computed for the node-feature branch and trains
    only a small MLP adapter on top of them — architecturally analogous
    to Eq. 17-18 (frozen encoder -> mean pooling -> trainable adapter),
    just with a non-neural frozen encoder standing in for the LLM.

    To stay compatible with utils/trainer.py (which tokenizes documents
    once via `self.enc.tokenizer(...)`, concatenates the resulting
    `input_ids` across the whole corpus in order, then does
    `self._all_input_ids[indices]` per mini-batch), the "tokenizer" here
    just hands back the running document index for each text seen (texts
    are processed strictly in corpus order by trainer._prepare_encodings,
    so indices line up with `paper_embeds`).
    """

    class _IndexTokenizer:
        def __init__(self):
            self._counter = 0

        def __call__(self, texts, **kw):
            n = len(texts)
            ids = torch.arange(self._counter, self._counter + n, dtype=torch.long).unsqueeze(1)
            self._counter += n
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def __init__(self, paper_embeds: np.ndarray, num_labels: int):
        super().__init__()
        self.max_seq_len = 512
        self._tokenizer_instance = self._IndexTokenizer()
        self.register_buffer(
            "embed_matrix", torch.tensor(paper_embeds, dtype=torch.float32), persistent=False
        )
        hidden_from = paper_embeds.shape[1]
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(hidden_from, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.LayerNorm(256), torch.nn.Linear(256, num_labels),
        )

    @property
    def tokenizer(self):
        return self._tokenizer_instance

    def forward(self, input_ids, attention_mask):
        doc_idx = input_ids.squeeze(-1)
        emb = self.embed_matrix[doc_idx]
        return self.adapter(emb)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    main()
