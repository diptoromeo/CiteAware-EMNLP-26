"""
run_suite.py — batch experiment driver used to produce the sandbox
validation results referenced in the supplementary write-up.

This is NOT part of the paper's reference pipeline (train.py is). It just
avoids rebuilding the CTC labels / graph / features from scratch for every
random seed (they don't depend on the seed — only the train/val/test split
and model initialisation do), so a multi-seed sweep across all datasets,
graphs, and ablations finishes in minutes instead of hours on CPU with the
offline backend.

Usage:
  python run_suite.py --epochs 40 --seeds 42 7 123 --offline_hidden_size 128
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
    CTC_MODEL_NAME, NODE_LLM_MODEL_NAME,
    LLM_MAX_SEQ_LEN, LLM_LR, LLM_WEIGHT_DECAY, LLM_WARMUP_RATIO,
    GCN_HIDDEN_DIM, GCN_NUM_LAYERS, GCN_DROPOUT, GCN_LR, GCN_WEIGHT_DECAY,
    TRAIN_RATIO, VAL_RATIO, THRESHOLD, GRAPH_WINDOW_SIZE, REMOVE_FREQ_LIMIT,
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
from train import _OfflineAdapterEncoder

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_suite")
logger.setLevel(logging.INFO)

DATASETS = ["arXiv", "DBLP", "Elsevier", "PubMed"]
GRAPHS = ["RPCG1", "RPCG2"]


def build_shared_artifacts(dataset: str, data_dir: str, ctc_model: str, node_model: str,
                            offline_hidden_size: int, device: torch.device, ctc_seed: int = 42):
    """Everything that is seed-independent: raw data, CTC labels, backends."""
    json_path = os.path.join(data_dir, f"{dataset}_citation_dataset.json")
    scholar_data = load_dataset(json_path)
    abstracts, titles, node_ids, citation_edges, main_positions = extract_corpus(scholar_data)
    num_papers = len(scholar_data)

    ctc_backend = get_embedding_backend(ctc_model, device=str(device), fallback_hidden_size=offline_hidden_size)
    node_backend = get_embedding_backend(node_model, device=str(device), fallback_hidden_size=offline_hidden_size)
    backend_info = {"ctc_backend": ctc_backend.name, "node_backend": node_backend.name}

    citing_texts_per_paper = collect_citing_texts(scholar_data)
    ctc = CTCLabeler(k_clusters=CTC_NUM_CLUSTERS, tau_min_frac=CTC_TAU_MIN_FRAC,
                      tau_max_frac=CTC_TAU_MAX_FRAC, seed=ctc_seed)
    Y, ctc_info = ctc.build_labels(citing_texts_per_paper, ctc_backend, num_papers)

    paper_embeds_full = build_paper_embeddings(abstracts, node_backend, batch_size=32, max_len=LLM_MAX_SEQ_LEN)
    paper_embeds_main = paper_embeds_full[np.array(main_positions)]

    return dict(
        scholar_data=scholar_data, abstracts=abstracts, node_ids=node_ids,
        citation_edges=citation_edges, main_positions=main_positions, num_papers=num_papers,
        backend_info=backend_info, node_backend=node_backend, node_model=node_model,
        Y=Y, ctc_info=ctc_info, paper_embeds_full=paper_embeds_full, paper_embeds_main=paper_embeds_main,
    )


def build_graph(shared: dict, graph: str, ablation: str, device: torch.device):
    citation_edges = [] if ablation == "no_citation_edges" else shared["citation_edges"]
    abstracts, node_ids = shared["abstracts"], shared["node_ids"]

    if graph == "RPCG1":
        adj_sp, _, _ = build_rpcg1(abstracts, node_ids, citation_edges,
                                    window_size=GRAPH_WINDOW_SIZE, freq_limit=REMOVE_FREQ_LIMIT)
        from utils.text_processing import build_vocabulary
        word_id_map, _ = build_vocabulary(abstracts, REMOVE_FREQ_LIMIT)
        word_embeds = build_word_embeddings(word_id_map, shared["node_backend"])
        features_np = combine_features_rpcg1(shared["paper_embeds_full"], word_embeds)
    else:
        adj_sp, _ = build_rpcg2(shared["scholar_data"], abstracts, node_ids, citation_edges, shared["main_positions"])
        import re
        author_to_id, paper_authors = {}, [[] for _ in range(len(abstracts))]
        for idx, entry in enumerate(shared["scholar_data"]):
            raw = entry.get("scraped_main_authors", "") or ""
            raw = re.split(r"[-\u2013\u2014]", raw)[0]
            authors = [p.strip() for p in raw.split(",") if p.strip() and len(p.strip()) < 40]
            node_pos = shared["main_positions"][idx]
            for a in authors:
                if a not in author_to_id:
                    author_to_id[a] = len(author_to_id)
                paper_authors[node_pos].append(author_to_id[a])
        author_embeds = build_author_embeddings(shared["paper_embeds_full"], paper_authors, len(author_to_id))
        features_np = combine_features_rpcg2(shared["paper_embeds_full"], author_embeds)

    adj_torch = sparse_to_torch(adj_sp, device)
    feat_tensor = torch.tensor(features_np, dtype=torch.float)
    return adj_torch, feat_tensor, adj_sp.shape[0], adj_sp.nnz


def run_once(shared: dict, graph_artifacts, seed: int, epochs: int, batch_size: int,
             ablation: str, device: torch.device, dataset: str, graph: str) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    adj_torch, feat_tensor, n_nodes, n_edges_nnz = graph_artifacts
    num_papers = shared["num_papers"]
    labels_tensor = torch.tensor(shared["Y"], dtype=torch.float)
    num_labels = labels_tensor.shape[1]
    main_positions_t = torch.tensor(shared["main_positions"], dtype=torch.long)

    idx_train, idx_val, idx_test = make_splits(num_papers, TRAIN_RATIO, VAL_RATIO, seed)

    feat_dim = feat_tensor.shape[1]
    gcn_model = GCN(nfeat=feat_dim, nhid=GCN_HIDDEN_DIM, nclass=num_labels,
                     dropout=GCN_DROPOUT, n_layers=GCN_NUM_LAYERS)
    text_encoder = _OfflineAdapterEncoder(shared["paper_embeds_main"], num_labels)

    trainer = CiteAwareTrainer(
        gcn_model=gcn_model, text_encoder=text_encoder, adj=adj_torch,
        features=feat_tensor, labels=labels_tensor,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        abstracts=[shared["abstracts"][p] for p in shared["main_positions"]], device=device,
        gcn_lr=GCN_LR, llm_lr=LLM_LR, gcn_wd=GCN_WEIGHT_DECAY, llm_wd=LLM_WEIGHT_DECAY,
        num_epochs=epochs, early_stop=8, batch_size=batch_size,
        threshold=THRESHOLD, checkpoint_dir=CKPT_DIR,
        dataset_name=dataset, graph_type=graph, warmup_ratio=LLM_WARMUP_RATIO,
        main_node_positions=main_positions_t,
        branch_mode=("gcn" if ablation == "gcn_only" else "llm" if ablation == "llm_only" else "both"),
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    test_metrics = trainer.evaluate_test()
    return {
        "dataset": dataset, "graph": graph, "ablation": ablation, "seed": seed,
        "n_nodes": int(n_nodes), "n_edges_nnz": int(n_edges_nnz),
        "test": test_metrics, "train_time_sec": round(train_time, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    p.add_argument("--offline_hidden_size", type=int, default=128)
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    args = p.parse_args()

    device = torch.device("cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for dataset in args.datasets:
        logger.info(f"=== {dataset}: building shared artifacts (CTC + embeddings) ===")
        shared = build_shared_artifacts(
            dataset, args.data_dir, CTC_MODEL_NAME, NODE_LLM_MODEL_NAME,
            args.offline_hidden_size, device,
        )
        logger.info(f"{dataset} CTC info: {shared['ctc_info']}")

        for graph in GRAPHS:
            # full / gcn_only / llm_only all share the identical graph
            # (same citation edges) — build it once and reuse.
            t0 = time.time()
            graph_with_edges = build_graph(shared, graph, "full", device)
            for ablation, seeds in [
                ("full", args.seeds),
                ("gcn_only", args.seeds[:1]),
                ("llm_only", args.seeds[:1]),
            ]:
                for seed in seeds:
                    r = run_once(shared, graph_with_edges, seed, args.epochs, args.batch_size,
                                 ablation, device, dataset, graph)
                    r["backend_info"] = shared["backend_info"]
                    r["ctc_info"] = shared["ctc_info"]
                    all_results.append(r)
                    logger.info(
                        f"{dataset} {graph} {ablation} seed={seed} "
                        f"acc={r['test']['acc']:.4f} f1m={r['test']['f1m']:.4f} "
                        f"f1i={r['test']['f1i']:.4f} ({r['train_time_sec']}s)"
                    )
            logger.info(f"  [{dataset}/{graph}] with-edges block took {time.time()-t0:.1f}s")

            # no_citation_edges — separate graph, one seed (Table 3 style)
            t0 = time.time()
            graph_no_edges = build_graph(shared, graph, "no_citation_edges", device)
            r = run_once(shared, graph_no_edges, args.seeds[0], args.epochs, args.batch_size,
                         "no_citation_edges", device, dataset, graph)
            r["backend_info"] = shared["backend_info"]
            r["ctc_info"] = shared["ctc_info"]
            all_results.append(r)
            logger.info(
                f"{dataset} {graph} no_citation_edges seed={args.seeds[0]} "
                f"acc={r['test']['acc']:.4f} f1m={r['test']['f1m']:.4f} "
                f"f1i={r['test']['f1i']:.4f} ({r['train_time_sec']}s)"
            )
            logger.info(f"  [{dataset}/{graph}] no-edges block took {time.time()-t0:.1f}s")

        # incremental save after every dataset finishes
        ds_out = os.path.join(OUTPUT_DIR, f"suite_results_{dataset}.json")
        with open(ds_out, "w") as f:
            json.dump([r for r in all_results if r["dataset"] == dataset], f, indent=2)
        logger.info(f"  saved partial results -> {ds_out}")

    out_path = os.path.join(OUTPUT_DIR, "suite_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"ALL DONE -> {out_path}  ({len(all_results)} runs)")


if __name__ == "__main__":
    main()
