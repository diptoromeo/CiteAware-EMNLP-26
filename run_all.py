"""
run_all.py  —  Train CiteAware on all 4 datasets × 2 graph types
─────────────────────────────────────────────────────────────────────────────
Usage:
  python run_all.py --data_dir data/ --label_mode keyword
  python run_all.py --data_dir data/ --label_mode hybrid --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

DATASETS    = ["arXiv", "DBLP", "Elsevier", "PubMed"]
GRAPH_TYPES = ["RPCG1", "RPCG2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--label_mode", default="keyword",
                   choices=["keyword", "semantic", "hybrid"])
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    summary = {}

    for ds in DATASETS:
        summary[ds] = {}
        for gt in GRAPH_TYPES:
            cmd = [sys.executable, "train.py",
                   "--dataset",    ds,
                   "--graph",      gt,
                   "--label_mode", args.label_mode,
                   "--data_dir",   args.data_dir]

            if args.epochs:     cmd += ["--epochs",     str(args.epochs)]
            if args.batch_size: cmd += ["--batch_size", str(args.batch_size)]
            if args.device:     cmd += ["--device",     args.device]

            logger.info(f"\n{'='*60}")
            logger.info(f"  Running: {ds} / {gt}")
            logger.info(f"{'='*60}")

            ret = subprocess.run(cmd, capture_output=False)

            # Load result if available
            result_path = os.path.join("outputs", f"{ds}_{gt}_results.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                summary[ds][gt] = r.get("test", {})
            else:
                summary[ds][gt] = {"error": f"returncode={ret.returncode}"}

    # ── Print summary table ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("  FULL RESULTS SUMMARY")
    print("="*80)
    header = f"{'Dataset':<12} {'Graph':<8} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Micro':>10}"
    print(header)
    print("-"*80)
    for ds in DATASETS:
        for gt in GRAPH_TYPES:
            m = summary[ds][gt]
            acc = m.get("accuracy", float("nan"))
            f1m = m.get("f1_macro", float("nan"))
            f1i = m.get("f1_micro", float("nan"))
            print(f"{ds:<12} {gt:<8} {acc:>10.4f} {f1m:>10.4f} {f1i:>10.4f}")
    print("="*80)

    # Save summary
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join("outputs", f"summary_{ts}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved → {path}")


if __name__ == "__main__":
    main()
