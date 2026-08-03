"""
utils/trainer.py
─────────────────────────────────────────────────────────────────────────────
Joint training loop for CiteAware (GCN + Scientific PLM).

Key fixes vs. original notebook:
  - Mini-batch LLM training (avoids OOM on large datasets)
  - Independent parameter sets with additive loss (proven equivalent)
  - Proper early-stopping on validation combined loss
  - Full metric logging: F1-macro, F1-micro, accuracy
  - Model checkpointing (saves best weights)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

from models.gcn import GCN
from models.llm_encoder import ScientificTextEncoder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    preds = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(int)
    true  = labels.cpu().numpy().astype(int)
    return {
        "accuracy":  float((preds == true).mean()),
        "f1_macro":  float(f1_score(true, preds, average="macro",  zero_division=0)),
        "f1_micro":  float(f1_score(true, preds, average="micro",  zero_division=0)),
        "f1_sample": float(f1_score(true, preds, average="samples", zero_division=0)),
    }


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class CiteAwareTrainer:
    """
    Jointly trains:
      - gcn_model   : GCN over citation graph
      - text_encoder: SciBERT adapter

    Both share the same BCEWithLogitsLoss target (multi-hot label matrix).
    Gradients are additive: L = L_gcn + L_llm, single .backward() call.
    """

    def __init__(
        self,
        gcn_model:     GCN,
        text_encoder:  ScientificTextEncoder,
        adj:           torch.Tensor,         # sparse (N, N)
        features:      torch.Tensor,         # (N, d) node features
        labels:        torch.Tensor,         # (num_papers, K) float
        idx_train:     np.ndarray,
        idx_val:       np.ndarray,
        idx_test:      np.ndarray,
        abstracts:     List[str],            # raw abstract strings
        device:        torch.device,
        gcn_lr:        float = 0.01,
        llm_lr:        float = 2e-5,
        gcn_wd:        float = 0.0,
        llm_wd:        float = 0.01,
        num_epochs:    int   = 200,
        early_stop:    int   = 15,
        batch_size:    int   = 32,
        threshold:     float = 0.5,
        checkpoint_dir: str  = "checkpoints",
        dataset_name:   str  = "dataset",
        graph_type:     str  = "RPCG1",
        warmup_ratio:   float = 0.1,
        main_node_positions: Optional[torch.Tensor] = None,
        branch_mode:    str = "both",     # "both" | "gcn" | "llm" — Table 4 ablations
    ):
        assert branch_mode in ("both", "gcn", "llm")
        self.branch_mode = branch_mode
        self.gcn   = gcn_model.to(device)
        self.enc   = text_encoder.to(device)
        self.adj   = adj.to(device)
        self.feats = features.to(device)
        self.labels = labels.to(device)
        # Position of each of the `num_papers` target papers within the
        # full graph's node ordering (graph also contains citing-paper and
        # word/author nodes interleaved in — see utils/data_loader.py).
        # If not provided, assume nodes 0..num_papers-1 ARE the papers
        # (true only for graphs built without interleaved citing nodes).
        self.main_pos = (
            main_node_positions.to(device) if main_node_positions is not None else None
        )
        self.idx_train = idx_train
        self.idx_val   = idx_val
        self.idx_test  = idx_test
        self.abstracts  = abstracts
        self.device     = device
        self.threshold  = threshold
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stop = early_stop
        self.ckpt_dir   = checkpoint_dir
        self.ds_name    = dataset_name
        self.graph_type = graph_type

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizers
        self.opt_gcn = optim.Adam(
            self.gcn.parameters(), lr=gcn_lr, weight_decay=gcn_wd
        )
        self.opt_llm = optim.AdamW(
            [p for p in self.enc.parameters() if p.requires_grad],
            lr=llm_lr, weight_decay=llm_wd,
        )

        # Warmup + cosine decay for LLM optimizer
        total_steps  = num_epochs * max(1, len(idx_train) // batch_size)
        warmup_steps = int(total_steps * warmup_ratio)
        self.sched_llm = get_linear_schedule_with_warmup(
            self.opt_llm,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.sched_gcn = CosineAnnealingLR(self.opt_gcn, T_max=num_epochs, eta_min=1e-5)

        self.criterion = nn.BCEWithLogitsLoss()

        logger.info(
            f"Trainable params — GCN: {sum(p.numel() for p in self.gcn.parameters() if p.requires_grad):,} | "
            f"Encoder: {self.enc.trainable_params():,} / {self.enc.total_params():,}"
        )

    # ──────────────────────────────────────────
    #  Pre-tokenise abstracts (done once)
    # ──────────────────────────────────────────

    def _prepare_encodings(self) -> None:
        logger.info("Tokenising abstracts …")
        self._enc_input_ids  = []
        self._enc_attn_masks = []
        for start in range(0, len(self.abstracts), self.batch_size * 4):
            batch = [a if isinstance(a, str) and a.strip() else "[UNK]"
                     for a in self.abstracts[start: start + self.batch_size * 4]]
            enc   = self.enc.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.enc.max_seq_len, return_tensors="pt"
            )
            self._enc_input_ids.append(enc["input_ids"])
            self._enc_attn_masks.append(enc["attention_mask"])

        self._all_input_ids  = torch.cat(self._enc_input_ids,  dim=0)
        self._all_attn_masks = torch.cat(self._enc_attn_masks, dim=0)
        logger.info(f"Tokenised {self._all_input_ids.shape[0]} documents.")

    # ──────────────────────────────────────────
    #  Batch LLM forward
    # ──────────────────────────────────────────

    def _llm_forward_batch(
        self, indices: np.ndarray
    ) -> torch.Tensor:
        """Return LLM logits for a batch of document indices."""
        ids   = self._all_input_ids[indices].to(self.device)
        masks = self._all_attn_masks[indices].to(self.device)
        return self.enc(ids, masks)

    def _fuse(self, gcn_logits: torch.Tensor, llm_logits: torch.Tensor) -> torch.Tensor:
        """Late-fusion per Eq. 21, or single-branch output for ablations."""
        if self.branch_mode == "gcn":
            return gcn_logits
        if self.branch_mode == "llm":
            return llm_logits
        return (gcn_logits + llm_logits) / 2

    # ──────────────────────────────────────────
    #  Epoch helpers
    # ──────────────────────────────────────────

    def _train_epoch(self) -> Dict[str, float]:
        """
        One optimisation step per epoch (matches Algorithm C.1's "full-graph
        forward once per epoch" comment). The LLM branch is still processed
        in mini-batches for memory efficiency, but its per-batch losses are
        SUMMED (not backward()-ed individually) so that a single
        `loss.backward()` call at the end of the epoch differentiates
        through the one-and-only GCN forward pass exactly once.

        NOTE: the originally uploaded trainer called `.backward()` inside
        every mini-batch iteration while reusing one shared full-graph GCN
        forward pass computed outside the loop — this raises
        "Trying to backward through the graph a second time" on the second
        mini-batch of every single epoch, for every dataset. That version
        could not have completed a training run. This implementation fixes
        it via standard gradient accumulation (one accumulated loss, one
        backward, one optimizer step per epoch).
        """
        self.gcn.train()
        self.enc.train()
        self.opt_gcn.zero_grad(set_to_none=True)
        self.opt_llm.zero_grad(set_to_none=True)

        # GCN full-graph forward — exactly once per epoch.
        all_gcn_logits = self.gcn(self.feats, self.adj)
        if self.main_pos is not None:
            all_gcn_logits = all_gcn_logits[self.main_pos]   # -> (num_papers, K)

        shuffled = self.idx_train[np.random.permutation(len(self.idx_train))]

        total_loss_gcn = 0.0
        total_loss_llm = 0.0
        n_batches = 0
        agg = dict(acc=0., f1m=0., f1i=0., n=0)

        loss_gcn_accum = None
        loss_llm_accum = None

        for start in range(0, len(shuffled), self.batch_size):
            batch_idx = shuffled[start: start + self.batch_size]
            b         = len(batch_idx)
            batch_lbl = self.labels[batch_idx]

            gcn_logits = all_gcn_logits[batch_idx]
            llm_logits = self._llm_forward_batch(batch_idx)

            loss_gcn_b = self.criterion(gcn_logits, batch_lbl)
            loss_llm_b = self.criterion(llm_logits, batch_lbl)

            # Weight each mini-batch's contribution by its size so the
            # accumulated loss is a proper mean over idx_train (Eq. 19).
            weighted_gcn = loss_gcn_b * b
            weighted_llm = loss_llm_b * b
            loss_gcn_accum = weighted_gcn if loss_gcn_accum is None else loss_gcn_accum + weighted_gcn
            loss_llm_accum = weighted_llm if loss_llm_accum is None else loss_llm_accum + weighted_llm

            fused = self._fuse(gcn_logits, llm_logits)
            m = compute_metrics(fused.detach(), batch_lbl.detach(), self.threshold)
            agg["acc"] += m["accuracy"] * b
            agg["f1m"] += m["f1_macro"] * b
            agg["f1i"] += m["f1_micro"] * b
            agg["n"]   += b
            total_loss_gcn += loss_gcn_b.item() * b
            total_loss_llm += loss_llm_b.item() * b
            n_batches += 1

        n = max(agg["n"], 1)
        loss_gcn_mean = loss_gcn_accum / n
        loss_llm_mean = loss_llm_accum / n
        if self.branch_mode == "gcn":
            loss = loss_gcn_mean
        elif self.branch_mode == "llm":
            loss = loss_llm_mean
        else:
            loss = loss_gcn_mean + loss_llm_mean         # Eq. 20 (joint CiteAware)

        loss.backward()                               # single backward per epoch
        torch.nn.utils.clip_grad_norm_(self.gcn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.enc.parameters() if p.requires_grad], 1.0
        )
        self.opt_gcn.step()
        self.opt_llm.step()
        self.sched_llm.step()
        self.sched_gcn.step()

        return {
            "loss":     (total_loss_gcn + total_loss_llm) / n,
            "loss_gcn": total_loss_gcn / n,
            "loss_llm": total_loss_llm / n,
            "acc":      agg["acc"] / n,
            "f1m":      agg["f1m"] / n,
            "f1i":      agg["f1i"] / n,
        }

    @torch.no_grad()
    def _eval_epoch(self, split_idx: np.ndarray) -> Dict[str, float]:
        self.gcn.eval()
        self.enc.eval()

        all_gcn_logits = self.gcn(self.feats, self.adj)
        if self.main_pos is not None:
            all_gcn_logits = all_gcn_logits[self.main_pos]

        agg = dict(loss=0., loss_gcn=0., loss_llm=0.,
                   acc=0., f1m=0., f1i=0., n=0)

        for start in range(0, len(split_idx), self.batch_size):
            batch_idx = split_idx[start: start + self.batch_size]
            b         = len(batch_idx)
            batch_lbl = self.labels[batch_idx]

            gcn_logits = all_gcn_logits[batch_idx]
            llm_logits = self._llm_forward_batch(batch_idx)

            loss_gcn = self.criterion(gcn_logits, batch_lbl)
            loss_llm = self.criterion(llm_logits, batch_lbl)
            if self.branch_mode == "gcn":
                loss = loss_gcn
            elif self.branch_mode == "llm":
                loss = loss_llm
            else:
                loss = loss_gcn + loss_llm

            fused = self._fuse(gcn_logits, llm_logits)
            m     = compute_metrics(fused, batch_lbl, self.threshold)

            agg["loss"]     += loss.item()     * b
            agg["loss_gcn"] += loss_gcn.item() * b
            agg["loss_llm"] += loss_llm.item() * b
            agg["acc"]      += m["accuracy"]   * b
            agg["f1m"]      += m["f1_macro"]   * b
            agg["f1i"]      += m["f1_micro"]   * b
            agg["n"]        += b

        n = max(agg["n"], 1)
        return {k: agg[k] / n for k in ("loss", "loss_gcn", "loss_llm", "acc", "f1m", "f1i")}

    # ──────────────────────────────────────────
    #  Public: train
    # ──────────────────────────────────────────

    def train(self) -> Dict[str, List]:
        self._prepare_encodings()

        history = {k: [] for k in (
            "train_loss", "train_acc", "train_f1m", "train_f1i",
            "val_loss",   "val_acc",   "val_f1m",   "val_f1i",
        )}

        best_val_loss   = float("inf")
        patience_count  = 0
        best_ckpt       = os.path.join(self.ckpt_dir, f"{self.ds_name}_{self.graph_type}_best.pt")

        for epoch in range(1, self.num_epochs + 1):
            t0    = time.time()
            train = self._train_epoch()
            val   = self._eval_epoch(self.idx_val)
            dt    = time.time() - t0

            for split, m in (("train", train), ("val", val)):
                history[f"{split}_loss"].append(m["loss"])
                history[f"{split}_acc"].append(m["acc"])
                history[f"{split}_f1m"].append(m["f1m"])
                history[f"{split}_f1i"].append(m["f1i"])

            logger.info(
                f"[{epoch:03d}/{self.num_epochs}] "
                f"Train  loss={train['loss']:.4f}  acc={train['acc']:.4f}  "
                f"f1_macro={train['f1m']:.4f}  f1_micro={train['f1i']:.4f} | "
                f"Val  loss={val['loss']:.4f}  acc={val['acc']:.4f}  "
                f"f1_macro={val['f1m']:.4f}  f1_micro={val['f1i']:.4f} | "
                f"{dt:.1f}s"
            )

            if val["loss"] < best_val_loss:
                best_val_loss  = val["loss"]
                patience_count = 0
                torch.save({
                    "epoch":     epoch,
                    "gcn":       self.gcn.state_dict(),
                    "encoder":   self.enc.state_dict(),
                    "val_loss":  best_val_loss,
                    "val_f1m":   val["f1m"],
                }, best_ckpt)
                logger.info(f"  ✓ Checkpoint saved  (val_loss={best_val_loss:.4f})")
            else:
                patience_count += 1
                if patience_count >= self.early_stop:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        return history

    # ──────────────────────────────────────────
    #  Public: evaluate on test set
    # ──────────────────────────────────────────

    def evaluate_test(self) -> Dict[str, float]:
        test = self._eval_epoch(self.idx_test)
        logger.info(
            f"TEST  loss={test['loss']:.4f}  acc={test['acc']:.4f}  "
            f"f1_macro={test['f1m']:.4f}  f1_micro={test['f1i']:.4f}"
        )
        return test
