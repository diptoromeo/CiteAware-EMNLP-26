"""
models/llm_encoder.py
─────────────────────────────────────────────────────────────────────────────
Scientific PLM text encoder with a lightweight classification adapter.

Default backbone: allenai/scibert_scivocab_uncased
  - Trained on 1.14M scientific papers
  - Superior to bert-base-uncased for academic text
  - Still classifies as a PLM (honest terminology)

To use a true LLM backbone (e.g. Llama-3.2-3B) set
  model_name = "meta-llama/Llama-3.2-3B-Instruct"
and enable frozen_layers = all (the adapter-only regime).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ScientificTextEncoder(nn.Module):
    """
    Wraps a pre-trained transformer with:
      - Selective layer freezing (reduces GPU memory, speeds training)
      - Mean-pool CLS aggregation
      - Two-layer MLP adapter → K logits

    The adapter is always trainable; the backbone is partially or fully frozen.
    This is the correct way to use any backbone (BERT, SciBERT, or LLM)
    without leaking label information through the weights.
    """

    def __init__(
        self,
        num_labels:    int,
        model_name:    str   = "allenai/scibert_scivocab_uncased",
        frozen_layers: int   = 8,      # freeze first N encoder layers (0=none, -1=all)
        hidden_dim:    int   = 256,
        dropout:       float = 0.1,
        max_seq_len:   int   = 512,
    ):
        super().__init__()
        self.model_name  = model_name
        self.max_seq_len = max_seq_len

        logger.info(f"Loading backbone: {model_name}")
        self.tokenizer   = AutoTokenizer.from_pretrained(model_name)
        self.backbone    = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

        # Freeze selected layers
        self._freeze_layers(frozen_layers)

        # Adapter head: backbone_dim → hidden → K
        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_labels),
        )

    # ──────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:  (B, K) logits — raw, before sigmoid.
        """
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding tokens
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.adapter(pooled)

    # ──────────────────────────────────────────
    #  Tokenisation helper
    # ──────────────────────────────────────────

    def tokenize(
        self,
        texts:  list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    # ──────────────────────────────────────────
    #  Layer freezing
    # ──────────────────────────────────────────

    def _freeze_layers(self, n: int) -> None:
        """
        n  >  0 : freeze first n transformer encoder layers + embeddings
        n == 0  : nothing frozen (full fine-tune)
        n == -1 : freeze entire backbone (adapter-only training)
        """
        if n == 0:
            return

        # Always freeze embeddings when any layers are frozen
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False

        if n == -1:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone fully frozen — adapter-only training.")
            return

        # Partial freeze
        encoder_layers = getattr(
            self.backbone,
            "encoder",
            getattr(self.backbone, "transformer", None)
        )
        if encoder_layers is None:
            logger.warning("Could not locate encoder layers — no layers frozen.")
            return

        layer_list = getattr(encoder_layers, "layer",
                     getattr(encoder_layers, "layers", []))
        for i, layer in enumerate(layer_list):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

        frozen = n if n <= len(layer_list) else len(layer_list)
        total  = len(layer_list)
        logger.info(f"Frozen {frozen}/{total} backbone layers + embeddings.")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
