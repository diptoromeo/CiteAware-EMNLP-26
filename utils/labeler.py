"""
utils/labeler.py
─────────────────────────────────────────────────────────────────────────────
Zero-leakage multi-label assignment using:
  1. Keyword matching against curated domain taxonomy  (fast, no model)
  2. SciBERT zero-shot similarity scoring              (accurate, no leakage)

Replaces the TF-IDF title-derived label approach which caused data leakage.
Labels are derived purely from the taxonomy — independent of abstract text
used as GCN/LLM input.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Keyword → label mapping (domain agnostic)
# ─────────────────────────────────────────────
KEYWORD_MAP: dict[str, list[str]] = {
    # arXiv / CS / Math
    "graph_theory":              ["graph", "vertex", "coloring", "automorphism", "domination",
                                   "clique", "chromatic", "isomorphism", "planar", "cycle",
                                   "tree", "bipartite", "hamiltonian", "adjacency"],
    "machine_learning":          ["machine learning", "classification", "regression", "feature",
                                   "training", "gradient", "overfitting", "generalization",
                                   "supervised", "unsupervised", "ensemble", "boosting"],
    "deep_learning":             ["neural network", "deep learning", "convolutional", "transformer",
                                   "attention", "embedding", "fine-tuning", "backpropagation",
                                   "activation", "layer", "architecture", "bert", "gpt"],
    "natural_language_processing":["natural language", "nlp", "text classification", "sentiment",
                                   "named entity", "machine translation", "language model",
                                   "tokenization", "parsing", "summarization", "question answering"],
    "computer_vision":           ["image", "vision", "object detection", "segmentation",
                                   "recognition", "pixel", "convolutional", "resnet", "yolo",
                                   "feature extraction", "visual"],
    "optimization":              ["optimization", "convergence", "gradient descent", "convex",
                                   "stochastic", "objective function", "constraint", "lagrangian",
                                   "linear programming", "heuristic", "metaheuristic"],
    "quantum_computing":         ["quantum", "qubit", "superconducting", "entanglement",
                                   "quantum circuit", "decoherence", "quantum gate", "qubits"],
    "algorithms":                ["algorithm", "complexity", "polynomial", "dynamic programming",
                                   "sorting", "search", "hash", "approximation", "randomized"],
    "data_structures":           ["data structure", "tree", "heap", "linked list", "stack",
                                   "queue", "hash table", "array", "trie", "segment tree"],
    "reinforcement_learning":    ["reinforcement learning", "reward", "policy", "agent",
                                   "environment", "q-learning", "markov", "exploration"],
    "network_science":           ["network", "social network", "community detection", "pagerank",
                                   "centrality", "connectivity", "link prediction"],
    "statistics":                ["bayesian", "probability", "distribution", "hypothesis",
                                   "variance", "regression", "statistical", "confidence interval"],
    "information_theory":        ["entropy", "mutual information", "channel capacity",
                                   "coding", "compression", "information theory"],
    # DBLP extras
    "databases":                 ["database", "sql", "query", "relational", "nosql",
                                   "transaction", "index", "schema", "join", "storage"],
    "security":                  ["security", "cryptography", "encryption", "authentication",
                                   "privacy", "vulnerability", "attack", "malware", "intrusion"],
    "human_computer_interaction":["user interface", "usability", "hci", "interaction",
                                   "user experience", "accessibility", "visualization"],
    "distributed_systems":       ["distributed", "consensus", "fault tolerance", "replication",
                                   "byzantine", "cloud", "scalability", "load balancing"],
    "internet_of_things":        ["iot", "sensor", "edge computing", "smart", "wireless",
                                   "embedded", "microcontroller", "actuator"],
    # Elsevier extras
    "energy_engineering":        ["energy", "renewable", "solar", "wind", "battery",
                                   "power grid", "tidal", "fuel cell", "turbine", "efficiency"],
    "materials_science":         ["material", "alloy", "composite", "nanoparticle",
                                   "polymer", "ceramic", "mechanical property", "microstructure"],
    "environmental_science":     ["environment", "pollution", "climate", "emission", "carbon",
                                   "biodiversity", "ecosystem", "waste", "water quality"],
    "biomedical_engineering":    ["biomedical", "prosthetic", "implant", "tissue engineering",
                                   "scaffold", "biomaterial", "medical device"],
    "chemical_engineering":      ["chemical", "reactor", "catalyst", "separation",
                                   "distillation", "thermodynamics", "kinetics"],
    # PubMed extras
    "oncology":                  ["cancer", "tumor", "carcinoma", "metastasis", "chemotherapy",
                                   "oncology", "lymphoma", "leukemia", "survival rate"],
    "cardiology":                ["cardiac", "heart", "myocardial", "arrhythmia", "echocardiography",
                                   "coronary", "atrial", "ventricular", "stroke"],
    "neurology":                 ["neuron", "brain", "alzheimer", "parkinson", "epilepsy",
                                   "cognitive", "dementia", "cerebellar", "neural"],
    "infectious_disease":        ["infection", "virus", "bacteria", "pathogen", "vaccine",
                                   "antibiotic", "epidemic", "pandemic", "covid", "influenza",
                                   "monkeypox", "sars", "hiv"],
    "immunology":                ["immune", "antibody", "antigen", "t-cell", "b-cell",
                                   "cytokine", "inflammation", "autoimmune", "allergy"],
    "genetics":                  ["gene", "dna", "rna", "mutation", "genomic", "sequencing",
                                   "snp", "chromosome", "expression", "protein"],
    "pharmacology":              ["drug", "pharmacokinetics", "dosage", "clinical trial",
                                   "adverse effect", "bioavailability", "receptor", "inhibitor"],
    "surgery":                   ["surgery", "surgical", "operative", "laparoscopic",
                                   "resection", "anastomosis", "incision", "wound"],
    "psychiatry":                ["mental health", "depression", "anxiety", "schizophrenia",
                                   "bipolar", "psychiatric", "cognitive behavioral", "ptsd"],
    "endocrinology":             ["diabetes", "insulin", "thyroid", "hormone", "obesity",
                                   "metabolic", "cortisol", "endocrine", "glucose"],
}


class ZeroLeakageLabeler:
    """
    Assigns multi-label targets to documents using an external taxonomy.
    Two modes:
      - 'keyword'  : fast regex-based keyword matching (no GPU required)
      - 'semantic' : SciBERT cosine similarity between abstract and label
                     description (more accurate, needs GPU for speed)
    Both modes are leakage-free: labels come from the taxonomy, not the text.
    """

    def __init__(
        self,
        label_set: List[str],
        mode: str = "keyword",
        model_name: str = "allenai/scibert_scivocab_uncased",
        device: Optional[torch.device] = None,
        threshold_keyword: float = 1.0,   # ≥1 keyword hit → positive
        threshold_semantic: float = 0.30, # cosine similarity threshold
    ):
        if mode not in ("keyword", "semantic", "hybrid"):
            raise ValueError("mode must be 'keyword', 'semantic', or 'hybrid'")

        self.label_set          = label_set
        self.mode               = mode
        self.threshold_keyword  = threshold_keyword
        self.threshold_semantic = threshold_semantic
        self.device             = device or torch.device("cpu")

        # Build compiled patterns for keyword mode
        self._patterns = {}
        for label in label_set:
            kws = KEYWORD_MAP.get(label, [label.replace("_", " ")])
            pattern = "|".join(re.escape(k) for k in kws)
            self._patterns[label] = re.compile(pattern, re.IGNORECASE)

        # Load encoder only for semantic / hybrid modes
        self._tokenizer = None
        self._encoder   = None
        if mode in ("semantic", "hybrid"):
            logger.info(f"Loading {model_name} for semantic labeling …")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._encoder   = AutoModel.from_pretrained(model_name).to(self.device)
            self._encoder.eval()
            self._label_embeds = self._embed_labels()

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def assign(self, texts: List[str]) -> np.ndarray:
        """
        Returns binary label matrix  Y  of shape (N, K).
        Each row is a multi-hot vector.
        """
        N, K = len(texts), len(self.label_set)

        if self.mode == "keyword":
            return self._keyword_assign(texts)

        if self.mode == "semantic":
            return self._semantic_assign(texts)

        # hybrid: OR of keyword and semantic
        kw  = self._keyword_assign(texts)
        sem = self._semantic_assign(texts)
        return np.clip(kw + sem, 0, 1)

    def label_names(self) -> List[str]:
        return self.label_set

    # ──────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────

    def _keyword_assign(self, texts: List[str]) -> np.ndarray:
        Y = np.zeros((len(texts), len(self.label_set)), dtype=np.float32)
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                continue
            for j, label in enumerate(self.label_set):
                hits = len(self._patterns[label].findall(text))
                if hits >= self.threshold_keyword:
                    Y[i, j] = 1.0
        # Fallback: if no label fires, assign the label with most hits
        empty_rows = Y.sum(axis=1) == 0
        if empty_rows.any():
            for i in np.where(empty_rows)[0]:
                text = texts[i] if isinstance(texts[i], str) else ""
                counts = [len(self._patterns[l].findall(text)) for l in self.label_set]
                best = int(np.argmax(counts))
                Y[i, best] = 1.0
        return Y

    def _semantic_assign(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        doc_embeds = self._embed_texts(texts, batch_size)
        # Cosine similarity: (N, d) × (K, d)^T → (N, K)
        d_norm = doc_embeds / (np.linalg.norm(doc_embeds, axis=1, keepdims=True) + 1e-9)
        l_norm = self._label_embeds / (np.linalg.norm(self._label_embeds, axis=1, keepdims=True) + 1e-9)
        sims   = d_norm @ l_norm.T          # (N, K)
        Y      = (sims >= self.threshold_semantic).astype(np.float32)
        # Ensure at least one label per document
        empty_rows = Y.sum(axis=1) == 0
        if empty_rows.any():
            best = sims[empty_rows].argmax(axis=1)
            Y[np.where(empty_rows)[0], best] = 1.0
        return Y

    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
            enc   = self._tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self._encoder(**enc)
            # Mean pool over non-padding tokens
            mask       = enc["attention_mask"].unsqueeze(-1).float()
            pooled     = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_embeds.append(pooled.cpu().numpy())
        return np.vstack(all_embeds)

    def _embed_labels(self) -> np.ndarray:
        """
        Embed human-readable label descriptions for cosine comparison.
        """
        descriptions = []
        for label in self.label_set:
            kws   = KEYWORD_MAP.get(label, [label.replace("_", " ")])
            desc  = label.replace("_", " ") + ": " + ", ".join(kws[:6])
            descriptions.append(desc)
        return self._embed_texts(descriptions, batch_size=len(descriptions))
