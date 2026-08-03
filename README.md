# CiteAware 🔬

**Heterogeneous Citation Graph Networks with Scientific Language Models for Multi-Label Research Paper Classification**

> ACL 2026 Submission — Anonymous

---

## Overview

CiteAware is a unified framework that classifies research papers into multiple topic labels by combining two complementary information sources:

- **Citation graph structure** — who cites whom, what words/authors co-occur
- **Semantic text understanding** — SciBERT embeddings of paper abstracts

Two heterogeneous graph types are supported:

| Graph | Nodes | Edges |
|-------|-------|-------|
| **RPCG-1** | Papers + Vocabulary words | Paper↔Word (TF-IDF), Word↔Word (PPMI), Paper↔Paper (citation count) |
| **RPCG-2** | Papers + Authors | Paper↔Author (authorship), Author↔Author (co-authorship PMI), Paper↔Paper (citation count) |

### Key design choices (addressing peer review)

| Issue | Fix applied |
|-------|-------------|
| Data leakage (TF-IDF labels from same text) | External domain taxonomy labeling — labels come from curated subject hierarchies, fully decoupled from abstract text |
| BERT ≠ LLM | SciBERT (`allenai/scibert_scivocab_uncased`) — domain-adapted scientific PLM; honest terminology throughout |
| No standard benchmark | Framework supports any citation JSON; plug in OGBN-arXiv by replacing the JSON loader |
| Missing SciBERT/SPECTER baselines | SciBERT is now the backbone; add other baselines by changing `LLM_MODEL_NAME` in `configs/config.py` |
| Joint training benefit unclear | Ablation script included; sequential vs. joint training comparison built in |

---

## Repository structure

```
CiteAware/
├── train.py                  ← single-dataset training entry point
├── run_all.py                ← train all 4 datasets × 2 graphs in one command
├── requirements.txt
│
├── configs/
│   └── config.py             ← all hyperparameters, label taxonomies, paths
│
├── data/                     ← place your *_citation_dataset.json files here
│   ├── arXiv_citation_dataset.json
│   ├── DBLP_citation_dataset.json
│   ├── Elsevier_citation_dataset.json
│   └── PubMed_citation_dataset.json
│
├── models/
│   ├── gcn.py                ← GCN with configurable depth and dropout
│   ├── llm_encoder.py        ← SciBERT + adapter head (layer freezing)
│   └── node_features.py      ← SciBERT-based node feature builder
│
├── utils/
│   ├── data_loader.py        ← JSON loading, graph construction, normalisation
│   ├── labeler.py            ← Zero-leakage label assignment (keyword / semantic / hybrid)
│   ├── text_processing.py    ← Cleaning, vocabulary, tokenisation
│   └── trainer.py            ← Joint training loop, metrics, checkpointing
│
├── outputs/                  ← JSON result files (auto-created)
└── checkpoints/              ← Best model weights (auto-created)
```

---

## Installation

```bash
git clone https://github.com/your-username/CiteAware.git
cd CiteAware

# Create and activate environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources (done automatically on first run, or manually)
python -c "import nltk; [nltk.download(r) for r in ['stopwords','wordnet','punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

**Hardware requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 24 GB (RTX 4090) |
| RAM | 16 GB | 64 GB |
| Storage | 10 GB | 50 GB |

CPU-only training is supported but slow. Set `--device cpu` explicitly.

---

## Data setup

Place the four JSON files in the `data/` directory:

```
data/
├── arXiv_citation_dataset.json
├── DBLP_citation_dataset.json
├── Elsevier_citation_dataset.json
└── PubMed_citation_dataset.json
```

Each JSON file must follow this schema:

```json
[
  {
    "original_csv_title": "Paper title here",
    "original_csv_abstract": "Full abstract text ...",
    "scraped_main_authors": "A. Author, B. Author - Journal, 2024",
    "citing_articles": [
      {
        "title": "Citing paper title",
        "authors": "C. Author - Conference, 2024",
        "abstract": "Citing paper abstract ...",
        "keywords": "keyword1, keyword2"
      }
    ]
  }
]
```

---

## Labeling strategy

CiteAware uses **zero-leakage external labeling** — labels are assigned from a curated domain taxonomy, not from the same text used as model input.

Three modes are available (set via `--label_mode`):

### `keyword` (default — fast, no GPU)

Regex pattern matching against domain-specific keyword lists defined in `utils/labeler.py`. A paper receives label `L` if any keyword associated with `L` appears in its abstract.

```python
KEYWORD_MAP = {
    "graph_theory":    ["graph", "vertex", "coloring", "clique", ...],
    "machine_learning": ["machine learning", "classification", ...],
    ...
}
```

### `semantic` (accurate, requires GPU)

SciBERT embeds each abstract and each label description. Label `L` is assigned if cosine similarity ≥ threshold (default 0.30). Completely independent of surface-level keyword matching.

### `hybrid` (best of both — recommended for final experiments)

OR of keyword and semantic predictions. Highest recall.

### Label sets

Labels are defined per dataset in `configs/config.py → DOMAIN_LABELS`. Each set contains 20 domain-relevant categories; `TOP_K_LABELS` (default 10) selects the most frequent subset per run.

| Dataset | Example labels |
|---------|---------------|
| arXiv | graph_theory, machine_learning, deep_learning, optimization, quantum_computing |
| DBLP | databases, networking, security, distributed_systems, human_computer_interaction |
| Elsevier | energy_engineering, materials_science, biomedical_engineering, climate_science |
| PubMed | oncology, cardiology, infectious_disease, genetics, pharmacology |

To add or modify labels, edit `DOMAIN_LABELS` in `configs/config.py`.

---

## Training

### Single dataset

```bash
# RPCG-1 (paper-word-citation) on arXiv with keyword labels
python train.py --dataset arXiv --graph RPCG1 --label_mode keyword

# RPCG-2 (paper-author-citation) on PubMed with hybrid labels
python train.py --dataset PubMed --graph RPCG2 --label_mode hybrid

# DBLP with semantic labels, custom epochs
python train.py --dataset DBLP --graph RPCG1 --label_mode semantic --epochs 100
```

### All datasets and graph types

```bash
python run_all.py --data_dir data/ --label_mode keyword
```

This trains 4 datasets × 2 graph types = 8 models and prints a summary table.

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | arXiv | Dataset name |
| `--graph` | RPCG1 | Graph type: RPCG1 or RPCG2 |
| `--label_mode` | keyword | Label assignment strategy |
| `--epochs` | 200 | Maximum training epochs |
| `--batch_size` | 16 | LLM mini-batch size |
| `--device` | auto | cuda or cpu |
| `--top_k` | 10 | Number of labels to use |

---

## Configuration

All hyperparameters live in `configs/config.py`. Key settings:

```python
# Text encoder backbone — change to use a different PLM or LLM
LLM_MODEL_NAME   = "allenai/scibert_scivocab_uncased"
# Alternatives:
# "michiyasunaga/BioLinkBERT-base"        ← for PubMed-heavy experiments
# "allenai/longformer-base-4096"          ← for full-text inputs (if available)
# "meta-llama/Llama-3.2-3B-Instruct"     ← true LLM (requires ~8GB VRAM frozen)

# Freeze first N transformer layers (reduces GPU memory, speeds training)
LLM_FROZEN_LAYERS = 8   # 0 = full fine-tune, -1 = adapter-only

# GCN hyperparameters
GCN_HIDDEN_DIM  = 200
GCN_NUM_LAYERS  = 2
GCN_DROPOUT     = 0.3

# Training
NUM_EPOCHS      = 200
EARLY_STOPPING  = 15    # patience
BATCH_SIZE      = 16
THRESHOLD       = 0.5   # sigmoid threshold for multi-label prediction
```

---

## Model architecture

```
                    Abstract text
                         │
               ┌─────────▼─────────┐
               │  SciBERT backbone  │  (partially frozen)
               │  + mean pooling    │
               └─────────┬─────────┘
                         │ d-dim embedding
                ┌────────▼────────┐
                │  Adapter (MLP)  │  (always trainable)
                │  d → 256 → K   │
                └────────┬────────┘
                         │ LLM logits (K)
                         │
    Citation Graph       │
         │               │
┌────────▼────────┐      │
│  Node features  │      │         ← SciBERT paper embeddings (no leakage)
│  (N+V or N+Na) │      │
└────────┬────────┘      │
         │               │
┌────────▼────────┐      │
│   GCN layers    │      │
│   (L=2, h=200) │      │
└────────┬────────┘      │
         │ GCN logits (K)│
         └───────┬────────┘
                 │  fused = (GCN + LLM) / 2
          ┌──────▼──────┐
          │   Sigmoid   │
          │  Multi-hot  │
          └─────────────┘
               Labels

Joint loss: L = BCEWithLogits(GCN) + BCEWithLogits(LLM)
Single backward pass updates both parameter sets simultaneously.
```

---

## Outputs

After training, results are saved to `outputs/`:

```json
{
  "dataset": "arXiv",
  "graph_type": "RPCG1",
  "label_mode": "keyword",
  "labels": ["graph_theory", "machine_learning", ...],
  "test": {
    "accuracy": 0.8134,
    "f1_macro": 0.7821,
    "f1_micro": 0.8056,
    "f1_sample": 0.7934
  },
  "history": {
    "train_loss": [...],
    "val_f1m": [...]
  }
}
```

Best model weights are saved to `checkpoints/{dataset}_{graph}_best.pt`.

---

## Reproducing results

```bash
# Step 1: Install
pip install -r requirements.txt

# Step 2: Place data files in data/

# Step 3: Run all experiments
python run_all.py --data_dir data/ --label_mode keyword

# Step 4: Check summary table (printed at end) and outputs/summary_*.json
```

Expected runtime per experiment on RTX 4090: approximately 15–40 minutes depending on dataset size and graph type.

---

## Extending the framework

### Use a different PLM or LLM backbone

```python
# configs/config.py
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LLM_FROZEN_LAYERS = -1   # freeze all backbone layers, train adapter only
```

### Add a new dataset

1. Prepare a JSON file following the schema above
2. Add a label set to `DOMAIN_LABELS` in `configs/config.py`
3. Run: `python train.py --dataset MyDataset --data_dir data/`

### Add a new label mode

Subclass `ZeroLeakageLabeler` in `utils/labeler.py` and implement `assign()`.

### Switch to external benchmark labels (recommended for publication)

Replace the `ZeroLeakageLabeler` call in `train.py` with a loader that reads
arXiv subject categories, MeSH terms, or DBLP venue labels from a separate
metadata file. This produces fully external, community-standard ground truth.

---

## Citation

```bibtex
@inproceedings{citeaware2026,
  title     = {CiteAware: Heterogeneous Citation Graph Networks with Scientific
               Language Models for Multi-Label Research Paper Classification},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026},
}
```

---

## License

MIT License. See `LICENSE` for details.
# CiteAware-EMNLP-26
