"""
CiteAware: Heterogeneous Citation Graph Networks with LLM
Configuration file — edit this to change all hyperparameters in one place.
"""

import os

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")

# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
DATASETS = ["arXiv", "DBLP", "Elsevier", "PubMed"]

# ─────────────────────────────────────────────
#  CTC (Citing-Paper Topic Clustering) — Section 4.2 of the paper
# ─────────────────────────────────────────────
# Labels come ONLY from citing-paper abstracts, never from the target
# paper's own text (see utils/ctc_labeler.py). This is the labeling
# scheme actually described in the paper and MUST be used for any
# reported results. DOMAIN_LABELS below is kept only as an optional,
# clearly-non-default ablation ("external taxonomy labeling") and is not
# used unless --label_mode=taxonomy is explicitly passed.
CTC_NUM_CLUSTERS   = 10          # K in Eq. 10
CTC_TAU_MIN_FRAC   = 0.05        # Appendix D.2 frequency filter (lower bound)
CTC_TAU_MAX_FRAC   = 0.60        # Appendix D.2 frequency filter (upper bound)
CTC_MODEL_NAME     = "allenai/scibert_scivocab_uncased"   # Eq. 8 encoder
CTC_MAX_SEQ_LEN    = 256

# Node / LLM-branch backbone (Section 4.1.4 / 4.3.2)
NODE_LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Domain-specific label taxonomies — OPTIONAL fallback ablation only
# (NOT the paper's method; kept for --label_mode=taxonomy comparisons).
DOMAIN_LABELS = {
    "arXiv": [
        "graph_theory", "machine_learning", "deep_learning", "natural_language_processing",
        "computer_vision", "optimization", "quantum_computing", "bioinformatics",
        "cryptography", "distributed_systems", "algorithms", "data_structures",
        "reinforcement_learning", "robotics", "signal_processing", "network_science",
        "computational_biology", "statistics", "information_theory", "software_engineering"
    ],
    "DBLP": [
        "databases", "networking", "machine_learning", "software_engineering",
        "computer_vision", "natural_language_processing", "security", "distributed_systems",
        "algorithms", "human_computer_interaction", "data_mining", "bioinformatics",
        "operating_systems", "programming_languages", "computer_graphics",
        "information_retrieval", "mobile_computing", "cloud_computing",
        "internet_of_things", "embedded_systems"
    ],
    "Elsevier": [
        "energy_engineering", "materials_science", "biomedical_engineering",
        "environmental_science", "mechanical_engineering", "chemical_engineering",
        "electrical_engineering", "civil_engineering", "physics", "chemistry",
        "biology", "medicine", "economics", "psychology", "ecology",
        "neuroscience", "pharmacology", "nutrition", "climate_science", "nanotechnology"
    ],
    "PubMed": [
        "oncology", "cardiology", "neurology", "infectious_disease", "immunology",
        "genetics", "pharmacology", "surgery", "psychiatry", "endocrinology",
        "pediatrics", "radiology", "dermatology", "gastroenterology", "pulmonology",
        "nephrology", "rheumatology", "hematology", "ophthalmology", "orthopedics"
    ]
}

# Top-K labels to use per dataset (subset of DOMAIN_LABELS)
TOP_K_LABELS = 10

# ─────────────────────────────────────────────
#  Text Preprocessing
# ─────────────────────────────────────────────
REMOVE_FREQ_LIMIT  = 3   # discard words appearing fewer than N times
GRAPH_WINDOW_SIZE  = 20  # sliding window for word co-occurrence (PMI)
MIN_ABSTRACT_LEN   = 20  # characters; shorter abstracts are skipped

# ─────────────────────────────────────────────
#  LLM (Text Encoder)
# ─────────────────────────────────────────────
# NODE_LLM_MODEL_NAME (defined above, Section 4.1.4/4.3.2 of the paper) is
# the frozen backbone used for the node-feature / LLM-adapter branch
# (Llama-3.2-3B-Instruct). LLM_MODEL_NAME is kept as an alias so existing
# call sites that import LLM_MODEL_NAME continue to work, and can be
# overridden from the CLI (--llm_model) for ablations against smaller
# encoders (e.g. SciBERT-only, matching Table 4's "LLM" row wording).
LLM_MODEL_NAME   = NODE_LLM_MODEL_NAME
LLM_MAX_SEQ_LEN  = 512
LLM_BATCH_SIZE   = 16
LLM_LR           = 2e-5
LLM_WEIGHT_DECAY = 0.01
LLM_WARMUP_RATIO = 0.1   # fraction of total steps for warmup
LLM_FROZEN_LAYERS = 8    # freeze first N transformer layers (0 = fully fine-tune)

# ─────────────────────────────────────────────
#  GCN
# ─────────────────────────────────────────────
GCN_HIDDEN_DIM  = 200
GCN_NUM_LAYERS  = 2
GCN_DROPOUT     = 0.3
GCN_LR          = 0.01
GCN_WEIGHT_DECAY = 0.0

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
NUM_EPOCHS      = 200
EARLY_STOPPING  = 15      # patience in epochs
TRAIN_RATIO     = 0.8
VAL_RATIO       = 0.1     # of remaining after train split
RANDOM_SEED     = 42
BATCH_SIZE      = 32      # LLM mini-batch size
THRESHOLD       = 0.5     # sigmoid threshold for multi-label prediction

# ─────────────────────────────────────────────
#  Graph type
# ─────────────────────────────────────────────
# "RPCG1" = paper-word-citation  |  "RPCG2" = paper-author-citation
GRAPH_TYPE = "RPCG1"
