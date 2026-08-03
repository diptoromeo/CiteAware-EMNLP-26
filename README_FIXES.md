# CiteAware — Corrected Implementation

This is a corrected version of the uploaded `CiteAware_code.zip`, fixed to
actually implement the method described in the paper, plus two unrelated
runtime bugs that were preventing the original code from training at all.
Read this before using it.

## 1. What was wrong, and what changed

### (a) Label leakage — labels were built from the wrong text
`utils/labeler.py` assigned labels by matching the *target paper's own
abstract* against a hand-written keyword taxonomy. That is exactly the
kind of leakage the paper's CTC method (Section 4.2, Eq. 8–13) is
designed to avoid — CTC labels must come only from *citing-paper*
abstracts.

**Fix:** `utils/ctc_labeler.py` (new) implements the real CTC procedure:
encode unique citing-paper abstracts → global K-Means (K=10) → multi-hot
label per target paper based on which clusters its citing papers fall
into → frequency filter (τ_min=0.05·Np, τ_max=0.60·Np). `train.py` now
uses this exclusively.

### (b) Node/graph misalignment
`extract_corpus` interleaves each main paper with its own citing-paper
nodes (`Main_0, Cite_0_0, Cite_0_1, Main_1, ...`), so main papers do
**not** sit at positions `0..Np-1` in the graph. The original code
assumed they did (when slicing paper embeddings, attaching RPCG-2
author edges, and reading back GCN outputs), which silently misaligned
data every time any paper had citing articles — i.e. always.

**Fix:** `extract_corpus` now also returns `main_positions`, threaded
through `data_loader.py`, `train.py`, and `trainer.py` so every
per-paper tensor is indexed consistently.

### (c) Training loop crash
The trainer computed one full-graph GCN forward pass per epoch (correct,
for efficiency) but called `.backward()` inside every mini-batch loop
iteration while reusing that same forward pass — this raises
`RuntimeError: Trying to backward through the graph a second time` on
the second mini-batch of literally every epoch. **The original trainer
could not complete a single epoch on any dataset with more than one
mini-batch.**

**Fix:** `utils/trainer.py` now accumulates per-batch losses (weighted
by batch size) into one epoch-level loss and does a single
`backward()`/optimizer step per epoch — standard gradient accumulation,
preserving Eq. 19–20's intent.

## 2. Backend behavior — read this before trusting any numbers

`models/embedding_backend.py` tries to load the real backbones named in
the paper:
- SciBERT (`allenai/scibert_scivocab_uncased`) for CTC labeling (Eq. 8)
- Llama-3.2-3B-Instruct for node/LLM-branch features (Sec. 4.1.4/4.3.2)

If the HuggingFace Hub is unreachable (no internet, no token for the
gated Llama model, no cached weights), it **automatically falls back**
to an offline TF-IDF+SVD substitute and logs a loud warning. Every
output JSON records which backend actually ran under `backend_info`.

**On your GPU machine with internet access, this should just work with
the real backbones — no code changes needed.** You may need to:
```bash
huggingface-cli login   # for gated meta-llama/Llama-3.2-3B-Instruct
```

**Any results produced with `offline-tfidf-svd*` as the backend name are
NOT representative of the paper's method and must not be reported as
such.** They were only used in this sandbox (no HF Hub access) to
validate that the corrected pipeline runs end-to-end without error. See
`appendix_repro.tex` for those sandbox numbers and their caveats.

## 3. How to run for real, reportable results

```bash
pip install -r requirements.txt
# put the four *_citation_dataset.json files in data/

python train.py --dataset arXiv   --graph RPCG1
python train.py --dataset arXiv   --graph RPCG2
python train.py --dataset DBLP    --graph RPCG1
python train.py --dataset DBLP    --graph RPCG2
python train.py --dataset Elsevier --graph RPCG1
python train.py --dataset Elsevier --graph RPCG2
python train.py --dataset PubMed  --graph RPCG2
python train.py --dataset PubMed  --graph RPCG1

# ablations matching Table 3 / Table 4:
python train.py --dataset arXiv --graph RPCG1 --ablation no_citation_edges
python train.py --dataset arXiv --graph RPCG1 --ablation gcn_only
python train.py --dataset arXiv --graph RPCG1 --ablation llm_only

# multiple seeds for mean±std (paper reports 5 runs):
python train.py --dataset arXiv --graph RPCG1 --seed 7
python train.py --dataset arXiv --graph RPCG1 --seed 123
...
```

Each run writes `outputs/{dataset}_{graph}_{ablation}_seed{seed}.json`
with test metrics, training history, backend info, and CTC statistics.

`run_suite.py` is a batch driver that avoids rebuilding the graph/CTC
labels redundantly across seeds — useful if you want the full sweep
without babysitting 40+ individual `train.py` calls:
```bash
python run_suite.py --epochs 200 --seeds 42 7 123 2>&1 | tee suite_log.txt
```
(Bump `--epochs` back up to the paper's 200 with early stopping once
you're running the real backbones on GPU — 40 was only used for the
CPU/offline sandbox validation to keep runtime reasonable.)

## 4. What I did NOT change

- `models/gcn.py` (GCN architecture) already matched the paper (Eq.
  14–16) and was left as-is.
- The RPCG-1/RPCG-2 edge-weighting formulas (TF-IDF, PMI, citation
  counts) already matched Eq. 1–4 and were left as-is, apart from the
  node-alignment fix in (b) above.
- `utils/text_processing.py` (cleaning/tokenization) was left as-is.

## 5. Files changed/added relative to the original zip

- **New:** `utils/ctc_labeler.py`, `models/embedding_backend.py`,
  `run_suite.py`
- **Modified:** `train.py` (rewritten), `utils/trainer.py` (backward-pass
  fix + branch-mode ablations), `utils/data_loader.py` (main_positions +
  RPCG-2 author-alignment fix), `models/node_features.py` (uses shared
  backend abstraction), `configs/config.py` (added CTC config block)
- **Superseded, kept for reference only:** `utils/labeler.py` (the old
  leaky labeler — no longer imported by `train.py`)
