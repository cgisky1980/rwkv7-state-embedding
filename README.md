# RWKV-7 State Semantic Embedding: A Unified Supervised Projection Framework

This repository systematically explores how to extract semantic embeddings from RWKV-7 internal states (hidden state and WKV state), based on the **albatross official inference engine** (unmodified), covering three standard evaluation tasks.

**Paper**: [paper_en.md](paper_en.md) (English) | [paper.md](paper.md) (Chinese)

## Key Results

| Task | Method | Metric | Comparison |
|------|--------|--------|------------|
| **Semantic Similarity** | Supervised Projection (46.9k pairs, deduplicated) + AnglE + 5-seed | Spearman=**0.8053** | Below bge-large ~0.85 (335M) and all-MiniLM-L6-v2 ~0.86 (22M); 0.86M projector params |
| **Topic Clustering (supervised)** | Supervised Contrastive Projection (sklearn full-text, held-out test) | v_measure=**0.6599** | vs unsupervised baseline 0.4724 (+40%); MTEB short-text unsupervised cap 0.33 |
| **Task Classification** | Hidden + MLP (independent test set) | test_acc=**0.9325** | dev_acc=0.9381, head selection on dev only |

## Key Insight

**Albatross hidden state contains semantic information (supervised classification reaches 0.93), but unsupervised methods cannot extract it (STS 0.46, clustering 0.47 on sklearn full-text / 0.33 on MTEB short-text); supervised projectors are needed to unlock its potential.**

| Task | Unsupervised | Supervised Projection | Improvement |
|------|--------------|----------------------|-------------|
| STS | 0.46 | 0.8053 | +75% |
| Clustering (sklearn full-text) | 0.4724 | 0.6599 | +40% |

**Task-specific projectors cannot be mixed**: STS learns similarity ranking, clustering learns class separation—objectives differ (STS projection transfer to clustering fails: 0.14 < 0.34 baseline).

## Strict Experimental Paradigm

- **STS**: Training data deduplicated against STS-B dev/test (removed 1249 overlapping pairs, single-sentence level)
- **Clustering**: sklearn full-text 20NG (removed headers/footers/quotes), text hash dedup (0.43% cross-post), stratified 70/15/15 split, train trains projector / dev selects best_state / test held-out
- **Classification**: Independent test set (15%), head selection only on dev

## Directory Structure

```
paper/
├── paper_en.md                       # Paper (English)
├── paper.md                          # Paper (Chinese)
├── README.md                         # This file
├── albatross_src/                    # albatross official source (multiple versions)
│   └── faster_251101/reference/      # reference implementation (rwkv7.py + cuda/)
├── models/                           # RWKV-7 0.4B model
│   └── rwkv7-g1d-0.4b-20260210-ctx8192.pth
├── cache_python/                     # feature cache (.npz)
└── scripts/
    ├── 00_setup.py                   # environment setup (download model + copy source)
    ├── extract_features.py           # batched concurrent feature extraction (albatross)
    ├── 01_clustering.py              # Task 1: clustering (unsupervised baseline, MTEB short-text)
    ├── 01b_clustering_unsupervised.py # Task 1b: unsupervised clustering (10 seeds KMeans)
    ├── 02_sts_similarity.py          # Task 2: STS (46.9k deduplicated training, supervised projection)
    ├── 03_classification.py          # Task 3: classification (Hidden+MLP, independent test set)
    ├── 04_cluster_with_projection.py # STS projection transfer to clustering (failed experiment)
    ├── 05_cluster_supervised_projection.py  # Clustering (supervised, MTEB short-text version)
    ├── 06_cluster_20ng_full.py       # Task 6: Clustering (sklearn full-text, strict dedup+split)
    ├── dedup_sts_train.py            # STS training data deduplication script
    ├── split_20ng_full.py            # 20NG sklearn full-text dedup + split script
    ├── diagnose_sts_overlap.py       # STS train/test overlap diagnosis
    ├── download_ag_news.py           # AG News download (unused)
    ├── run_with_msvc.bat             # Windows MSVC environment activation
    └── lib/
        ├── albatross_wrapper.py      # albatross wrapper (with batch concurrent extraction)
        ├── cache.py                  # .npz cache I/O
        ├── rwkv7.py                   # albatross official code
        └── cuda/                      # WKV CUDA kernel
```

## Requirements

### Software
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (package manager)
- CUDA GPU (for albatross inference)
- MSVC on Windows (to compile CUDA extensions)

### Python Dependencies

```bash
# Install with uv (recommended)
uv pip install numpy torch scikit-learn scipy huggingface_hub flag_gems

# Or with pip
pip install numpy torch scikit-learn scipy huggingface_hub flag_gems
```

## Reproduction Steps

### Step 0: Environment Setup

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# Download RWKV-7 0.4B model (BlinkDL/rwkv7-g1) + copy albatross source
uv run --project ../../scripts python 00_setup.py
```

### Step 1: Download Datasets

```powershell
cd c:\work\niceui\rwkv-router

# Download STS + clustering + classification datasets
uv run --project scripts python scripts/download_embedding_eval_data.py
```

Datasets will be downloaded to:
- `data/clustering/twentynewsgroups.jsonl` (clustering, 59545 samples)
- `data/sts/sts_train.jsonl` etc. (STS-B)
- `data/sts/nli_train.jsonl`, `extra_train.jsonl`, `sickr.jsonl` (extra training data)
- `data/golden_balanced.jsonl` (classification)

### Step 2: Extract Features (albatross concurrent inference)

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# STS features (STS-B train/dev/test)
.\run_with_msvc.bat extract_features.py --task sts --batch-size 16 --max-length 128

# STS extra training data (NLI + extra_train + SICK-R, 42k pairs total)
.\run_with_msvc.bat extract_features.py --task sts_extra --batch-size 16 --max-length 128

# Clustering full features (59545 samples)
.\run_with_msvc.bat extract_features.py --task cluster_full --batch-size 16 --max-length 128

# Classification features (limit to 8000 samples for speed)
.\run_with_msvc.bat extract_features.py --task classification --batch-size 16 --max-length 128 --limit 8000
```

**Feature extraction speed**: 250 samples/s (bucketed by length for concurrency, 3.6x speedup)

### Step 3: Run Three Tasks

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# Task 2: Semantic Similarity (46.9k deduplicated training + 5-seed ensemble) → Spearman 0.8053
uv run --project ../../scripts python 02_sts_similarity.py --data-dir ../../data/sts_dedup --device cuda --n-epochs 50

# Task 6: Topic Clustering (supervised contrastive learning, optimal config + 5-seed ensemble) → v_measure 0.6599
uv run --project ../../scripts python 06_cluster_sklearn.py --device cuda --temperature 0.3 --n-pairs 80000 --dropout 0.1

# Task 3: Task Classification (Hidden + MLP) → test_acc 0.9325
uv run --project ../../scripts python 03_classification.py
```

### Step 4 (Optional): Ablation Experiments

```powershell
# Unsupervised clustering baseline (Hidden + standardize + KMeans) → v_measure 0.34
uv run --project ../../scripts python 01_clustering.py

# STS projection transfer to clustering (failed experiment) → v_measure 0.14
uv run --project ../../scripts python 04_cluster_with_projection.py
```

## Expected Output

### Task 2: Semantic Similarity (deduplicated training data)
```
Conclusion:
  Unsupervised Hidden cosine:  Spearman = 0.4600
  5-seed ensemble (dedup):     Spearman = 0.8053
```

### Task 6: Topic Clustering (sklearn full-text, held-out test)
```
Conclusion:
  Baseline (Hidden + standardize, 10 seeds):    v_measure = 0.4724 ± 0.0103
  Projection + KMeans (supervised, 5 seeds):     v_measure = 0.6599
```

### Task 3: Task Classification (independent test set)
```
Result:
  Hidden + MLP:        dev_acc = 0.9381  test_acc = 0.9325
  Top-8 head + PCA:    dev_acc = 0.9247  test_acc = 0.9208
```

## Key Design Decisions

### 1. Why Supervised Projection (not Unsupervised)
Albatross hidden state exhibits severe anisotropy (unsupervised STS only 0.46), but supervised MLP classification reaches 0.93, proving the features contain information. Supervised projectors (MLP + AnglE Loss) map hidden to a linearly separable semantic space.

### 2. Why STS and Clustering Need Different Projectors
- **STS**: learns similarity ranking (relative distance), L2 normalize compresses inter-class distance
- **Clustering**: learns class separation (absolute distance), preserves absolute position

STS projection transfer to clustering fails (0.14 < 0.34 baseline), proving projectors must align with task objectives.

### 3. Why Data Scale is Key
STS scaling from 5.7k to 48.1k (8x) yields +47% improvement, indicating albatross hidden's semantic information requires sufficient data to extract via supervised learning.

### 4. Why albatross (not Rust web-rwkv)
- albatross is the official PyTorch implementation, supporting batched concurrent inference
- This work uses the official inference engine entirely, without modifying any source code
- albatross path optimal τ=0.50 (Rust path is 0.1), root cause is hidden value range difference

## Summary of Failed Directions

| Method | Result | Failure Reason |
|--------|--------|----------------|
| STS projection transfer to clustering | 0.14 | STS learns ranking, clustering needs separation |
| Unsupervised KMeans (MTEB short-text) | 0.29 | Hidden anisotropy severe, unsupervised cannot extract |
| Unsupervised KMeans (sklearn full-text) | 0.45 | Better than short-text but still limited |
| Unsupervised Hidden cosine (STS) | 0.46 | Anisotropy causes 0.46 << 0.80 |
| UMAP nonlinear dim. reduction | 0.33 | Nonlinear reduction also ineffective |
| DeepCluster self-supervised iter. | 0.30 | Pseudo-label quality too low |
| Multi-layer hidden concatenation | 0.26 | Shallow-layer noise dilutes deep semantics |
| Pure WKV state (albatross, Q-Readout) | 0.11 | State value range small, std=0.13 |
| WKV state aggregation stats | 0.10 | row_sum/diag/trace have no clustering info |
| AG News transfer to 20NG (abandoned) | - | Different class granularity (4 vs 20), STS transfer already failed |

## Honest Disclosure

Early versions reported STS 0.8504 and clustering 0.8466, which were **inflated by data leakage**:
- **STS**: Training data (STS12-16) had 303 pair-level overlaps + 1020 sentence-level overlaps with STS-B test
- **Clustering**: Projector trained on 20NG labels then evaluated on the same 20NG (supervised clustering, not unsupervised)

This version fixes both issues:
- **STS**: Strict deduplication (removed 1249 leak pairs, single-sentence level) → 0.8053
- **Clustering**: sklearn full-text 20NG, strict dedup + 70/15/15 split, held-out test → 0.6599

## Citation

```bibtex
@misc{rwkv-state-embedding-2026,
  title={RWKV-7 State Semantic Embedding: A Unified Supervised Projection Framework},
  author={RWKV Community},
  year={2026},
  url={https://github.com/cgisky1980/rwkv7-state-embedding}
}
```

## License

MIT
