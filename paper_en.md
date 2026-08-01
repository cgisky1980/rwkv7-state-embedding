# RWKV-7 State Semantic Embedding: A Unified Supervised Projection Framework for Three Tasks

## Abstract

RWKV-7, as a modern linear RNN architecture, contains two levels of internal states: (1) **WKV state**—a recurrent memory matrix shaped `[num_heads, key_dim, value_dim]`, the core of the Delta Rule; (2) **Hidden state**—the output vector of each TMix layer. This paper finds that the hidden state extracted by albatross (BlinkDL's inference engine) contains rich semantic information but exhibits severe anisotropy, preventing unsupervised methods (KMeans, cosine similarity) from effectively mining it—achieving only v_measure=0.44 on 20 Newsgroups (sklearn full-text version) clustering and Spearman=0.46 on STS-B semantic similarity. As a comparison, we also tried various aggregation methods on WKV state (Q-Readout, row_sum, diag, etc.), yielding even worse clustering performance (best 0.11), indicating that hidden state is more suitable for semantic extraction than WKV state. This paper presents a key insight: **hidden state contains semantic information (supervised MLP classification achieves 0.93), but unsupervised methods cannot extract it; supervised data is needed to train task-specific projectors to unlock its semantic potential**. Based on this insight, all three tasks achieve the following results: **(1) Semantic Similarity**: trained an STS-specific projector using 46.9k labeled pairs from NLI/STS/SICK-R (strictly deduplicated, removing 1249 leak pairs overlapping with STS-B dev/test), achieving Spearman=0.8188; **(2) Topic Clustering**: Validated on two versions of 20 Newsgroups—MTEB short-text version (59,545 samples) supervised projection v_measure=0.9506 (+217% over unsupervised baseline 0.2912); sklearn full-text version (strict dedup + stratified 70/15/15 split) SupCon Loss v_measure=0.6660 (+50% over unsupervised baseline 0.4434); **(3) Task Classification**: achieved test_acc=0.9325 on an independent test set using Hidden+MLP. It should be emphasized that STS and clustering tasks use supervised projectors and are thus supervised methods; the comparison with unsupervised baselines is only to validate the claim that "hidden state requires supervised projection," not to claim superiority over unsupervised methods. All three tasks are based on the albatross inference engine (source code unmodified), a 0.4B RWKV-7 model, and CPU training (parameters ~3.15M), with all code open-source and reproducible.

**Keywords**: RWKV-7, Albatross, Supervised Projection, Semantic Embedding, Anisotropy

---

## 1. Introduction

### 1.1 Background

RWKV-7 [1] is a modern linear RNN that achieves O(L) complexity sequence modeling through the Delta Rule [2]. It contains two levels of internal states:

- **WKV state** $S \in \mathbb{R}^{\text{num\_heads} \times \text{key\_dim} \times \text{value\_dim}}$: recurrent memory matrix, the core of the Delta Rule, recursively updated at each time step
- **Hidden state** $h \in \mathbb{R}^{\text{hidden\_dim}}$: the output vector of each TMix (time mix) and CMix (channel mix) layer, an intermediate representation of forward propagation

This paper studies the semantic extraction capability of **hidden state**. WKV state was also experimented with as a baseline (§4.3.3), but its clustering performance was much lower than hidden state (0.11 vs 0.47). Albatross [3] is an efficient inference engine developed by BlinkDL, providing CUDA-accelerated WKV kernels and batch concurrent inference capability.

### 1.2 Problem

Although albatross-extracted hidden state contains semantic information, it suffers from severe anisotropy:

| Task | Unsupervised Method | Metric |
|------|---------------------|--------|
| Topic Clustering | Hidden + standardize + KMeans | v_measure = 0.47 (sklearn full-text version) |
| Semantic Similarity | Hidden cosine | Spearman = 0.46 |
| Task Classification | Hidden + MLP (supervised) | test_acc = 0.93 |

The supervised MLP classification achieving 0.93 proves that hidden state does contain semantic information; however, unsupervised methods perform poorly on clustering and similarity tasks, indicating that **information requires nonlinear transformation to be extracted**.

### 1.3 Contributions

1. **Key Insight**: The anisotropy of albatross hidden state can be mitigated through supervised projectors without modifying the inference engine
2. **Task-Specific Projection**: STS learns similarity ranking (46.9k pairs, strictly deduplicated), clustering learns class separation (12.8k labeled samples, sklearn full-text version with strict deduplication + split)—the two objectives differ and cannot be mixed
3. **Strict Experimental Paradigm**: STS training data is strictly deduplicated against STS-B dev/test (removing 1249 overlapping pairs); clustering uses sklearn full-text 20NG with strict deduplication + stratified 70/15/15 split, where train is for training / dev for best_state selection / test for held-out evaluation; classification adds an independent test set, with head selection using only dev
4. **Three Tasks Results**: STS Spearman=0.8188; clustering validated on two datasets (MTEB short-text v=0.9506, sklearn full-text SupCon v=0.6660, +217%/+50% over unsupervised baselines respectively); classification test_acc=0.9325. Note that STS and clustering use supervised projectors and are thus supervised methods
5. **Systematic Unsupervised Comparison**: 7 categories and 30+ unsupervised methods (KMeans/PCA/UMAP/Whitening/DeepCluster) peak at only 0.33 on the MTEB short-text version, empirically confirming the limitation of unsupervised methods and supporting the necessity of supervised projection
6. **Pure Python Implementation**: Based on albatross inference engine, bucketed by length for concurrency, 250 samples/s

---

## 2. Related Work

### 2.1 RWKV and Albatross

RWKV [1] combines Transformer's parallel training with RNN's inference efficiency. RWKV-7 introduces Delta Rule updates: $S_t = \text{Diag}(w_t) S_{t-1} + \beta_t (v_t - S_{t-1}^\top k_t) k_t^\top$. Albatross [3] is the official PyTorch inference implementation, providing CUDA WKV kernels and `forward_seq_batch` for batch concurrent inference.

### 2.2 Embedding Evaluation Benchmarks

- **MTEB** [4]: Massive Text Embedding Benchmark
- **STS-Benchmark** [5]: Semantic Textual Similarity, Spearman correlation
- **20 Newsgroups** [6]: 20-category topic document clustering
- **NLI / SICK-R** [7]: Natural Language Inference and sentence pair scoring

### 2.3 Anisotropy and Contrastive Learning

- **AnglE** [8]: Angle-based contrastive loss, mitigating embedding anisotropy
- **SimCSE** [9]: Contrastive learning to alleviate anisotropy, but requires many negative samples
- **Whitening** [10]: Linear transformation to eliminate anisotropy, but destroys nonlinear structure

---

## 3. Method

### 3.1 Albatross Feature Extraction

**Model**: RWKV-7-g1 0.4B (hidden=1024, 24 layers, 16 heads, head_size=64)

**Features**:
- **Hidden state** $h \in \mathbb{R}^{1024}$: Mean pooling of the last layer's FFN output
- **WKV state** $S \in \mathbb{R}^{16 \times 64 \times 64}$: Recurrent memory matrix at layer 12

**Batch Concurrent Extraction**: Bucketed by token length, sequences within each bucket have identical length (no padding needed), using albatross's `forward_seq_batch` for concurrent inference at 250 samples/s (vs 70 samples/s for single sequence, 3.6x speedup).

**Important Note**: This work entirely uses the albatross inference engine without modifying any source code. The feature extraction script only replicates the `forward_seq_batch` loop structure to simultaneously extract intermediate-layer state and hidden (the original function only returns logits).

### 3.2 Unified Projection Framework

Core idea: Use a 2-hidden-layer MLP to project hidden state into a 128-dimensional semantic space.

$$\text{emb} = \text{normalize}(\text{MLP}(\text{BatchNorm}(h)))$$

**MLP Architecture** (parameters ~0.86M):

```
Input(1024) → BatchNorm → Linear(1024→512) → GELU → LayerNorm → Dropout(0.2)
           → Linear(512→512) → GELU → LayerNorm → Dropout(0.2)
           → Linear(512→128) → L2 Normalize
```

**Training Objectives Differ by Task** (key design):

| Task | Training Data | Loss | Temperature τ |
|------|---------------|------|----------------|
| STS | 46.9k labeled pairs (continuous scores 0-5, deduplicated) | AnglE (regression) | 0.50 |
| Clustering | 12.8k labeled samples (pair score 0/1, sklearn full-text version split) | AnglE (contrastive) | 0.30 |

**Why Not Mix**: STS learns "similarity ranking" (relative distance), while clustering requires "class separation" (absolute distance). Experiments show that transferring the STS projector to clustering fails (v_measure drops from 0.34 to 0.14, see §4.4).

### 3.3 Task 1: Semantic Similarity (STS)

**Training Data** (total 46,898 pairs after strict deduplication):

Originally 48,147 pairs; after removing pairs overlapping with STS-B dev/test, 46,898 pairs remain (1,249 pairs removed, 2.59%):

| Dataset | Original pairs | After dedup | score range | Source |
|---------|---------------|-------------|-------------|--------|
| STS-B train | 5,749 | 5,076 | 0-5 | GLUE |
| NLI train | 10,000 | 9,952 | 1-5 | SNLI/MultiNLI |
| extra_train | 22,471 | 21,943 | 0-5 | STS12-16 |
| SICK-R | 9,927 | 9,927 | 1-5 | SICK-R |

**Deduplication Rule** (strictest, single-sentence level): Any training pair containing a sentence that appears in the STS-B dev/test sentence set is removed, preventing the model from memorizing eval sentence representations.

**AnglE Loss** (τ=0.50, optimal for albatross path, much higher than Rust path's 0.1):

$$\mathcal{L} = -\frac{1}{n} \sum_i \left[ y_i \log \sigma\left(\frac{\cos(\theta_i)}{\tau}\right) + (1-y_i) \log \left(1 - \sigma\left(\frac{\cos(\theta_i)}{\tau}\right)\right) \right]$$

where $\cos(\theta_i) = \text{emb}_1 \cdot \text{emb}_2$ (embeddings are L2-normalized), and $y_i$ is the min-max normalization of the score: $y_i = (\text{score}_i - \min) / (\max - \min)$, unifying heterogeneous score ranges (0-5, 1-5) from different datasets into [0,1].

**5-seed Ensemble**: Train 5 models with different seeds (42, 123, 456, 789, 1024), average embeddings and re-apply L2 normalize.

### 3.4 Task 2: Topic Clustering

**Data Source**: sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes')), removing headers/footers/quotes to prevent the model from memorizing meta-information. Original 18,846 samples → filter 515 empty documents → text-normalized hash deduplication removes 78 (0.43%, cross-group cross-posts) → 18,253 samples.

**Stratified Split (70/15/15)**: train 12,777 / dev 2,738 / test 2,738. dev is used for model selection (best_state), test for final evaluation (held-out, not involved in training).

**Supervised Contrastive Learning**:
- Pair construction: 50% same class (score=1), 50% different class (score=0)
- Resample 80,000 pairs per epoch
- AnglE Loss, τ=0.30
- 5-seed ensemble

**Evaluation**: On held-out test set, use projection + KMeans to evaluate v_measure. **Note**: This method uses class labels to train the projector, making it supervised clustering—not directly comparable to unsupervised MTEB benchmarks (e.g., bge-large's 0.57). The goal is to verify the semantic potential of hidden state, not to propose a new unsupervised clustering method.

### 3.5 Task 3: Task Classification

**Method**: Hidden state + MLP classifier (Top-K Head selection + PCA is included as a comparison baseline).
- Hidden + MLP: 1024 → 256 → 4, cross-entropy loss
- Top-K Head (comparison): evaluate each head's accuracy, select Top-8 heads' state (32768 dims) → PCA to 256 dims → MLP

**Strict Split (70/15/15)**: train 70% / dev 15% (head selection + early stopping) / test 15% (final evaluation, held-out). Head selection is ranked only on dev, never touching test, avoiding overfitting.

---

## 4. Experiments

### 4.1 Experimental Setup

- **Backbone**: RWKV-7-g1 0.4B (`rwkv7-g1d-0.4b-20260210-ctx8192.pth`, BlinkDL/rwkv7-g1)
- **Inference Engine**: albatross (official PyTorch + CUDA WKV kernel)
- **Training Hardware**: CPU (Intel i7-12700K) + RTX 2080 Ti (backbone inference only)
- **Training Framework**: PyTorch 2.0+ (CPU mode for Projection training)
- **State Layer**: L12 (experimentally determined optimal)
- **Max Sequence Length**: 128 tokens (truncated)

### 4.2 Datasets

| Task | Dataset | Samples | Usage |
|------|---------|---------|-------|
| Clustering | 20NG (sklearn full-text, deduplicated) | 18,253 | train/dev/test split (70/15/15) |
| Clustering (unsupervised comparison) | 20NG (MTEB short-text) | 59,545 | unsupervised evaluation (comparable to MTEB leaderboard) |
| STS | STS-B train + NLI + extra + SICK-R (deduplicated) | 46,898 pairs | training |
| STS | STS-B dev/test | 1,500/1,379 | evaluation |
| Classification | golden_balanced | 16,751 | train/dev/test split (70/15/15) |

### 4.3 Main Results

#### 4.3.1 STS Task

| Method | Training Data | Test Spearman |
|--------|---------------|--------------|
| Unsupervised Hidden cosine | - | 0.4600 |
| MLP + STS-B train (5.7k, not deduplicated) | 5,749 | 0.5818 |
| MLP + all training data (46.9k, strictly deduplicated, baseline) | 46,898 | 0.8053 |
| **MLP + all training data (optimal config: h1024, out512, drop0.1)** | **46,898** | **0.8188** |

| seed | dev | test | ensemble |
|------|-----|------|----------|
| 42 | 0.8125 | 0.7786 | 0.7786 |
| 123 | 0.8071 | 0.7791 | 0.8006 |
| 456 | 0.8021 | 0.7900 | 0.8127 |
| 789 | 0.8022 | 0.7808 | 0.8157 |
| 1024 | 0.8102 | 0.7874 | 0.8188 |

**Configuration**: hidden_dim=1024, output_dim=512, dropout=0.1, τ=0.5, 50 epoch, 5 seed

**Conclusion**: After strict deduplication (removing 1249 pairs overlapping with STS-B dev/test), baseline Spearman dropped from 0.8504 to 0.8053 (-0.0451), and optimal config dropped from 0.8680 to 0.8188 (-0.0492), confirming data leakage impact. The optimal config still improves +0.0135 over baseline, and the result is more honest, still significantly outperforming the unsupervised baseline (0.46).

**Comparison with Supervised Embedding Models**:

| Method | Type | STS-B Spearman | Parameters |
|--------|------|----------------|------------|
| **Ours (RWKV-7 0.4B + Supervised Projection, deduplicated)** | **Supervised** | **0.8188** | **3.15M (projector)** |
| bge-large-en-v1.5 [12] | Supervised | ~0.85 | 335M |
| all-MiniLM-L6-v2 [13] | Supervised | ~0.86 | 22M |
| Unsupervised Hidden cosine | Unsupervised | 0.46 | - |

*Supervised model data source: MTEB Leaderboard [14]. Training data and evaluation splits may differ slightly across models; this is an approximate comparison.*

Our method achieves STS-B Spearman of 0.8188 with only 3.15M projector parameters (0.4B language model frozen, only projector trained), lower than bge-large-en-v1.5 (335M full fine-tuning) at ~0.85 and all-MiniLM-L6-v2 (22M) at ~0.86. The gap is mainly attributable to the representation capacity ceiling of the 0.4B language model and the training data scale. It should be emphasized that this method requires additional supervised training data (46.9k pairs), and the 0.4B language model parameters are not included in the comparison.

#### 4.3.2 Clustering Task (Supervised Projection + KMeans)

This paper evaluates clustering on two complementary datasets:

- **MTEB short-text version** (59,545 samples, title-level): Comparable with MTEB leaderboard, data download is difficult
- **sklearn full-text version** (18,253 samples, strict dedup + stratified 70/15/15 split): Reproducible by anyone, held-out evaluation

**Main results comparison**:

| Dataset | Samples | Unsupervised baseline | Supervised (AnglE) | **Supervised (SupCon)** |
|---------|---------|----------------------|---------------------|-------------------------|
| MTEB short-text | 59,545 | 0.2912 | 0.9506 | - |
| **sklearn full-text** | 18,253 | 0.4434 ± 0.0146 | 0.6373 | **0.6660** |

**sklearn full-text detailed results**:

| Method | Training Type | Test v_measure |
|--------|--------------|----------------|
| Hidden + standardize + KMeans (baseline, 10 seeds) | Unsupervised | 0.4434 ± 0.0146 |
| STS projection transfer (failed) | Supervised (STS) | 0.1424 |
| Supervised contrastive (AnglE Loss, τ=0.3, 80k pairs) | Supervised (Clustering) | 0.6373 |
| **Supervised contrastive (SupCon Loss, τ=0.07, batch=320)** | **Supervised (Clustering)** | **0.6660** |

| seed | dev_v | test_v | ensemble |
|------|-------|--------|----------|
| 42 | 0.6351 | 0.6402 | 0.6402 |
| 123 | 0.6374 | 0.6378 | 0.6528 |
| 456 | 0.6281 | 0.6458 | 0.6614 |
| 789 | 0.6338 | 0.6473 | 0.6647 |
| 1024 | 0.6355 | 0.6418 | **0.6660** |

**Configuration**: SupCon Loss, τ=0.07, batch=320 (PK sampler, 20 classes × 16), dropout=0.1, 30 epochs, 5-seed ensemble

**MTEB short-text detailed results**:

| Configuration | dev_v | test_v (5-seed ensemble) |
|----------------|-------|---------------------------|
| baseline (Hidden + standardize + KMeans, 10 seeds) | - | 0.2912 |
| AnglE (τ=0.3, 80k pairs, drop0.1) | 0.9497 | **0.9506** |

**Note**: sklearn full-text test set (2738 samples) is held-out, not used in training; MTEB short-text version uses 64/16/20 split. The unsupervised baseline is the mean of 10 KMeans random_states on the test set. MTEB short-text results are fully reproduced (baseline=0.2912, supervised=0.9506); sklearn full-text results are slightly lower due to GPU non-determinism in feature extraction (baseline=0.4434/AnglE=0.6373/SupCon=0.6660), but the relative ordering SupCon > AnglE > baseline is consistent.

**Analysis of differences between datasets**:
1. sklearn full-text supervised result (0.6660) is much lower than MTEB full (0.9506), due to insufficient training data (12k vs 38k) + harder long-text clustering
2. But sklearn full-text unsupervised baseline (0.4434) is higher than MTEB full (0.2912), because removing headers/footers/quotes yields cleaner samples
3. Both datasets validate the core thesis: supervised projection significantly outperforms unsupervised methods (sklearn +50%, MTEB +217%)

**Why MTEB short-text v_measure=0.9506 has no data leakage**:

Since this result is notably higher than the sklearn full-text version (0.6660), we detail the data split and evaluation pipeline here to confirm no leakage. **A common confusion must be clarified first**: an earlier version mistakenly labeled "Projection + standardize + KMeans = 0.8466" (a supervised-projection result) as the unsupervised baseline, creating the false impression that "the unsupervised baseline already reaches 0.8466." After rerunning `01b_clustering_unsupervised.py` (10-seed KMeans) and `05_cluster_supervised_projection.py` (5 seeds supervised projection), the **true unsupervised baseline of the MTEB short-text version is 0.2912** (Hidden + standardize + KMeans), consistent with the 0.2912 in §4.3.3. The 0.8466 is a supervised-projection result, not an unsupervised baseline.

1. **Data Split (64/16/20 stratified split)**: The 59,545 samples are split via stratified sampling by class labels into:
   - train: 38,109 samples (64%) — used only to train the projector
   - dev: 9,527 samples (16%) — used only for best_state model selection
   - test: 11,909 samples (20%) — **fully held-out, never used in training or model selection**

2. **Evaluation Pipeline (unsupervised clustering + supervised metric)**:
   - The projector is trained only on train; best_state is selected by v_measure on dev
   - Final evaluation runs on test: project test hidden states to 128-d via the trained projector, then cluster with **KMeans (unsupervised, n_clusters=20)**
   - **Class labels are used only to compute the v_measure metric, not in the KMeans clustering step**, so there is no label leakage

3. **Why the high result is reasonable (supervised projection unlocks the semantic potential of hidden state)**:
   - The MTEB short-text **unsupervised baseline is only 0.2912** (KMeans directly on hidden states), consistent with the 0.2912 in §4.3.3, indicating severe hidden-state anisotropy that unsupervised methods cannot extract clustering structure from
   - Supervised projection lifts v_measure from 0.2912 to 0.9506 (**+217%**), an even larger gain than the +50% on the sklearn full-text version (0.4434→0.6660), indicating that supervised projection unlocks more semantic potential in the short-text scenario
   - Short-text samples have fewer tokens and more concentrated semantics, making it easier for supervised projection to learn clear inter-class separation boundaries; long texts are affected by noise and multiple topics, leaving less room for supervised-projection gains

4. **No-overfitting evidence**: dev_v=0.9497 is close to test_v=0.9506 (after 5-seed ensemble), indicating the best_state selected on dev performs equally well on test, with no dev→test overfitting

**Conclusion**: SupCon Loss (batch-based, same-class samples as positives) is more suitable for clustering than AnglE Loss (pair-based), improving +0.0287 on sklearn full-text (0.6373→0.6660). dev_v and test_v are close (e.g., seed 42: dev=0.6351, test=0.6402), indicating no overfitting. Note this result uses class labels and is not directly comparable to unsupervised MTEB benchmarks.

#### 4.3.3 Unsupervised Clustering Methods Comparison (MTEB Short-text Version)

To validate the claim that "unsupervised methods cannot extract clustering structure from hidden state", we systematically compared 7 categories and 30+ unsupervised methods on the official MTEB 20NG short-text version (59,545 samples, title-level, comparable to the MTEB leaderboard):

| Method Category | Best Method | Best v_measure |
|-----------------|-------------|----------------|
| Hidden + KMeans (baseline) | L23 + standardize | 0.2912 |
| Nonlinear feature transform | L23 + sign·sqrt + standardize | 0.3181 |
| Nonlinear dim. reduction (UMAP) | UMAP(32) + KMeans | 0.3263 |
| Multi-layer concatenation | Last3 (L16+L20+L23) | 0.2553 |
| Layer difference | L23−L0 + sign·sqrt | 0.3087 |
| Whitening / top-k PC removal | PCA whitening 256 | 0.1629 |
| WKV state aggregation stats | row_mean + col_mean | 0.1020 |
| DeepCluster self-supervised iter. | Round 1 ensemble | 0.3021 |

**Analysis**:

1. **Linear methods cap at ~0.32**: KMeans, PCA, Whitening and other linear methods peak at 0.32, far below the unsupervised MTEB SOTA of 0.57
2. **Nonlinear dim. reduction no breakthrough**: UMAP(32) reaches 0.3263 (slightly better than linear), but far from breakthrough, indicating the problem is not in the dimensionality reduction method
3. **Multi-layer concatenation degrades**: Multi-layer concat (0.26) is worse than single-layer L23 (0.32), indicating shallow-layer noise dilutes deep-layer semantics
4. **DeepCluster pseudo-label degradation**: Initial KMeans pseudo-labels are only 0.30, and iteration cannot self-improve (0.30→0.30), proving pseudo-label quality is insufficient to drive MLP to learn effective projections
5. **WKV state has no clustering info**: Various aggregation stats of state (row_sum, diag, trace, etc.) peak at only 0.10, indicating albatross's WKV state has too small value range (std=0.13); clustering info resides in hidden, not state

**Root cause**: The 0.4B language model's optimization objective is next-token prediction, not clustering. The semantic information in hidden state requires nonlinear transformation (supervised projection MLP) to unlock—this is exactly the empirical evidence supporting the core claim of this paper. Unsupervised methods cap at 0.33 (MTEB short-text version) / 0.44 (sklearn full-text version) vs supervised projection 0.6660 (sklearn full-text, SupCon) / 0.9506 (MTEB full, AnglE), a 50% / 227% improvement.

#### 4.3.4 Classification Task

| Method | dev_acc | test_acc |
|--------|---------|---------|
| **Hidden + MLP** | 0.9381 | **0.9325** |
| Top-8 head + PCA256 + MLP | 0.9247 | 0.9208 |

**Strict Split (70/15/15)**: dev is used for head selection + early stopping, test is held-out for final evaluation. Head selection is ranked only on dev, never touching test.

**Analysis**: Hidden + MLP outperforms Top-K head + PCA (test_acc 0.9325 > 0.9208), indicating that albatross hidden state already possesses good task separability—directly training a classifier on hidden is sufficient; Top-K head selection + PCA dimensionality reduction actually loses some information. dev_acc and test_acc are close (0.9381 vs 0.9325), indicating no overfitting. This is consistent with the core insight in §5.1—hidden state contains semantic information that can be extracted with a simple MLP (classification is a discriminative task that doesn't require projection to a semantic space). The Top-K head method is included as a comparison baseline.

### 4.4 Ablation Studies

#### 4.4.1 STS Projection Transfer to Clustering (Failed)

This experiment uses a sampled subset of 2000 samples (100 per class), different from the full test set (11909 samples) in §4.3.2:

| Method | v_measure |
|--------|-----------|
| Baseline (Hidden + standardize, sampled 2000) | 0.3426 |
| STS Projection + KMeans | 0.1424 |
| STS Projection + standardize + KMeans | 0.1305 |
| STS Projection + PCA(64) + KMeans | 0.1284 |
| Hidden + STS Projection concat | 0.2756 |

**Failure Reason**: STS learns "similarity ranking" (relative distance); after L2 normalize, embedding std=0.0884 is too small, compressing inter-class distances. Clustering requires "class separation" (absolute distance)—the two objectives differ.

#### 4.4.2 Training Data Scale Impact (STS)

| Training Data | pairs | Test Spearman |
|---------------|-------|--------------|
| STS-B train only | 5,749 | 0.5818 |
| + NLI + extra + SICK-R (all, not deduplicated) | 48,147 | 0.8504 (leaked) |
| + NLI + extra + SICK-R (all, strictly deduplicated, baseline) | 46,898 | 0.8053 |
| + NLI + extra + SICK-R (all, strictly deduplicated, optimal) | 46,898 | **0.8188** |

**Conclusion**: Data volume is the key bottleneck for STS task, 8x increase brings +40% performance improvement (0.5818→0.8188). The non-deduplicated version 0.8504 was inflated due to training data sentence overlap with dev/test, dropping to 0.8188 (optimal config) as the true generalization result.

### 4.5 Summary of Failed Directions

| Method | Result | Failure Reason |
|--------|--------|----------------|
| STS projection transfer to clustering | 0.14 | STS learns similarity ranking, clustering needs class separation |
| Unsupervised KMeans | 0.29 | Hidden anisotropy severe, linear methods cannot extract |
| Unsupervised Hidden cosine (STS) | 0.46 | Anisotropy causes 0.46 << 0.85 |
| UMAP nonlinear dim. reduction | 0.33 | Nonlinear reduction also ineffective, problem not in reduction method |
| DeepCluster self-supervised iter. | 0.30 | Pseudo-label quality too low, cannot self-improve |
| Multi-layer hidden concatenation | 0.26 | Shallow-layer noise dilutes deep-layer semantics |
| Whitening / top-k PC removal | 0.16 | Anisotropy fix actually degrades performance |
| Pure WKV state (albatross, Q-Readout) | 0.11 | State value range small, std=0.13 |
| WKV state aggregation stats | 0.10 | row_sum/diag/trace etc. have no clustering info |
| Disable v_first mechanism | 0.16 | State std only 0.16, below Rust level |
| fp32 state (vs fp16) | 0.13 | std 0.13≈0.13, precision not the root cause |

---

## 5. Discussion

### 5.1 Core Insight: Supervised Projection Unlocks Hidden Semantic Potential

The anisotropy of albatross hidden state (unsupervised STS only 0.46) is not a feature defect, but evidence that **unsupervised methods cannot extract nonlinear structure**. Supervised MLP classification achieving 0.94 proves the features contain information; only a supervised projector (MLP + AnglE Loss) is needed to map it to a linearly separable semantic space.

### 5.2 Task-Specific Projectors Cannot Be Mixed

| Task | Objective | Projector Behavior |
|------|-----------|---------------------|
| STS | Similarity ranking | Preserves relative distances, L2 normalize compresses inter-class |
| Clustering | Class separation | Amplifies inter-class distances, preserves absolute positions |

STS projector transfer to clustering fails (0.14 < 0.34 baseline), proving that **projectors must align with task objectives**.

### 5.3 Data Scale is Key

STS task expansion from 5.7k to 46.9k (8x) brings +39% improvement, indicating that albatross hidden's semantic information requires sufficient data to extract through supervised learning. This is consistent with SimCSE [9]'s finding: contrastive learning requires large amounts of samples.

### 5.4 Albatross vs Rust (web-rwkv)

The optimal τ=0.50 for albatross path is much higher than Rust path's 0.1, rooted in hidden value range differences (albatross std=1.70 vs Rust std=3.54). This work abandons comparison with Rust, using albatross official implementation as the standard to avoid modifying the inference engine.

---

## 6. Conclusion

This paper proposes a supervised projection-based framework for RWKV-7 hidden state semantic embedding extraction, adopting a strict experimental paradigm (STS training data deduplication, clustering on sklearn full-text version with strict deduplication + stratified split, classification with an independent test set), achieving the following results on three standard tasks:

- **Semantic Similarity (supervised)**: Spearman=0.8188 (after deduplication), lower than bge-large-en-v1.5 (~0.85, 335M) and all-MiniLM-L6-v2 (~0.86, 22M), but with only 3.15M projector parameters. The gap is mainly attributable to the representation capacity ceiling of the 0.4B language model
- **Topic Clustering (supervised)**: Validated on two datasets—MTEB short-text v_measure=0.9506 (+217% over unsupervised baseline 0.2912); sklearn full-text SupCon Loss v_measure=0.6660 (+50% over unsupervised baseline 0.4434). SupCon improves +0.0287 over AnglE on sklearn version
- **Task Classification**: test_acc=0.9325 (independent test set, head selection uses only dev)

Core insights: albatross hidden state contains semantic information but requires supervised projection to unlock; task-specific projectors cannot be mixed; data scale is the key bottleneck. WKV state shows much lower clustering performance than hidden state (0.11 vs 0.44), indicating hidden state is more suitable for semantic extraction. All methods are based on a 0.4B model, albatross inference engine (source code unmodified), CPU trainable with ~3.15M parameters, suitable for edge deployment.

**Honest Disclosure**: The early-version reported STS 0.8504 was inflated due to data leakage (STS training data overlapped with STS-B test by 1249 pairs); the clustering 0.8466 was actually a "Projection + standardize + KMeans" result that was mistakenly labeled as the unsupervised baseline. After rerunning `01b_clustering_unsupervised.py` (10 seeds) and `05_cluster_supervised_projection.py` (5 seeds), and reran all three supervised clustering experiments for verification, the true unsupervised baseline of the MTEB short-text version is 0.2912 (consistent with the 0.2912 in §4.3.3). This version has been strictly deduplicated, uses held-out evaluation, and corrects the baseline-labeling error, giving more honest results.

---

## References

[1] RWKV. RWKV-7: Linear RNN with Delta Rule. https://www.rwkv.cn/

[2] Delta Rule. https://en.wikipedia.org/wiki/Delta_rule

[3] BlinkDL. Albatross: Efficient RWKV-7 Inference Engine. https://github.com/BlinkDL/Albatross

[4] Muennighoff, N. et al. MTEB: Massive Text Embedding Benchmark. arXiv:2210.07316

[5] Cer, D. et al. STS-Benchmark. SemEval 2017

[6] 20 Newsgroups. http://qwone.com/~jason/20Newsgroups/

[7] Marelli, M. et al. SICK: A Sentiment Inference Corpus. LREC 2014

[8] Li, X. et al. AnglE: Angle-optimized Contrastive Loss. arXiv:2309.12871

[9] Gao, T. et al. SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP 2021

[10] Su, J. et al. Whitening Sentence Representations for Better Semantics. arXiv:2103.15316

[11] Reimers, N. and Gurevych, I. Sentence-BERT. EMNLP 2019

[12] Xiao, S. et al. BAAI/bge-large-en-v1.5. https://huggingface.co/BAAI/bge-large-en-v1.5

[13] Wang, W. et al. sentence-transformers/all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[14] MTEB Leaderboard. https://huggingface.co/spaces/mteb/leaderboard

---

## Appendix A: Reproduction Guide

### Requirements

- Python 3.10+, uv (package manager)
- CUDA GPU (for albatross inference)
- Windows requires MSVC (for CUDA extension compilation)

### Reproduction Commands

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# 1. Download model and albatross source
uv run --project ../../scripts python 00_setup.py

# 2. Extract features
.\run_with_msvc.bat extract_features.py --task sts --sts-subdir sts_dedup --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task sts_extra --sts-subdir sts_dedup --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task cluster_sklearn --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task classification --batch-size 16 --max-length 128 --limit 8000

# 3. Run three tasks
uv run --project ../../scripts python 02_sts_similarity.py --data-dir ../../data/sts_dedup --device cuda --n-epochs 50 --hidden-dim 1024 --output-dim 512 --dropout 0.1  # STS: 0.8188
uv run --project ../../scripts python 07_cluster_supcon.py --device cuda  # Clustering: sklearn full-text: 0.6660 (SupCon)
uv run --project ../../scripts python 06_cluster_sklearn.py --device cuda --temperature 0.3 --n-pairs 80000 --dropout 0.1  # Clustering: sklearn full-text: 0.6373 (AnglE)
# MTEB short-text: 0.9506 (requires cluster_full feature extraction first, see 05_cluster_supervised_projection.py)
uv run --project ../../scripts python 03_classification.py          # Classification: 0.9325
```

### Expected Output

```
STS:    5seed ensemble:  Spearman = 0.8188
Clustering:    5seed ensemble:  v_measure = 0.6660 (sklearn SupCon) / 0.9506 (MTEB AnglE)
Classification:    test_acc = 0.9325
```
