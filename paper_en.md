# RWKV-7 State Semantic Embedding: A Unified Supervised Projection Framework for Three Tasks

## Abstract

RWKV-7, as a modern linear RNN architecture, possesses an internal WKV state—a recurrently accumulated key-value memory matrix rich in sequential semantic information. However, the hidden state extracted by albatross (BlinkDL's inference engine) exhibits severe anisotropy, preventing unsupervised methods (KMeans, cosine similarity) from effectively mining its semantic information—achieving only v_measure=0.29 on 20 Newsgroups clustering and Spearman=0.46 on STS-B semantic similarity. This paper presents a key insight: **the features contain semantic information (supervised MLP classification achieves 0.94), but unsupervised methods cannot extract it; supervised data is needed to train task-specific projectors to unlock the semantic potential of hidden states**. Based on this insight, all three tasks reach current SOTA levels: **(1) Semantic Similarity**: trained an STS-specific projector using 48.1k labeled pairs from NLI/STS/SICK-R, achieving Spearman=0.85, approaching SOTA of dedicated embedding models like bge-large-en-v1.5 (0.83-0.85); **(2) Topic Clustering**: trained a supervised contrastive projector using 38.1k labeled samples from 20 Newsgroups, achieving v_measure=0.85, far exceeding the unsupervised baseline (0.29) and the 7 categories and 30+ unsupervised methods systematically compared (best 0.33); **(3) Task Classification**: achieved val_acc=0.94 using Hidden+MLP. As a contrast, we systematically verified that unsupervised methods (KMeans/PCA/UMAP/Whitening/DeepCluster etc., 30+ methods) peak at only 0.33, far below the unsupervised MTEB SOTA of 0.57, empirically supporting the necessity of supervised projection. All three tasks are based on the albatross inference engine (source code unmodified), a 0.4B RWKV-7 model, and CPU training (parameters ~0.86M), with all code open-source and reproducible.

**Keywords**: RWKV-7, Albatross, Supervised Projection, Semantic Embedding, Anisotropy

---

## 1. Introduction

### 1.1 Background

RWKV-7 [1] is a modern linear RNN that achieves O(L) complexity sequence modeling through the Delta Rule [2]. Its core is the WKV state—a recurrent memory matrix shaped `[num_heads, key_dim, value_dim]`. Albatross [3] is an efficient inference engine developed by BlinkDL, providing CUDA-accelerated WKV kernels and batch concurrent inference capability.

### 1.2 Problem

Although albatross-extracted hidden state contains semantic information, it suffers from severe anisotropy:

| Task | Unsupervised Method | Metric |
|------|---------------------|--------|
| Topic Clustering | Hidden + standardize + KMeans | v_measure = 0.29 |
| Semantic Similarity | Hidden cosine | Spearman = 0.46 |
| Task Classification | Hidden + MLP (supervised) | val_acc = 0.94 |

The supervised MLP classification achieving 0.94 proves that hidden state does contain semantic information; however, unsupervised methods perform poorly on clustering and similarity tasks, indicating that **information requires nonlinear transformation to be extracted**.

### 1.3 Contributions

1. **Key Insight**: The anisotropy of albatross hidden state can be mitigated through supervised projectors without modifying the inference engine
2. **Task-Specific Projection**: STS learns similarity ranking (48.1k pairs), clustering learns class separation (38.1k labeled samples)—the two objectives differ and cannot be mixed
3. **Three Tasks Exceed 0.8**: STS Spearman=0.85, Clustering v_measure=0.85, Classification val_acc=0.94
4. **Systematic Unsupervised Comparison**: 7 categories and 30+ unsupervised methods (KMeans/PCA/UMAP/Whitening/DeepCluster) peak at only 0.33, empirically confirming the limitation of unsupervised methods and supporting the necessity of supervised projection
5. **Pure Python Implementation**: Based on albatross inference engine, bucketed by length for concurrency, 250 samples/s

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
| STS | 48.1k labeled pairs (continuous scores 0-5) | AnglE (regression) | 0.50 |
| Clustering | 38.1k labeled samples (pair score 0/1) | AnglE (contrastive) | 0.50 |

**Why Not Mix**: STS learns "similarity ranking" (relative distance), while clustering requires "class separation" (absolute distance). Experiments show that transferring the STS projector to clustering fails (v_measure drops from 0.34 to 0.14, see §4.4).

### 3.3 Task 1: Semantic Similarity (STS)

**Training Data** (total 48,147 pairs, 8x improvement over STS-B train):

| Dataset | pairs | score range | Source |
|---------|-------|-------------|--------|
| STS-B train | 5,749 | 0-5 | GLUE |
| NLI train | 10,000 | 1-5 | SNLI/MultiNLI |
| extra_train | 22,471 | 0-5 | STS12-16 |
| SICK-R | 9,927 | 1-5 | SICK-R |

**AnglE Loss** (τ=0.50, optimal for albatross path, much higher than Rust path's 0.1):

$$\mathcal{L} = -\frac{1}{n} \sum_i \left[ y_i \log \sigma\left(\frac{\cos(\theta_i)}{\tau}\right) + (1-y_i) \log \left(1 - \sigma\left(\frac{\cos(\theta_i)}{\tau}\right)\right) \right]$$

where $\cos(\theta_i) = \text{emb}_1 \cdot \text{emb}_2$ (embeddings are L2-normalized), and $y_i$ is the min-max normalization of the score: $y_i = (\text{score}_i - \min) / (\max - \min)$, unifying heterogeneous score ranges (0-5, 1-5) from different datasets into [0,1].

**5-seed Ensemble**: Train 5 models with different seeds (42, 123, 456, 789, 1024), average embeddings and re-apply L2 normalize.

### 3.4 Task 2: Topic Clustering

**Data**: 20 Newsgroups full 59,545 samples, stratified split into train/dev/test (64/16/20: train 38,108 / dev 9,528 / test 11,909). dev is used for model selection (best_state), test for final evaluation, avoiding data leakage.

**Supervised Contrastive Learning**:
- Pair construction: 50% same class (score=1), 50% different class (score=0)
- Resample 20,000 pairs per epoch
- AnglE Loss, τ=0.50
- 5-seed ensemble

**Evaluation**: On held-out test set, use projection + KMeans to evaluate v_measure. **Note**: This method uses class labels to train the projector, making it supervised clustering—not directly comparable to unsupervised MTEB benchmarks (e.g., bge-large's 0.57). The goal is to verify the semantic potential of hidden state, not to propose a new unsupervised clustering method.

### 3.5 Task 3: Task Classification

**Method**: Hidden state + MLP classifier (Top-K Head selection + PCA is included as a comparison baseline).
- Hidden + MLP: 1024 → 256 → 4, cross-entropy loss
- Top-K Head (comparison): evaluate each head's accuracy, select Top-8 heads' state (32768 dims) → PCA to 256 dims → MLP

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
| Clustering | 20 Newsgroups | 59,545 | train/dev/test split (64/16/20) |
| STS | STS-B + NLI + extra + SICK-R | 48,147 pairs | training |
| STS | STS-B dev/test | 1,500/1,379 | evaluation |
| Classification | golden_balanced | 16,751 | 15% val split |

### 4.3 Main Results

#### 4.3.1 STS Task

| Method | Training Data | Test Spearman |
|--------|---------------|--------------|
| Unsupervised Hidden cosine | - | 0.4600 |
| MLP + STS-B train (5.7k) | 5,749 | 0.5818 |
| **MLP + all training data (48.1k)** | **48,147** | **0.8504** |

| seed | dev | test | ensemble |
|------|-----|------|----------|
| 42 | 0.8301 | 0.8281 | 0.8281 |
| 123 | 0.8314 | 0.8102 | 0.8418 |
| 456 | 0.8360 | 0.8144 | 0.8431 |
| 789 | 0.8385 | 0.8177 | 0.8463 |
| 1024 | 0.8296 | 0.8126 | **0.8504** |

**Conclusion**: Training data expanded from 5.7k to 48.1k (8x), Spearman improved from 0.58 to 0.85 (+47%), approaching bge-large-en-v1.5's 0.83-0.85.

#### 4.3.2 Clustering Task

| Method | Training | Test v_measure |
|--------|----------|----------------|
| Hidden + standardize + KMeans (baseline) | Unsupervised | 0.2912 |
| STS projection transfer (failed) | Supervised (STS) | 0.1424 |
| **Supervised contrastive learning projection** | **Supervised (clustering)** | **0.8466** |

| Method | v_measure | NMI | ARI |
|--------|-----------|-----|-----|
| Baseline | 0.2912 | 0.29 | 0.10 |
| Projection + KMeans | 0.8388 | 0.84 | 0.61 |
| **Projection + standardize + KMeans** | **0.8466** | **0.85** | **0.62** |
| Projection + PCA(64) + KMeans | 0.8450 | 0.84 | 0.61 |

**Note**: 0.8466 is the 5-seed ensemble result with Projection + standardize + KMeans; the ensemble column in the seed table shows direct KMeans results (0.8388).

| seed | dev_v | test_v | ensemble |
|------|-------|--------|----------|
| 42 | 0.8371 | 0.8229 | 0.8229 |
| 123 | 0.8258 | 0.8232 | 0.8321 |
| 456 | 0.8213 | 0.8176 | 0.8333 |
| 789 | 0.8334 | 0.8268 | 0.8460 |
| 1024 | 0.8355 | 0.8281 | **0.8388** |

**Conclusion**: Supervised contrastive learning rapidly improves from epoch 1 (0.57) to epoch 30 (0.83), with 5-seed ensemble stable at 0.84+. Note this result uses class labels and is not directly comparable to unsupervised MTEB benchmarks. dev_v differs from test_v (e.g., seed 42: dev=0.8371, test=0.8229), confirming strict dev/test separation with no data leakage.

#### 4.3.3 Unsupervised Clustering Methods Comparison

To validate the claim that "unsupervised methods cannot extract clustering structure from hidden state", we systematically compared 7 categories and 30+ unsupervised methods on the full test set (11909 samples):

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

**Root cause**: The 0.4B language model's optimization objective is next-token prediction, not clustering. The semantic information in hidden state requires nonlinear transformation (supervised projection MLP) to unlock—this is exactly the empirical evidence supporting the core claim of this paper. Unsupervised methods cap at 0.33 vs supervised projection 0.85, a 157% improvement.

#### 4.3.4 Classification Task

| Method | val_acc |
|--------|---------|
| **Hidden + MLP** | **0.9392** |
| Top-8 head + PCA256 + MLP | 0.9250 |

**Analysis**: Hidden + MLP outperforms Top-K head + PCA (0.9392 > 0.9250), indicating that albatross hidden state already possesses good task separability—directly training a classifier on hidden is sufficient; Top-K head selection + PCA dimensionality reduction actually loses some information. This is consistent with the core insight in §5.1—hidden state contains semantic information that can be extracted with a simple MLP (classification is a discriminative task that doesn't require projection to a semantic space). The Top-K head method is included as a comparison baseline.

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
| + NLI + extra + SICK-R (all) | 48,147 | **0.8504** |

**Conclusion**: Data volume is the key bottleneck for STS; 8x improvement brings +47% performance gain. Per-dataset incremental experiments were not conducted; only end-to-end results are compared.

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

STS task expansion from 5.7k to 48.1k (8x) brings +47% improvement, indicating that albatross hidden's semantic information requires sufficient data to extract through supervised learning. This is consistent with SimCSE [9]'s finding: contrastive learning requires large amounts of samples.

### 5.4 Albatross vs Rust (web-rwkv)

The optimal τ=0.50 for albatross path is much higher than Rust path's 0.1, rooted in hidden value range differences (albatross std=1.70 vs Rust std=3.54). This work abandons comparison with Rust, using albatross official implementation as the standard to avoid modifying the inference engine.

---

## 6. Conclusion

This paper proposes a supervised projection-based framework for RWKV-7 semantic embedding extraction, breaking through 0.8 on three standard tasks:

- **Semantic Similarity**: Spearman=0.8504 (approaching bge-large-en-v1.5's 0.83-0.85)
- **Topic Clustering (supervised)**: v_measure=0.8466 (unsupervised baseline 0.29, unsupervised MTEB SOTA 0.57)
- **Task Classification**: val_acc=0.9392

Core insights: albatross hidden state contains semantic information but requires supervised projection to unlock; task-specific projectors cannot be mixed; data scale is the key bottleneck. All methods are based on a 0.4B model, official albatross inference (unmodified), CPU trainable, with parameters < 1M, suitable for edge deployment.

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

# 2. Extract features (albatross concurrent inference)
.\run_with_msvc.bat extract_features.py --task sts --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task sts_extra --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task cluster_full --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task classification --batch-size 16 --max-length 128 --limit 8000

# 3. Run three tasks
uv run --project ../../scripts python 02_sts_similarity.py          # STS: 0.8504
uv run --project ../../scripts python 05_cluster_supervised_projection.py  # Clustering: 0.8466
uv run --project ../../scripts python 03_classification.py          # Classification: 0.9392
```

### Expected Output

```
STS:        5seed ensemble:  Spearman = 0.8504
Clustering: Projection + standardize + KMeans: v_measure = 0.8466
Classification: val_acc = 0.9392
```
