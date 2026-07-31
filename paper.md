# RWKV-7 Hidden State 语义嵌入：基于监督投影的三任务统一框架

## 摘要

RWKV-7 作为现代线性 RNN 架构，其核心包含两个层级的内部状态：(1) **WKV state**——形状为 `[num_heads, key_dim, value_dim]` 的递归记忆矩阵，是 Delta Rule 的核心；(2) **Hidden state**——每层 TMix 的输出向量。本文研究发现，albatross（BlinkDL 推理引擎）提取的 hidden state 蕴含丰富的语义信息，但存在严重的各向异性，无监督方法（KMeans、余弦相似度）无法有效挖掘——在 20 Newsgroups 聚类上 v_measure 仅 0.29，STS-B 语义相似度 Spearman 仅 0.46。作为对比，我们也尝试了 WKV state 的多种聚合方法（Q-Readout、row_sum、diag 等），聚类效果更差（最高 0.11），说明 hidden state 比 WKV state 更适合语义提取。本文提出一个关键洞察：**hidden state 中蕴含语义信息（监督 MLP 分类达 0.94），但无监督方法无法提取，需要用监督数据训练专用投影器（Projection）来释放其语义潜力**。基于此洞察，三任务均取得与监督嵌入模型可比的结果：**(1) 语义相似度**：用 NLI/STS/SICK-R 共 48.1k 标注对训练 STS 专用投影器，Spearman=0.85，与监督训练的专用嵌入模型 bge-large-en-v1.5（STS-B ~0.85）相当；**(2) 主题聚类**：用 20 Newsgroups 38.1k 标注样本训练监督对比投影器，v_measure=0.85，远超无监督 baseline（0.29）及系统对比的 7 类 30+ 种无监督方法（最高 0.33）；**(3) 任务分类**：用 Hidden+MLP 达 val_acc=0.94。需要强调的是，STS 和聚类任务使用了监督投影器，属于监督方法，与无监督 baseline 的对比仅用于验证"hidden state 需要监督投影"这一论点，而非声称超越无监督方法。三任务均基于 albatross 推理引擎（无修改源码）、0.4B RWKV-7 模型、CPU 训练（参数量 ~0.86M），所有代码开源可复现。

**关键词**：RWKV-7, Albatross, 监督投影, 语义嵌入, 各向异性

---

## 1. 引言

### 1.1 背景

RWKV-7 [1] 是一种现代线性 RNN，通过 Delta Rule [2] 实现 O(L) 复杂度的序列建模。其内部包含两个层级的状态：

- **WKV state** $S \in \mathbb{R}^{\text{num\_heads} \times \text{key\_dim} \times \text{value\_dim}}$：递归记忆矩阵，是 Delta Rule 的核心，在每个时间步递归更新
- **Hidden state** $h \in \mathbb{R}^{\text{hidden\_dim}}$：每层 TMix（时间混合）和 CMix（通道混合）的输出向量，是前向传播的中间表示

本文研究的是 **hidden state** 的语义提取能力。WKV state 作为对比基线也进行了实验（§4.3.3），但其聚类效果远低于 hidden state（0.11 vs 0.29）。Albatross [3] 是 BlinkDL 开发的高效推理引擎，提供 CUDA 加速的 WKV kernel 和批量并发推理能力。

### 1.2 问题

Albatross 提取的 hidden state 虽然蕴含语义信息，但存在严重的各向异性问题：

| 任务 | 无监督方法 | 指标 |
|------|-----------|------|
| 主题聚类 | Hidden + standardize + KMeans | v_measure = 0.29 |
| 语义相似度 | Hidden cosine | Spearman = 0.46 |
| 任务分类 | Hidden + MLP（监督） | val_acc = 0.94 |

分类任务的监督 MLP 能达到 0.94，证明 hidden 中确实存在语义信息；但无监督方法在聚类和相似度任务上表现差，说明**信息需要非线性变换才能提取**。

### 1.3 贡献

1. **关键洞察**：albatross hidden state 的各向异性问题可通过监督投影器缓解，无需修改推理引擎
2. **任务专用投影**：STS 学相似度排序（48.1k pairs），聚类学类间分离（38.1k labeled samples），两者目标不同不可混用
3. **三任务与监督嵌入模型可比**：STS Spearman=0.85（与 bge-large-en-v1.5 ~0.85 相当）、监督聚类 v_measure=0.85、分类 val_acc=0.94。注意 STS 和聚类使用了监督投影器，属监督方法
4. **无监督方法系统对比**：7 类 30+ 种无监督方法（KMeans/PCA/UMAP/Whitening/DeepCluster）最高仅 0.33，实证无监督方法的局限，支撑监督投影的必要性
5. **全 Python 实现**：基于 albatross 推理引擎，按长度分桶并发，250 samples/s

---

## 2. 相关工作

### 2.1 RWKV 与 Albatross

RWKV [1] 结合了 Transformer 的并行训练和 RNN 的推理效率。RWKV-7 引入 Delta Rule 更新：$S_t = \text{Diag}(w_t) S_{t-1} + \beta_t (v_t - S_{t-1}^\top k_t) k_t^\top$。Albatross [3] 是官方 PyTorch 推理实现，提供 CUDA WKV kernel 和 `forward_seq_batch` 批量并发推理。

### 2.2 嵌入评估基准

- **MTEB** [4]：大规模嵌入评估基准
- **STS-Benchmark** [5]：语义文本相似度，Spearman 相关系数
- **20 Newsgroups** [6]：20 类主题文档聚类
- **NLI / SICK-R** [7]：自然语言推理与句子对评分

### 2.3 各向异性与对比学习

- **AnglE** [8]：基于角度的对比损失，缓解嵌入各向异性
- **SimCSE** [9]：对比学习缓解各向异性，但需大量负样本
- **白化（Whitening）** [10]：线性变换消除各向异性，但破坏非线性结构

---

## 3. 方法

### 3.1 Albatross 特征提取

**模型**：RWKV-7-g1 0.4B（hidden=1024, 24 layers, 16 heads, head_size=64）

**特征**：
- **Hidden state** $h \in \mathbb{R}^{1024}$：最后一层 FFN 输出的 mean pooling
- **WKV state** $S \in \mathbb{R}^{16 \times 64 \times 64}$：第 12 层的递归记忆矩阵

**批量并发提取**：按 token 长度分桶，桶内序列长度相同无需 padding，用 albatross 的 `forward_seq_batch` 并发推理，速度 250 samples/s（vs 单序列 70 samples/s，3.6x 加速）。

**重要说明**：本工作完全使用 albatross 推理引擎，未修改任何源码。特征提取脚本仅复制 `forward_seq_batch` 的循环结构以同时提取中间层 state 和 hidden（原函数只返回 logits）。

### 3.2 统一投影框架

核心思想：用一个 2 隐藏层 MLP 将 hidden state 投影到 128 维语义空间。

$$\text{emb} = \text{normalize}(\text{MLP}(\text{BatchNorm}(h)))$$

**MLP 架构**（参数量 ~0.86M）：

```
Input(1024) → BatchNorm → Linear(1024→512) → GELU → LayerNorm → Dropout(0.2)
           → Linear(512→512) → GELU → LayerNorm → Dropout(0.2)
           → Linear(512→128) → L2 Normalize
```

**训练目标因任务而异**（关键设计）：

| 任务 | 训练数据 | Loss | 温度 τ |
|------|---------|------|--------|
| STS | 48.1k 标注对（连续分数 0-5） | AnglE（回归） | 0.50 |
| 聚类 | 38.1k 标注样本（pair score 0/1） | AnglE（对比） | 0.50 |

**为什么不能混用**：STS 学的是"相似度排序"（相对距离），聚类需要"类间分离"（绝对距离）。实验证明 STS 投影器迁移到聚类任务失败（v_measure 从 0.34 跌至 0.14，见 §4.4）。

### 3.3 任务一：语义相似度（STS）

**训练数据**（合计 48,147 对，比 STS-B train 提升 8x）：

| 数据集 | pairs | score 范围 | 来源 |
|--------|-------|-----------|------|
| STS-B train | 5,749 | 0-5 | GLUE |
| NLI train | 10,000 | 1-5 | SNLI/MultiNLI |
| extra_train | 22,471 | 0-5 | STS12-16 |
| SICK-R | 9,927 | 1-5 | SICK-R |

**AnglE Loss**（τ=0.50，albatross 路径最优）：

$$\mathcal{L} = -\frac{1}{n} \sum_i \left[ y_i \log \sigma\left(\frac{\cos(\theta_i)}{\tau}\right) + (1-y_i) \log \left(1 - \sigma\left(\frac{\cos(\theta_i)}{\tau}\right)\right) \right]$$

其中 $\cos(\theta_i) = \text{emb}_1 \cdot \text{emb}_2$（embedding 已 L2 normalized），$y_i$ 为 score 的 min-max 归一化：$y_i = (\text{score}_i - \min) / (\max - \min)$，将不同数据集的异构分数范围（0-5, 1-5）统一到 [0,1]。

**5 seed 集成**：训练 5 个不同种子（42, 123, 456, 789, 1024）的模型，embedding 平均后重新 L2 normalize。

### 3.4 任务二：主题聚类（监督投影 + KMeans）

**数据**：20 Newsgroups 全量 59,545 样本，按类别 stratified 划分为 train/dev/test（64/16/20：train 38,108 / dev 9,528 / test 11,909）。dev 用于模型选择（best_state），test 用于最终评估，避免数据泄露。

**监督对比学习**：
- Pair 构造：50% 同类（score=1），50% 不同类（score=0）
- 每 epoch 重新采样 20,000 pairs
- AnglE Loss，τ=0.50
- 5 seed 集成

**评估**：在 held-out test set 上用 projection + KMeans 评估 v_measure。**注意**：本方法使用了类别标签训练投影器，属于监督聚类，与 MTEB 基准的无监督聚类（如 bge-large 的 0.57）不直接可比。本工作的目标是验证 hidden state 的语义潜力，而非提出新的无监督聚类方法。

### 3.5 任务三：任务分类

**方法**：Hidden state + MLP 分类器（Top-K Head 筛选 + PCA 作为对比基线）。
- Hidden + MLP：1024 → 256 → 4，交叉熵损失
- Top-K Head（对比）：评估每个 head 的准确率，选 Top-8 head 的 state（32768 维）→ PCA 降到 256 维 → MLP

---

## 4. 实验

### 4.1 实验设置

- **Backbone**：RWKV-7-g1 0.4B（`rwkv7-g1d-0.4b-20260210-ctx8192.pth`，BlinkDL/rwkv7-g1）
- **推理引擎**：albatross（官方 PyTorch + CUDA WKV kernel）
- **训练硬件**：CPU（Intel i7-12700K）+ RTX 2080 Ti（仅 backbone 推理）
- **训练框架**：PyTorch 2.0+（CPU 模式训练 Projection）
- **State 层**：L12（实验确定为最优）
- **最大序列长度**：128 tokens（截断）

### 4.2 数据集

| 任务 | 数据集 | 样本数 | 用途 |
|------|--------|--------|------|
| 聚类 | 20 Newsgroups | 59,545 | train/dev/test split (64/16/20) |
| STS | STS-B train + NLI + extra + SICK-R | 48,147 pairs | 训练 |
| STS | STS-B dev/test | 1,500/1,379 | 评估 |
| 分类 | golden_balanced | 16,751 | 15% val split |

### 4.3 主要结果

#### 4.3.1 STS 任务

| 方法 | 训练数据 | Test Spearman |
|------|---------|--------------|
| 无监督 Hidden cosine | - | 0.4600 |
| MLP + STS-B train (5.7k) | 5,749 | 0.5818 |
| **MLP + 全部训练数据 (48.1k)** | **48,147** | **0.8504** |

| seed | dev | test | 集成 |
|------|-----|------|------|
| 42 | 0.8301 | 0.8281 | 0.8281 |
| 123 | 0.8314 | 0.8102 | 0.8418 |
| 456 | 0.8360 | 0.8144 | 0.8431 |
| 789 | 0.8385 | 0.8177 | 0.8463 |
| 1024 | 0.8296 | 0.8126 | **0.8504** |

**结论**：训练数据从 5.7k 扩展到 48.1k（8x），Spearman 从 0.58 提升到 0.85（+47%）。

**与监督嵌入模型的公平对比**：

| 方法 | 类型 | STS-B Spearman | 参数量 |
|------|------|----------------|--------|
| **本文 (RWKV-7 0.4B + 监督投影)** | **监督** | **0.8504** | **0.86M (投影器)** |
| bge-large-en-v1.5 [12] | 监督 | ~0.85 | 335M |
| all-MiniLM-L6-v2 [13] | 监督 | ~0.86 | 22M |
| 无监督 Hidden cosine | 无监督 | 0.46 | - |

*监督模型数据来源：MTEB Leaderboard [14]。各模型训练数据和评测 split 可能略有差异，此处为近似对比。*

本文方法在仅 0.86M 投影参数（0.4B 语言模型冻结，仅训练投影器）的条件下，STS-B Spearman 与监督训练的专用嵌入模型 bge-large-en-v1.5（335M 全参数微调）相当。需要强调的是，本方法需要额外的监督训练数据（48.1k pairs），且 0.4B 语言模型本身参数未计入对比。

#### 4.3.2 聚类任务（监督投影 + KMeans）

| 方法 | 训练方式 | Test v_measure |
|------|---------|----------------|
| Hidden + standardize + KMeans（baseline） | 无监督 | 0.2912 |
| STS projection 迁移（失败） | 监督（STS） | 0.1424 |
| **监督对比学习 projection** | **监督（聚类）** | **0.8466** |

| 方法 | v_measure | NMI | ARI |
|------|-----------|-----|-----|
| Baseline | 0.2912 | 0.29 | 0.10 |
| Projection + KMeans | 0.8388 | 0.84 | 0.61 |
| **Projection + standardize + KMeans** | **0.8466** | **0.85** | **0.62** |
| Projection + PCA(64) + KMeans | 0.8450 | 0.84 | 0.61 |

**注**：0.8466 为 5 seed 集成后 Projection + standardize + KMeans 的结果；seed 表中的 ens 列为直接 KMeans 结果（0.8388）。

| seed | dev_v | test_v | 集成 |
|------|-------|--------|------|
| 42 | 0.8371 | 0.8229 | 0.8229 |
| 123 | 0.8258 | 0.8232 | 0.8321 |
| 456 | 0.8213 | 0.8176 | 0.8333 |
| 789 | 0.8334 | 0.8268 | 0.8460 |
| 1024 | 0.8355 | 0.8281 | **0.8388** |

**结论**：监督对比学习从 epoch 1（0.57）快速提升到 epoch 30（0.83），5 seed 集成稳定在 0.84+。注意此结果使用了类别标签，与无监督 MTEB 基准不直接可比。dev_v 与 test_v 不同（如 seed 42: dev=0.8371, test=0.8229），说明 dev/test 严格分离，无数据泄露。

#### 4.3.3 无监督聚类方法对比

为验证"无监督方法无法提取 hidden 中的聚类结构"这一论点，在全量 test set（11909 样本）上系统对比了 7 类共 30+ 种无监督方法：

| 方法类别 | 最佳方法 | 最佳 v_measure |
|----------|----------|----------------|
| Hidden + KMeans（baseline） | L23 + standardize | 0.2912 |
| 非线性特征变换 | L23 + sign·sqrt + standardize | 0.3181 |
| 非线性降维（UMAP） | UMAP(32) + KMeans | 0.3263 |
| 多层拼接 | Last3 (L16+L20+L23) | 0.2553 |
| 层间差值 | L23−L0 + sign·sqrt | 0.3087 |
| Whitening / 去 top-k PC | PCA whitening 256 | 0.1629 |
| WKV state 聚合统计量 | row_mean + col_mean | 0.1020 |
| DeepCluster 自监督迭代 | Round 1 集成 | 0.3021 |

**分析**：

1. **线性方法上限 ~0.32**：KMeans、PCA、Whitening 等线性方法最高仅 0.32，远低于 MTEB 无监督 SOTA 0.57
2. **非线性降维无突破**：UMAP(32) 达 0.3263（略好于线性），但远未突破，说明问题不在降维方法
3. **多层拼接反而下降**：多层拼接（0.26）不如单层 L23（0.32），说明浅层噪声稀释了深层语义
4. **DeepCluster 伪标签退化**：初始 KMeans 伪标签仅 0.30，迭代后无法自我改善（0.30→0.30），证明伪标签质量不足以驱动 MLP 学习有效投影
5. **WKV state 无聚类信息**：state 的各种聚合统计量（row_sum、diag、trace 等）最高仅 0.10，说明 albatross 的 WKV state 数值范围过小（std=0.13），聚类信息主要存在于 hidden 而非 state

**根本原因**：0.4B 语言模型的优化目标是 next-token prediction，而非聚类。hidden state 中蕴含的语义信息需要非线性变换（监督投影 MLP）才能释放——这正是本文核心论点的实证支撑。无监督方法上限 0.33 vs 监督投影 0.85，提升 157%。

#### 4.3.4 分类任务

| 方法 | val_acc |
|------|---------|
| **Hidden + MLP** | **0.9392** |
| Top-8 head + PCA256 + MLP | 0.9250 |

**分析**：Hidden + MLP 优于 Top-K head + PCA（0.9392 > 0.9250），说明 albatross 的 hidden state 已具备良好的任务可分性，直接用 hidden 训练分类器即可；Top-K head 筛选+PCA 降维反而丢失部分信息。这一结果与 §5.1 的核心洞察一致——hidden 中蕴含语义信息，只需简单 MLP 即可提取（分类是判别任务，不需要投影到语义空间）。Top-K head 方法列于此作为对比基线。

### 4.4 消融实验

#### 4.4.1 STS Projection 迁移到聚类（失败）

此实验使用采样 2000 样本（每类 100）的子集，与 §4.3.2 的全量 test set（11909 样本）不同：

| 方法 | v_measure |
|------|-----------|
| Baseline (Hidden + standardize, 采样 2000) | 0.3426 |
| STS Projection + KMeans | 0.1424 |
| STS Projection + standardize + KMeans | 0.1305 |
| STS Projection + PCA(64) + KMeans | 0.1284 |
| Hidden + STS Projection 拼接 | 0.2756 |

**失败原因**：STS 学的是"相似度排序"（相对距离），L2 normalize 后 embedding std=0.0884 太小，类间距离被压缩。聚类需要"类间分离"（绝对距离），两者目标不同。

#### 4.4.2 训练数据规模影响（STS）

| 训练数据 | pairs | Test Spearman |
|---------|-------|--------------|
| STS-B train only | 5,749 | 0.5818 |
| + NLI + extra + SICK-R (全部) | 48,147 | **0.8504** |

**结论**：数据量是 STS 任务的关键瓶颈，8x 提升带来 +47% 性能提升。单一数据集的逐项增量实验未在本工作中执行，仅对比端到端结果。

### 4.5 失败方向总结

| 方法 | 结果 | 失败原因 |
|------|------|---------|
| STS projection 迁移聚类 | 0.14 | STS 学相似度排序，聚类需类间分离 |
| 无监督 KMeans | 0.29 | hidden 各向异性严重，线性方法无法提取 |
| 无监督 Hidden cosine (STS) | 0.46 | 各向异性导致 0.46 << 0.85 |
| UMAP 非线性降维 | 0.33 | 非线性降维也无效，问题不在降维方法 |
| DeepCluster 自监督迭代 | 0.30 | 伪标签质量太低，无法自我改善 |
| 多层 hidden 拼接 | 0.26 | 浅层噪声稀释深层语义 |
| Whitening / 去 top-k PC | 0.16 | 各向异性修复后反而下降 |
| 纯 WKV state (albatross, Q-Readout) | 0.11 | state 数值范围小，std=0.13 |
| WKV state 聚合统计量 | 0.10 | row_sum/diag/trace 等无聚类信息 |
| 禁用 v_first 机制 | 0.16 | state std 仅 0.16，未达 Rust 水平 |
| fp32 state (vs fp16) | 0.13 | std 0.13≈0.13，精度非根因 |

---

## 5. 讨论

### 5.1 核心洞察：监督投影释放 hidden 语义潜力

albatross hidden state 的各向异性问题（无监督 STS 仅 0.46）并非特征缺陷，而是**无监督方法无法提取非线性结构**的证据。监督 MLP 分类达 0.94 证明特征中有信息，只需用监督投影器（MLP + AnglE Loss）将其映射到线性可分的语义空间。

### 5.2 任务专用投影器不可混用

| 任务 | 目标 | 投影器行为 |
|------|------|-----------|
| STS | 相似度排序 | 保留相对距离，L2 normalize 压缩类间 |
| 聚类 | 类间分离 | 放大类间距离，保留绝对位置 |

STS 投影器迁移到聚类失败（0.14 < 0.34 baseline），证明**投影器必须与任务目标对齐**。

### 5.3 数据规模是关键

STS 任务从 5.7k 扩展到 48.1k（8x）带来 +47% 提升，说明 albatross hidden 的语义信息需要足够数据才能通过监督学习提取。这与 SimCSE [9] 的发现一致：对比学习需要大量样本。

### 5.4 Albatross vs Rust（web-rwkv）

albatross 路径最优 τ=0.50 远高于 Rust 路径的 0.1，根因是 hidden 数值范围差异（albatross std=1.70 vs Rust std=3.54）。本工作放弃与 Rust 对比，以 albatross 推理引擎为标准，避免修改推理实现。

---

## 6. 结论

本文提出基于监督投影的 RWKV-7 hidden state 语义嵌入提取框架，在三个标准任务上取得与监督嵌入模型可比的结果：

- **语义相似度（监督）**：Spearman=0.8504，与监督训练的 bge-large-en-v1.5（~0.85, 335M）和 all-MiniLM-L6-v2（~0.86, 22M）相当，投影器仅 0.86M 参数
- **主题聚类（监督）**：v_measure=0.8466，远超无监督 baseline（0.29）及 30+ 种无监督方法（最高 0.33）
- **任务分类**：val_acc=0.9392

核心洞察：albatross hidden state 蕴含语义信息但需监督投影释放；任务专用投影器不可混用；数据规模是关键瓶颈。WKV state 的聚类效果远低于 hidden state（0.11 vs 0.29），说明 hidden state 更适合语义提取。所有方法基于 0.4B 模型，albatross 推理引擎（无修改源码），CPU 可训，参数量 ~0.86M，适合边缘部署。

---

## 参考文献

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

## 附录 A：复现指南

### 环境要求

- Python 3.10+, uv（包管理）
- CUDA GPU（albatross 推理用）
- Windows 需 MSVC（编译 CUDA 扩展）

### 复现命令

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# 1. 下载模型和 albatross 源码
uv run --project ../../scripts python 00_setup.py

# 2. 提取特征（albatross 并发推理）
.\run_with_msvc.bat extract_features.py --task sts --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task sts_extra --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task cluster_full --batch-size 16 --max-length 128
.\run_with_msvc.bat extract_features.py --task classification --batch-size 16 --max-length 128 --limit 8000

# 3. 运行三任务
uv run --project ../../scripts python 02_sts_similarity.py          # STS: 0.8504
uv run --project ../../scripts python 05_cluster_supervised_projection.py  # 聚类: 0.8466
uv run --project ../../scripts python 03_classification.py          # 分类: 0.9392
```

### 预期输出

```
STS:    5seed 集成:  Spearman = 0.8504
聚类:    Projection + standardize + KMeans: v_measure = 0.8466
分类:    val_acc = 0.9392
```
