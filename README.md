# RWKV-7 State 语义嵌入：基于监督投影的三任务统一框架

本仓库系统性探索如何从 RWKV-7 的 hidden state 提取语义嵌入，基于 **albatross 官方推理引擎**（无修改），覆盖三个标准评估任务。

**论文**：[paper.md](paper.md)（中文）| [paper_en.md](paper_en.md)（English）

## 核心结果

| 任务 | 方法 | 指标 | 对比 |
|------|------|------|------|
| **语义相似度** | 监督投影 (48.1k pairs) + AnglE + 5seed | Spearman=**0.8504** | 接近 bge-large 0.83-0.85 |
| **主题聚类（监督）** | 监督对比学习投影 + KMeans | v_measure=**0.8466** | 无监督 baseline 0.29，无监督 SOTA 0.57 |
| **任务分类** | Hidden + MLP | val_acc=**0.9392** | - |

## 核心洞察

**albatross hidden state 蕴含语义信息（监督分类达 0.94），但无监督方法无法提取（STS 0.46、聚类 0.29），需要监督投影器释放其潜力。**

| 任务 | 无监督 | 监督投影 | 提升 |
|------|--------|---------|------|
| STS | 0.46 | 0.85 | +85% |
| 聚类 | 0.29 | 0.85 | +193% |

**任务专用投影器不可混用**：STS 学相似度排序，聚类学类间分离，两者目标不同（STS 投影迁移到聚类失败：0.14 < 0.34 baseline）。

## 目录结构

```
paper/
├── paper.md                          # 论文（中文）
├── paper_en.md                       # Paper (English)
├── README.md                         # 本文件
├── albatross_src/                    # albatross 官方源码（多个版本）
│   └── faster_251101/reference/      # 参考实现（rwkv7.py + cuda/）
├── models/                           # RWKV-7 0.4B 模型
│   └── rwkv7-g1d-0.4b-20260210-ctx8192.pth
├── cache_python/                     # 特征缓存（.npz）
└── scripts/
    ├── 00_setup.py                   # 环境配置（下载模型+复制源码）
    ├── extract_features.py           # 批量并发特征提取（albatross）
    ├── 01_clustering.py              # 任务一：聚类（无监督 baseline）
    ├── 02_sts_similarity.py          # 任务二：STS（48.1k 训练，监督投影）
    ├── 03_classification.py          # 任务三：分类（Hidden+MLP）
    ├── 04_cluster_with_projection.py # STS 投影迁移聚类（失败实验）
    ├── 05_cluster_supervised_projection.py  # 聚类（监督对比学习投影）
    ├── run_with_msvc.bat             # Windows MSVC 环境激活
    └── lib/
        ├── albatross_wrapper.py      # albatross 封装（含 batch 并发提取）
        ├── cache.py                   # .npz 缓存读写
        ├── rwkv7.py                   # albatross 官方代码
        └── cuda/                      # WKV CUDA kernel
```

## 环境要求

### 软件
- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（包管理）
- CUDA GPU（albatross 推理用）
- Windows 需 MSVC（编译 CUDA 扩展）

### Python 依赖

```bash
# 用 uv 安装（推荐）
uv pip install numpy torch scikit-learn scipy huggingface_hub flag_gems

# 或用 pip
pip install numpy torch scikit-learn scipy huggingface_hub flag_gems
```

## 运行步骤

### 步骤 0：环境配置

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# 下载 RWKV-7 0.4B 模型（BlinkDL/rwkv7-g1）+ 复制 albatross 源码
uv run --project ../../scripts python 00_setup.py
```

### 步骤 1：下载数据集

```powershell
cd c:\work\niceui\rwkv-router

# 下载 STS + 聚类 + 分类数据集
uv run --project scripts python scripts/download_embedding_eval_data.py
```

数据集将下载到：
- `data/clustering/twentynewsgroups.jsonl`（聚类，59545 样本）
- `data/sts/sts_train.jsonl` 等（STS-B）
- `data/sts/nli_train.jsonl`, `extra_train.jsonl`, `sickr.jsonl`（额外训练数据）
- `data/golden_balanced.jsonl`（分类）

### 步骤 2：提取特征（albatross 并发推理）

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# STS 特征（STS-B train/dev/test）
.\run_with_msvc.bat extract_features.py --task sts --batch-size 16 --max-length 128

# STS 额外训练数据（NLI + extra_train + SICK-R，共 42k pairs）
.\run_with_msvc.bat extract_features.py --task sts_extra --batch-size 16 --max-length 128

# 聚类全量特征（59545 样本）
.\run_with_msvc.bat extract_features.py --task cluster_full --batch-size 16 --max-length 128

# 分类特征（限制 8000 样本以加速）
.\run_with_msvc.bat extract_features.py --task classification --batch-size 16 --max-length 128 --limit 8000
```

**特征提取速度**：250 samples/s（按长度分桶并发，3.6x 加速）

### 步骤 3：运行三任务

```powershell
cd c:\work\niceui\rwkv-router\paper\scripts

# 任务二：语义相似度（48.1k 训练 + 5seed 集成）→ Spearman 0.8504
uv run --project ../../scripts python 02_sts_similarity.py

# 任务五：主题聚类（监督对比学习 + 5seed 集成）→ v_measure 0.8466
uv run --project ../../scripts python 05_cluster_supervised_projection.py

# 任务三：任务分类（Hidden + MLP）→ val_acc 0.9392
uv run --project ../../scripts python 03_classification.py
```

### 步骤 4（可选）：消融实验

```powershell
# 无监督聚类 baseline（Hidden + standardize + KMeans）→ v_measure 0.34
uv run --project ../../scripts python 01_clustering.py

# STS 投影迁移聚类（失败实验）→ v_measure 0.14
uv run --project ../../scripts python 04_cluster_with_projection.py
```

## 预期输出

### 任务二：语义相似度
```
结论:
  无监督 Hidden cosine:  Spearman = 0.4600
  单模型均值:            Spearman = 0.8166
  5seed 集成:             Spearman = 0.8504
```

### 任务五：主题聚类
```
结论:
  Baseline (Hidden + standardize):    v_measure = 0.2912
  Projection + KMeans (直接):         v_measure = 0.8388
  Projection + standardize + KMeans:  v_measure = 0.8466
  Projection + PCA(64) + KMeans:      v_measure = 0.8450
```

### 任务三：任务分类
```
结果:
  val_acc = 0.9392
```

## 关键设计决策

### 1. 为什么用监督投影（而非无监督）
albatross hidden state 存在严重各向异性（无监督 STS 仅 0.46），但监督 MLP 分类达 0.94，证明特征中有信息。监督投影器（MLP + AnglE Loss）能将 hidden 映射到线性可分的语义空间。

### 2. 为什么 STS 和聚类需要不同的投影器
- **STS**：学相似度排序（相对距离），L2 normalize 压缩类间距离
- **聚类**：学类间分离（绝对距离），保留绝对位置

STS 投影迁移到聚类失败（0.14 < 0.34 baseline），证明投影器必须与任务目标对齐。

### 3. 为什么数据规模是关键
STS 从 5.7k 扩展到 48.1k（8x）带来 +47% 提升，说明 albatross hidden 的语义信息需要足够数据才能通过监督学习提取。

### 4. 为什么用 albatross（而非 Rust web-rwkv）
- albatross 是官方 PyTorch 实现，支持批量并发推理
- 本工作完全使用官方推理引擎，未修改任何源码
- albatross 路径最优 τ=0.50（Rust 路径为 0.1），根因是 hidden 数值范围差异

## 失败方向总结

| 方法 | 失败原因 |
|------|---------|
| STS projection 迁移聚类 | STS 学相似度排序，聚类需类间分离 |
| 无监督 KMeans | hidden 各向异性严重，无监督无法提取 |
| 无监督 Hidden cosine (STS) | 各向异性导致 0.46 << 0.85 |
| 纯 WKV state (albatross, Q-Readout) | state 数值范围小，std=0.13，Q-Readout 聚类 v_measure=0.11 |
| 禁用 v_first 机制 | state std 仅 0.16，未达 Rust 水平 |
| fp32 state (vs fp16) | std 0.13≈0.13，精度非根因 |

## RWKV-7 模型规格

| 模型 | hidden_dim | num_heads | head_size | state_dim/层 |
|------|-----------|-----------|-----------|--------------|
| 0.4B | 1024 | 16 | 64 | 65536 |
| 1.5B | 2048 | 32 | 64 | 131072 |
| 2.9B | 2560 | 40 | 64 | 163840 |
| 7B   | 4096 | 64 | 64 | 262144 |

## 引用

```bibtex
@misc{rwkv-state-embedding-2026,
  title={RWKV-7 State Semantic Embedding: A Unified Supervised Projection Framework},
  author={RWKV Community},
  year={2026},
  url={https://github.com/opensquilla/rwkv-router}
}
```

## 许可证

MIT
