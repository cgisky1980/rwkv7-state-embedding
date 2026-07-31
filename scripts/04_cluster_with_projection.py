"""
任务四：聚类 - 用 Universal Projection 提升聚类效果

核心思路:
  STS 训练的 universal projection (47.6k pairs, AnglE Loss) 学到了通用语义 embedding,
  能缓解 hidden state 的各向异性问题. 将该 projection 迁移到聚类任务:
    hidden (1024维) → projection → 128维 embedding → KMeans

  这是一种迁移学习: STS 的监督信号 (语义相似度) 与聚类目标 (语义分组) 一致,
  因此 projection 能有效提升聚类的线性可分性.

依赖:
  - universal_projection_l12.pt (由 02_sts_similarity.py 训练并保存)
  - cluster_l12.npz (由 extract_features.py --task cluster 生成)

运行:
  cd paper/scripts
  uv run --project ../../scripts python 04_cluster_with_projection.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from cache import load_npz  # noqa: E402

LAYER = 12
NUM_HEADS = 16
HEAD_SIZE = 64
N_CLUSTERS = 20


# ============================================================
# MlpProj (必须与 02_sts_similarity.py 一致)
# ============================================================
class MlpProj(nn.Module):
    """2层 MLP 投影器: 与 02_sts_similarity.py 一致."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return F.normalize(self.net(x), p=2, dim=-1)


def standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def kmeans_eval(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = N_CLUSTERS) -> dict:
    pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)
    return {
        "v_measure": v_measure_score(labels, pred),
        "nmi": normalized_mutual_info_score(labels, pred),
        "ari": adjusted_rand_score(labels, pred),
    }


def main():
    parser = argparse.ArgumentParser(description="任务四: 聚类 + Universal Projection")
    parser.add_argument("--cache-dir", type=Path, default=Path("../cache_python"))
    parser.add_argument("--proj-path", type=Path, default=Path(f"../cache_python/universal_projection_l{LAYER}.pt"))
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务四: 聚类 + Universal Projection (迁移自 STS)", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载聚类数据
    cluster_cache = args.cache_dir / f"cluster_l{LAYER}.npz"
    print(f"\n加载聚类数据: {cluster_cache}", flush=True)
    states, hiddens = load_npz(cluster_cache)
    with np.load(cluster_cache) as data:
        labels = data["labels"].astype(np.int32)
    print(f"  hiddens: {hiddens.shape}, labels: {len(labels)} classes: {len(set(labels))}", flush=True)
    print(f"  hidden std: {hiddens.std():.4f}", flush=True)

    # 2. Baseline: Hidden + standardize + KMeans
    print(f"\n-- Baseline: Hidden (L23) + standardize + KMeans --", flush=True)
    res_base = kmeans_eval(standardize(hiddens), labels)
    print(f"  v_measure = {res_base['v_measure']:.4f}", flush=True)
    print(f"  NMI       = {res_base['nmi']:.4f}", flush=True)
    print(f"  ARI       = {res_base['ari']:.4f}", flush=True)

    # 3. 加载 universal projection
    print(f"\n加载 universal projection: {args.proj_path}", flush=True)
    proj_data = torch.load(args.proj_path, map_location="cpu", weights_only=False)
    config = proj_data["config"]
    state_dicts = proj_data["state_dicts"]
    seeds = proj_data["seeds"]
    print(f"  seeds: {seeds}", flush=True)
    print(f"  config: {config}", flush=True)

    # 4. 用 projection 处理 hidden → 128维 embedding
    # 5 seed 集成: 每个 seed 的 projection 输出平均
    print(f"\n-- 用 projection 处理 hidden → 128维 embedding (5 seed 集成) --", flush=True)
    hiddens_t = torch.from_numpy(hiddens).float()

    all_emb = []
    for i, state_dict in enumerate(state_dicts):
        model = MlpProj(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
        )
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            emb = model(hiddens_t).numpy()  # (N, 128) L2 normalized
        all_emb.append(emb)
        print(f"  seed={seeds[i]}: emb shape={emb.shape}, std={emb.std():.4f}", flush=True)

    # 集成: 平均后重新 L2 normalize
    emb_avg = np.mean(all_emb, axis=0)
    emb_avg = emb_avg / (np.linalg.norm(emb_avg, axis=1, keepdims=True) + 1e-12)
    print(f"  集成后: emb shape={emb_avg.shape}, std={emb_avg.std():.4f}", flush=True)

    # 5. 评估: Projection embedding + KMeans
    print(f"\n-- 方法 A: Projection embedding + KMeans (直接) --", flush=True)
    res_proj = kmeans_eval(emb_avg, labels)
    print(f"  v_measure = {res_proj['v_measure']:.4f}", flush=True)
    print(f"  NMI       = {res_proj['nmi']:.4f}", flush=True)
    print(f"  ARI       = {res_proj['ari']:.4f}", flush=True)

    # 6. 评估: Projection embedding + standardize + KMeans
    print(f"\n-- 方法 B: Projection embedding + standardize + KMeans --", flush=True)
    res_proj_std = kmeans_eval(standardize(emb_avg), labels)
    print(f"  v_measure = {res_proj_std['v_measure']:.4f}", flush=True)
    print(f"  NMI       = {res_proj_std['nmi']:.4f}", flush=True)
    print(f"  ARI       = {res_proj_std['ari']:.4f}", flush=True)

    # 7. 评估: Projection embedding + PCA + KMeans
    print(f"\n-- 方法 C: Projection embedding + PCA(64) + KMeans --", flush=True)
    pca = PCA(n_components=64, random_state=42)
    emb_pca = pca.fit_transform(emb_avg)
    res_proj_pca = kmeans_eval(emb_pca, labels)
    print(f"  v_measure = {res_proj_pca['v_measure']:.4f}", flush=True)
    print(f"  NMI       = {res_proj_pca['nmi']:.4f}", flush=True)
    print(f"  ARI       = {res_proj_pca['ari']:.4f}", flush=True)

    # 8. 评估: Hidden + Projection 拼接
    print(f"\n-- 方法 D: Hidden + Projection embedding 拼接 + standardize + KMeans --", flush=True)
    concat = np.concatenate([standardize(hiddens), standardize(emb_avg)], axis=1)
    res_concat = kmeans_eval(concat, labels)
    print(f"  v_measure = {res_concat['v_measure']:.4f}", flush=True)
    print(f"  NMI       = {res_concat['nmi']:.4f}", flush=True)
    print(f"  ARI       = {res_concat['ari']:.4f}", flush=True)

    # 9. 对比单个 seed 的效果
    print(f"\n-- 单 seed 对比 --", flush=True)
    for i, emb in enumerate(all_emb):
        r = kmeans_eval(emb, labels)
        print(f"  seed={seeds[i]}: v_measure = {r['v_measure']:.4f}", flush=True)

    # 10. 结论
    print(f"\n{'='*60}", flush=True)
    print(f"结论:", flush=True)
    print(f"  Baseline (Hidden + standardize):    v_measure = {res_base['v_measure']:.4f}", flush=True)
    print(f"  Projection + KMeans (直接):         v_measure = {res_proj['v_measure']:.4f}", flush=True)
    print(f"  Projection + standardize + KMeans:  v_measure = {res_proj_std['v_measure']:.4f}", flush=True)
    print(f"  Projection + PCA(64) + KMeans:      v_measure = {res_proj_pca['v_measure']:.4f}", flush=True)
    print(f"  Hidden + Projection 拼接:           v_measure = {res_concat['v_measure']:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
