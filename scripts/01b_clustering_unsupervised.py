#!/usr/bin/env python3
"""聚类任务 - 严格无监督评估（不使用任何标签训练 projector）

方法：MTEB 官方 twentynewsgroups-clustering 数据集（单 test split）
  1. Hidden state (mean pooling, L23 最后一层)
  2. standardize
  3. KMeans 聚类成 20 簇
  4. 用 10 个不同 random_state 的 KMeans 求均值+std（避免单次随机性）
  5. 评估 v_measure / NMI / ARI

严格不使用任何 20NG 标签训练 projector——这是纯无监督评估。

特征提取:
    run_with_msvc.bat extract_features.py --task cluster_full
    生成 cache_python/cluster_full_l12.npz

运行:
    cd paper/scripts
    uv run --project ../../scripts python 01b_clustering_unsupervised.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from cache import load_npz  # noqa: E402

PAPER_DIR = SCRIPT_DIR.parent
DATA_DIR = PAPER_DIR.parent / "data"
CACHE_DIR = PAPER_DIR / "cache_python"
LAYER = 12
N_CLUSTERS = 20
SEEDS = [42, 123, 456, 789, 1024, 2026, 314, 271, 161, 828]  # 10 个种子


def standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def kmeans_eval_multi_seed(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = N_CLUSTERS, seeds=None):
    """用多个 random_state 的 KMeans 求均值+std"""
    if seeds is None:
        seeds = SEEDS
    v_scores, nmi_scores, ari_scores = [], [], []
    for i, seed in enumerate(seeds):
        pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(embeddings)
        v_scores.append(v_measure_score(labels, pred))
        nmi_scores.append(normalized_mutual_info_score(labels, pred))
        ari_scores.append(adjusted_rand_score(labels, pred))
        print(f"  seed={seed:4d}: v={v_scores[-1]:.4f} nmi={nmi_scores[-1]:.4f} ari={ari_scores[-1]:.4f}", flush=True)
    return {
        "v_mean": np.mean(v_scores), "v_std": np.std(v_scores),
        "nmi_mean": np.mean(nmi_scores), "nmi_std": np.std(nmi_scores),
        "ari_mean": np.mean(ari_scores), "ari_std": np.std(ari_scores),
        "v_scores": v_scores, "nmi_scores": nmi_scores, "ari_scores": ari_scores,
    }


def main():
    print("=" * 60, flush=True)
    print("聚类任务 - 严格无监督评估 (不使用标签训练 projector)", flush=True)
    print("数据: MTEB twentynewsgroups-clustering (官方 test split)", flush=True)
    print("方法: Hidden (L23) + standardize + KMeans × 10 seeds", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载全量数据 (MTEB 官方 test split)
    cache_path = CACHE_DIR / f"cluster_full_l{LAYER}.npz"
    print(f"\n加载特征: {cache_path}", flush=True)
    with np.load(cache_path) as data:
        hiddens = data["hiddens"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.int32, copy=False) if "labels" in data else None

    # 如果缓存里没有 labels，从原始数据读取
    if labels is None:
        import json
        data_path = DATA_DIR / "clustering" / "twentynewsgroups.jsonl"
        records = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
        labels = np.array([r["label"] for r in records], dtype=np.int32)

    labels = np.asarray(labels).astype(np.int32)
    n_samples = hiddens.shape[0]
    if len(labels) != n_samples:
        n = min(n_samples, len(labels))
        hiddens = hiddens[:n]
        labels = labels[:n]
    print(f"  样本数: {n_samples}", flush=True)
    print(f"  hiddens: {hiddens.shape}, labels: {len(labels)}", flush=True)
    print(f"  类别数: {len(set(labels))}", flush=True)

    # 2. Baseline: Hidden (raw) + KMeans
    print(f"\n-- Baseline: Hidden (raw) + KMeans × 10 seeds --", flush=True)
    res_raw = kmeans_eval_multi_seed(hiddens, labels)
    print(f"  v_measure = {res_raw['v_mean']:.4f} ± {res_raw['v_std']:.4f}", flush=True)

    # 3. Hidden + standardize + KMeans
    print(f"\n-- Hidden + standardize + KMeans × 10 seeds --", flush=True)
    h_std = standardize(hiddens)
    res_std = kmeans_eval_multi_seed(h_std, labels)
    print(f"  v_measure = {res_std['v_mean']:.4f} ± {res_std['v_std']:.4f}", flush=True)

    # 4. Hidden + standardize + sign·sqrt 非线性变换
    print(f"\n-- Hidden + standardize + sign·sqrt + KMeans × 10 seeds --", flush=True)
    h_nonlinear = np.sign(h_std) * np.sqrt(np.abs(h_std))
    h_nonlinear = standardize(h_nonlinear)
    res_nonlinear = kmeans_eval_multi_seed(h_nonlinear, labels)
    print(f"  v_measure = {res_nonlinear['v_mean']:.4f} ± {res_nonlinear['v_std']:.4f}", flush=True)

    # 5. 结论
    print(f"\n{'=' * 60}", flush=True)
    print(f"结论 (严格无监督, 不使用标签训练 projector):", flush=True)
    print(f"  Hidden (raw) + KMeans:                  v = {res_raw['v_mean']:.4f} ± {res_raw['v_std']:.4f}", flush=True)
    print(f"  Hidden + standardize + KMeans:           v = {res_std['v_mean']:.4f} ± {res_std['v_std']:.4f}", flush=True)
    print(f"  Hidden + standardize + sign·sqrt + KMeans: v = {res_nonlinear['v_mean']:.4f} ± {res_nonlinear['v_std']:.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
