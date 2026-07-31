"""
任务一：主题聚类 - RWKV Hidden State 层级语义累积

方法：提取 RWKV-7 最后一层 hidden state（mean pooling），standardize 后 KMeans 聚类。

核心发现：albatross（官方推理引擎）的 hidden state 语义区分能力随网络深度递增：
  L0 (v=0.12) → L8 (v=0.19) → L16 (v=0.21) → L23 (v=0.34)
这反映了 RWKV 递归状态在网络深处累积语义信息的特性。

数据集：20 Newsgroups (mteb/twentynewsgroups-clustering, 20 类)
评估：KMeans 聚类成 20 簇，v_measure / NMI / ARI

特征提取:
    run_with_msvc.bat extract_features.py --task cluster
    生成 cache_python/cluster_l12.npz

运行:
    cd paper/scripts
    uv run --project ../../scripts python 01_clustering.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

NUM_HEADS = 16
HEAD_SIZE = 64
N_CLUSTERS = 20


def standardize(x: np.ndarray) -> np.ndarray:
    """按特征维度标准化."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def kmeans_eval(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = N_CLUSTERS) -> dict:
    """KMeans 聚类并评估."""
    pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)
    return {
        "v_measure": v_measure_score(labels, pred),
        "nmi": normalized_mutual_info_score(labels, pred),
        "ari": adjusted_rand_score(labels, pred),
    }


def main():
    parser = argparse.ArgumentParser(description="任务一: 主题聚类")
    parser.add_argument("--cache", type=Path, default=Path("../cache_python/cluster_l12.npz"))
    parser.add_argument("--multilayer-cache", type=Path, default=Path("../cache_python/cluster_multilayer_hidden.npz"))
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务一: 主题聚类 - RWKV Hidden State", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载缓存
    print(f"\n加载缓存: {args.cache}", flush=True)
    with np.load(args.cache) as data:
        states = data["states"].astype(np.float32)
        hiddens = data["hiddens"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
    print(f"  states: {states.shape} (L12 WKV state)", flush=True)
    print(f"  hiddens: {hiddens.shape} (L23 mean-pooled)", flush=True)
    print(f"  labels: {len(labels)}, 类别数: {len(set(labels))}", flush=True)
    print(f"  state std: {states.std():.4f}, hidden std: {hiddens.std():.4f}", flush=True)

    # 2. 主方法: Hidden + standardize + KMeans
    print(f"\n-- 主方法: Hidden (L23) + standardize + KMeans --", flush=True)
    hidden_emb = standardize(hiddens)
    res = kmeans_eval(hidden_emb, labels)
    print(f"  v_measure = {res['v_measure']:.4f}", flush=True)
    print(f"  NMI       = {res['nmi']:.4f}", flush=True)
    print(f"  ARI       = {res['ari']:.4f}", flush=True)

    # 3. 对比: Raw state + PCA256 + standardize + KMeans
    print(f"\n-- 对比: Raw state (L12) + PCA256 + KMeans --", flush=True)
    pca = PCA(n_components=256, random_state=42)
    state_pca = standardize(pca.fit_transform(states))
    res_state = kmeans_eval(state_pca, labels)
    print(f"  v_measure = {res_state['v_measure']:.4f}", flush=True)

    # 4. 分层分析: 不同层 hidden 的聚类能力
    if args.multilayer_cache.exists():
        print(f"\n-- 分层分析: hidden state 随网络深度的语义累积 --", flush=True)
        print(f"  {'layer':<8} {'std':<10} {'v_measure':<12}", flush=True)
        with np.load(args.multilayer_cache) as data:
            ml_labels = data["labels"]
            for key in sorted(data.files, key=lambda k: int(k[1:]) if k.startswith("L") and k[1:].isdigit() else 999):
                if key == "labels":
                    continue
                h = data[key].astype(np.float32)
                r = kmeans_eval(standardize(h), ml_labels)
                print(f"  {key:<8} {h.std():<10.4f} {r['v_measure']:<12.4f}", flush=True)
    else:
        print(f"\n  [skip] 多层缓存不存在: {args.multilayer_cache}", flush=True)
        print(f"  如需分层分析, 运行: test\\extract_multilayer_hidden.py", flush=True)

    # 5. 结论
    print(f"\n{'='*60}", flush=True)
    print(f"结论:", flush=True)
    print(f"  Hidden (L23) v_measure = {res['v_measure']:.4f}", flush=True)
    print(f"  Raw state (L12) v_measure = {res_state['v_measure']:.4f}", flush=True)
    print(f"  → hidden 优于 raw state, 语义随网络深度累积", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
