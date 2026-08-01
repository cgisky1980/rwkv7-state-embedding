#!/usr/bin/env python3
"""任务六: 聚类 - 20NG sklearn 全文版严格去重 + 监督投影迁移

数据源: sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
  - 全文内容 (vs MTEB 版标题级短文本, mean 30 chars)
  - 严格去重 (文本归一化哈希, 0.43% 跨组交叉post)
  - 分层 70/15/15 split: train 12777 / dev 2738 / test 2738

实验范式 (科学严谨):
  1. train (12777, 看标签) 用监督对比学习训练 projector (AnglE loss)
  2. dev (2738) 用于 early stopping + best_state 选择
  3. test (2738, held-out, 不参与训练) 用 projection + KMeans 评估 v_measure
  4. 5 seed 集成
  5. 与无监督 baseline (Hidden + standardize + KMeans) 对比

对比基准:
  - 无监督 baseline: Hidden + standardize + KMeans (test set)
  - 监督投影: train 训练 projector → test 评估 (held-out)
  - 注意: 这是监督方法 (用 20NG train 标签训练), 不可与无监督 MTEB 直接对比

依赖:
  - cache_python/cluster_20ng_full_l12_{train,dev,test}.npz
    (由 extract_features.py --task cluster_20ng_full 生成)

运行:
  cd paper/scripts
  uv run --project ../../scripts python 06_cluster_20ng_full.py
"""

import argparse
import sys
import time
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

LAYER = 12
N_CLUSTERS = 20


# ============================================================
# MlpProj (与 05 脚本一致)
# ============================================================
class MlpProj(nn.Module):
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


def angle_loss(emb1, emb2, scores, temperature=0.5):
    cos_sim = (emb1 * emb2).sum(dim=-1)
    cos_sim_scaled = cos_sim / temperature
    loss = -torch.mean(
        scores * torch.log(torch.sigmoid(cos_sim_scaled) + 1e-8)
        + (1 - scores) * torch.log(1 - torch.sigmoid(cos_sim_scaled) + 1e-8)
    )
    return loss


def standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def kmeans_eval(embeddings, labels, n_clusters=N_CLUSTERS, seed=42):
    pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(embeddings)
    return {
        "v_measure": v_measure_score(labels, pred),
        "nmi": normalized_mutual_info_score(labels, pred),
        "ari": adjusted_rand_score(labels, pred),
    }


def kmeans_eval_multi_seed(embeddings, labels, n_clusters=N_CLUSTERS, seeds=None):
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    v, nmi, ari = [], [], []
    for s in seeds:
        r = kmeans_eval(embeddings, labels, n_clusters, seed=s)
        v.append(r["v_measure"])
        nmi.append(r["nmi"])
        ari.append(r["ari"])
    return {
        "v_mean": np.mean(v), "v_std": np.std(v),
        "nmi_mean": np.mean(nmi), "nmi_std": np.std(nmi),
        "ari_mean": np.mean(ari), "ari_std": np.std(ari),
    }


def make_supervised_pairs(hiddens, labels, n_pairs=None, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n_pairs is None:
        n_pairs = n * 2
    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos
    classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in classes}

    s1, s2, sc = [], [], []
    for _ in range(n_pos):
        c = rng.choice(classes)
        if len(class_indices[c]) < 2:
            continue
        i, j = rng.choice(class_indices[c], 2, replace=False)
        s1.append(hiddens[i]); s2.append(hiddens[j]); sc.append(1.0)
    for _ in range(n_neg):
        c1, c2 = rng.choice(classes, 2, replace=False)
        i = rng.choice(class_indices[c1])
        j = rng.choice(class_indices[c2])
        s1.append(hiddens[i]); s2.append(hiddens[j]); sc.append(0.0)
    return np.array(s1, dtype=np.float32), np.array(s2, dtype=np.float32), np.array(sc, dtype=np.float32)


def train_one(seed, train_h, train_y, dev_h, dev_y, temperature=0.5, n_epochs=30, device="cpu", n_pairs_per_epoch=20000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MlpProj(input_dim=train_h.shape[1], hidden_dim=512, output_dim=128, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_v = -1.0
    best_state = None
    batch_size = 256

    for epoch in range(n_epochs):
        s1, s2, sc = make_supervised_pairs(train_h, train_y, n_pairs=n_pairs_per_epoch, seed=seed * 1000 + epoch)
        s1_t = torch.from_numpy(s1).float().to(device)
        s2_t = torch.from_numpy(s2).float().to(device)
        sc_t = torch.from_numpy(sc).float().to(device)

        model.train()
        perm = torch.randperm(len(sc_t))
        for i in range(0, len(sc_t), batch_size):
            idx = perm[i:i + batch_size]
            emb1 = model(s1_t[idx]); emb2 = model(s2_t[idx])
            loss = angle_loss(emb1, emb2, sc_t[idx], temperature)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            dev_t = torch.from_numpy(dev_h).float().to(device)
            dev_emb = model(dev_t).cpu().numpy()
        dev_v = kmeans_eval(dev_emb, dev_y)["v_measure"]
        if dev_v > best_v:
            best_v = dev_v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch={epoch+1}/{n_epochs} loss={loss.item():.4f} dev_v={dev_v:.4f} best={best_v:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model, best_v


def main():
    parser = argparse.ArgumentParser(description="任务六: 20NG sklearn 全文版监督投影迁移")
    parser.add_argument("--cache-dir", type=Path, default=Path("../cache_python"))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务六: 20NG (sklearn 全文, 严格去重 split) 监督投影迁移", flush=True)
    print("数据源: sklearn.fetch_20newsgroups(remove=('headers','footers','quotes'))", flush=True)
    print("去重: 文本归一化哈希 (跨组交叉post)", flush=True)
    print("split: 70/15/15 stratified (train看标签 / dev选best_state / test held-out)", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载三个 split
    splits = {}
    for split in ["train", "dev", "test"]:
        cache_path = args.cache_dir / f"cluster_20ng_full_l{LAYER}_{split}.npz"
        print(f"\n加载 {split}: {cache_path}", flush=True)
        with np.load(cache_path) as data:
            hiddens = data["hiddens"].astype(np.float32, copy=False)
            labels = data["labels"].astype(np.int32, copy=False)
        splits[split] = {"hiddens": hiddens, "labels": labels}
        print(f"  hiddens: {hiddens.shape}, labels: {len(labels)}, classes: {len(set(labels))}", flush=True)
        print(f"  hidden std: {hiddens.std():.4f}", flush=True)

    train_h, train_y = splits["train"]["hiddens"], splits["train"]["labels"]
    dev_h, dev_y = splits["dev"]["hiddens"], splits["dev"]["labels"]
    test_h, test_y = splits["test"]["hiddens"], splits["test"]["labels"]

    # 2. Baseline: Hidden + standardize + KMeans (test set, 10 seeds)
    print(f"\n-- Baseline: Hidden + standardize + KMeans (test set, 10 seeds) --", flush=True)
    res_base = kmeans_eval_multi_seed(standardize(test_h), test_y)
    print(f"  v_measure = {res_base['v_mean']:.4f} ± {res_base['v_std']:.4f}", flush=True)
    print(f"  NMI       = {res_base['nmi_mean']:.4f} ± {res_base['nmi_std']:.4f}", flush=True)
    print(f"  ARI       = {res_base['ari_mean']:.4f} ± {res_base['ari_std']:.4f}", flush=True)

    # 3. 训练 5 seed 集成 (train 训练, dev 选 best_state, test 评估)
    print(f"\n-- 训练 {len(args.seeds)} seeds 监督对比学习 (train 训练, dev 选 best_state) --", flush=True)
    all_test_emb = []
    for seed in args.seeds:
        t0 = time.time()
        print(f"\n  seed={seed}:", flush=True)
        model, best_dev_v = train_one(
            seed, train_h, train_y, dev_h, dev_y,
            args.temperature, args.n_epochs, args.device
        )
        model.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(test_h).float()
            test_emb = model(test_t).cpu().numpy()
        all_test_emb.append(test_emb)
        res_single = kmeans_eval(test_emb, test_y)
        emb_avg = np.mean(all_test_emb, axis=0)
        emb_avg = emb_avg / (np.linalg.norm(emb_avg, axis=1, keepdims=True) + 1e-12)
        res_ens = kmeans_eval(emb_avg, test_y)
        print(f"  seed={seed} dev_v={best_dev_v:.4f} test_v={res_single['v_measure']:.4f} "
              f"ens_v={res_ens['v_measure']:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # 4. 最终评估 (test set, 10 seeds KMeans)
    emb_avg = np.mean(all_test_emb, axis=0)
    emb_avg = emb_avg / (np.linalg.norm(emb_avg, axis=1, keepdims=True) + 1e-12)

    print(f"\n{'='*60}", flush=True)
    print(f"最终评估 (test set, {len(test_y)} samples, 10 seeds KMeans 均值±std):", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n  A. Baseline: Hidden + standardize + KMeans (无监督):")
    print(f"     v_measure = {res_base['v_mean']:.4f} ± {res_base['v_std']:.4f}", flush=True)

    res_a = kmeans_eval_multi_seed(emb_avg, test_y)
    print(f"\n  B. Projection + KMeans (监督, train 训练 → test 评估):")
    print(f"     v_measure = {res_a['v_mean']:.4f} ± {res_a['v_std']:.4f}", flush=True)
    print(f"     NMI       = {res_a['nmi_mean']:.4f} ± {res_a['nmi_std']:.4f}", flush=True)
    print(f"     ARI       = {res_a['ari_mean']:.4f} ± {res_a['ari_std']:.4f}", flush=True)

    res_b = kmeans_eval_multi_seed(standardize(emb_avg), test_y)
    print(f"\n  C. Projection + standardize + KMeans:")
    print(f"     v_measure = {res_b['v_mean']:.4f} ± {res_b['v_std']:.4f}", flush=True)

    pca = PCA(n_components=64, random_state=42)
    emb_pca = pca.fit_transform(emb_avg)
    res_c = kmeans_eval_multi_seed(emb_pca, test_y)
    print(f"\n  D. Projection + PCA(64) + KMeans:")
    print(f"     v_measure = {res_c['v_mean']:.4f} ± {res_c['v_std']:.4f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"结论:", flush=True)
    print(f"  无监督 baseline (Hidden+std+KMeans):       v = {res_base['v_mean']:.4f}", flush=True)
    print(f"  监督投影 (train训练→test评估, held-out):   v = {res_a['v_mean']:.4f}", flush=True)
    print(f"  提升: +{(res_a['v_mean']-res_base['v_mean'])*100:.1f}%", flush=True)
    print(f"  注: 监督方法 (用 train 标签), 不可与无监督 MTEB 直接对比", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
