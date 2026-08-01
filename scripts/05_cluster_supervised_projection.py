"""
任务五：聚类 - 监督对比学习训练专用 Projection

核心思路:
  STS projection 迁移到聚类失败 (0.14), 因为 STS 学的是"相似度排序"而非"类间分离".
  新方案: 直接用 twentynewsgroups 的 label 做监督对比学习, 训练聚类专用 projection.

  这是合理的迁移学习:
    - twentynewsgroups 全量 59545 样本
    - 划分 train/test (80/20, stratified)
    - train 用监督对比学习 (同类=正样本 score=1, 不同类=负样本 score=0)
    - test 用 projection + KMeans 评估 (真正的 held-out 评估)

  与 STS 任务用 NLI/extra_train/sickr 训练是同样范式: 用监督数据训练 projection,
  然后在 held-out test 上评估.

依赖:
  - cluster_full_l12.npz (由 extract_features.py --task cluster_full 生成)

运行:
  cd paper/scripts
  uv run --project ../../scripts python 05_cluster_supervised_projection.py
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
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from cache import load_npz  # noqa: E402

LAYER = 12
N_CLUSTERS = 20


# ============================================================
# MlpProj (与 02_sts_similarity.py 一致, 但不强制 L2 normalize)
# ============================================================
class MlpProj(nn.Module):
    """2层 MLP 投影器."""

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


def angle_loss(emb1: torch.Tensor, emb2: torch.Tensor, scores: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """AnglE loss: 同类 (score=1) 拉近, 不同类 (score=0) 推开."""
    cos_sim = (emb1 * emb2).sum(dim=-1)
    cos_sim_scaled = cos_sim / temperature
    loss = -torch.mean(
        scores * torch.log(torch.sigmoid(cos_sim_scaled) + 1e-8)
        + (1 - scores) * torch.log(1 - torch.sigmoid(cos_sim_scaled) + 1e-8)
    )
    return loss


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


def make_supervised_pairs(hiddens, labels, n_pairs=None, seed=42):
    """构造监督 pair: 50% 同类 (score=1), 50% 不同类 (score=0)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n_pairs is None:
        n_pairs = n * 2  # 每个样本约 2 个 pair

    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos

    # 同类 pair
    classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in classes}

    pairs_s1 = []
    pairs_s2 = []
    pairs_scores = []

    # 正样本对 (同类)
    for _ in range(n_pos):
        c = rng.choice(classes)
        if len(class_indices[c]) < 2:
            continue
        i, j = rng.choice(class_indices[c], 2, replace=False)
        pairs_s1.append(hiddens[i])
        pairs_s2.append(hiddens[j])
        pairs_scores.append(1.0)

    # 负样本对 (不同类)
    for _ in range(n_neg):
        c1, c2 = rng.choice(classes, 2, replace=False)
        i = rng.choice(class_indices[c1])
        j = rng.choice(class_indices[c2])
        pairs_s1.append(hiddens[i])
        pairs_s2.append(hiddens[j])
        pairs_scores.append(0.0)

    s1 = np.array(pairs_s1, dtype=np.float32)
    s2 = np.array(pairs_s2, dtype=np.float32)
    scores = np.array(pairs_scores, dtype=np.float32)
    return s1, s2, scores


def train_one(seed, train_hiddens, train_labels, dev_hiddens, dev_labels,
              temperature=0.5, n_epochs=30, device="cpu", n_pairs_per_epoch=20000, dropout=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_hiddens.shape[1]
    model = MlpProj(input_dim=input_dim, hidden_dim=512, output_dim=128, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_v = -1.0
    best_state = None
    batch_size = 256

    for epoch in range(n_epochs):
        # 每个 epoch 重新采样 pair
        s1, s2, scores = make_supervised_pairs(train_hiddens, train_labels, n_pairs=n_pairs_per_epoch, seed=seed * 1000 + epoch)
        s1_t = torch.from_numpy(s1).float().to(device)
        s2_t = torch.from_numpy(s2).float().to(device)
        scores_t = torch.from_numpy(scores).float().to(device)

        model.train()
        perm = torch.randperm(len(scores_t))
        for i in range(0, len(scores_t), batch_size):
            idx = perm[i : i + batch_size]
            emb1 = model(s1_t[idx])
            emb2 = model(s2_t[idx])
            loss = angle_loss(emb1, emb2, scores_t[idx], temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # 评估 (dev set)
        model.eval()
        with torch.no_grad():
            dev_t = torch.from_numpy(dev_hiddens).float().to(device)
            dev_emb = model(dev_t).cpu().numpy()
        dev_v = kmeans_eval(dev_emb, dev_labels)["v_measure"]
        if dev_v > best_v:
            best_v = dev_v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch={epoch+1}/{n_epochs} loss={loss.item():.4f} dev_v={dev_v:.4f} best={best_v:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model, best_v


def main():
    parser = argparse.ArgumentParser(description="任务五: 监督对比学习聚类")
    parser.add_argument("--cache-dir", type=Path, default=Path("../cache_python"))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test-size", type=float, default=0.2, help="test split 比例")
    parser.add_argument("--dev-size", type=float, default=0.2, help="dev split 占 train 的比例")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务五: 监督对比学习聚类 (twentynewsgroups)", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载全量数据
    cache_path = args.cache_dir / f"cluster_full_l{LAYER}.npz"
    print(f"\n加载全量数据: {cache_path}", flush=True)
    states, hiddens = load_npz(cache_path)
    with np.load(cache_path) as data:
        labels = data["labels"].astype(np.int32)
    print(f"  hiddens: {hiddens.shape}, labels: {len(labels)}, classes: {len(set(labels))}", flush=True)
    print(f"  hidden std: {hiddens.std():.4f}", flush=True)

    # 2. 划分 train/dev/test (stratified, 64/16/20)
    #    dev 用于模型选择 (best_state), test 用于最终评估, 避免数据泄露
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=args.test_size, random_state=42, stratify=labels
    )
    train_idx, dev_idx = train_test_split(
        train_idx, test_size=args.dev_size, random_state=42, stratify=labels[train_idx]
    )
    train_hiddens = hiddens[train_idx]
    train_labels = labels[train_idx]
    dev_hiddens = hiddens[dev_idx]
    dev_labels = labels[dev_idx]
    test_hiddens = hiddens[test_idx]
    test_labels = labels[test_idx]
    print(f"\n  train: {len(train_labels)} (每类 {len(train_labels)//N_CLUSTERS})", flush=True)
    print(f"  dev:   {len(dev_labels)} (每类 {len(dev_labels)//N_CLUSTERS}) ← 用于模型选择", flush=True)
    print(f"  test:  {len(test_labels)} (每类 {len(test_labels)//N_CLUSTERS}) ← 最终评估", flush=True)

    # 3. Baseline: Hidden + standardize + KMeans (test set)
    print(f"\n-- Baseline: Hidden + standardize + KMeans (test set) --", flush=True)
    res_base = kmeans_eval(standardize(test_hiddens), test_labels)
    print(f"  v_measure = {res_base['v_measure']:.4f}", flush=True)
    print(f"  NMI       = {res_base['nmi']:.4f}", flush=True)
    print(f"  ARI       = {res_base['ari']:.4f}", flush=True)

    # 4. 训练 5 seed 集成
    print(f"\n-- 训练 {len(args.seeds)} seeds 监督对比学习 --", flush=True)
    all_test_emb = []
    for seed in args.seeds:
        t0 = time.time()
        print(f"\n  seed={seed}:", flush=True)
        model, best_dev_v = train_one(
            seed, train_hiddens, train_labels, dev_hiddens, dev_labels,
            args.temperature, args.n_epochs, args.device, args.n_pairs, args.dropout
        )
        # test 评估
        model.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(test_hiddens).float().to(args.device)
            test_emb = model(test_t).cpu().numpy()
        all_test_emb.append(test_emb)
        # 单 seed 评估
        res_single = kmeans_eval(test_emb, test_labels)
        # 集成
        emb_avg = np.mean(all_test_emb, axis=0)
        emb_avg = emb_avg / (np.linalg.norm(emb_avg, axis=1, keepdims=True) + 1e-12)
        res_ens = kmeans_eval(emb_avg, test_labels)
        print(f"  seed={seed} dev_v={best_dev_v:.4f} test_v={res_single['v_measure']:.4f} "
              f"ens_v={res_ens['v_measure']:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # 5. 最终结果
    emb_avg = np.mean(all_test_emb, axis=0)
    emb_avg = emb_avg / (np.linalg.norm(emb_avg, axis=1, keepdims=True) + 1e-12)

    print(f"\n{'='*60}", flush=True)
    print(f"最终评估 (test set, {len(test_labels)} samples):", flush=True)
    print(f"{'='*60}", flush=True)

    # 方法 A: Projection + KMeans (直接)
    res_a = kmeans_eval(emb_avg, test_labels)
    print(f"\n  A. Projection + KMeans (直接):")
    print(f"     v_measure = {res_a['v_measure']:.4f}", flush=True)
    print(f"     NMI       = {res_a['nmi']:.4f}", flush=True)
    print(f"     ARI       = {res_a['ari']:.4f}", flush=True)

    # 方法 B: Projection + standardize + KMeans
    res_b = kmeans_eval(standardize(emb_avg), test_labels)
    print(f"\n  B. Projection + standardize + KMeans:")
    print(f"     v_measure = {res_b['v_measure']:.4f}", flush=True)

    # 方法 C: Projection + PCA + KMeans
    pca = PCA(n_components=64, random_state=42)
    emb_pca = pca.fit_transform(emb_avg)
    res_c = kmeans_eval(emb_pca, test_labels)
    print(f"\n  C. Projection + PCA(64) + KMeans:")
    print(f"     v_measure = {res_c['v_measure']:.4f}", flush=True)

    # 方法 D: Hidden + Projection 拼接
    concat = np.concatenate([standardize(test_hiddens), standardize(emb_avg)], axis=1)
    res_d = kmeans_eval(concat, test_labels)
    print(f"\n  D. Hidden + Projection 拼接 + standardize + KMeans:")
    print(f"     v_measure = {res_d['v_measure']:.4f}", flush=True)

    # 结论
    print(f"\n{'='*60}", flush=True)
    print(f"结论:", flush=True)
    print(f"  Baseline (Hidden + standardize):    v_measure = {res_base['v_measure']:.4f}", flush=True)
    print(f"  Projection + KMeans (直接):         v_measure = {res_a['v_measure']:.4f}", flush=True)
    print(f"  Projection + standardize + KMeans:  v_measure = {res_b['v_measure']:.4f}", flush=True)
    print(f"  Projection + PCA(64) + KMeans:      v_measure = {res_c['v_measure']:.4f}", flush=True)
    print(f"  Hidden + Projection 拼接:           v_measure = {res_d['v_measure']:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
