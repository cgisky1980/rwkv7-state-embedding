#!/usr/bin/env python3
"""任务六: 聚类 - sklearn 全文版 20NG 监督投影 + KMeans

论文 §4.3.2 规范:
  数据: sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
        18,253 样本 (去重后), 分层 70/15/15 split: train 12,777 / dev 2,738 / test 2,738
  特征: albatross L12 hidden state (1024维)
  范式: train 训练→dev 选 best_state→test held-out 评估

支持:
  - GPU 加速 (RTX 2080 Ti, 18k样本训练<1min)
  - 最优超参 (τ=0.3, 80k pairs/epoch, drop=0.1)
  - 5 seed 集成
  - 多配置对比 (baseline + 最优)

依赖:
  - cache_python/cluster_sklearn_20ng_l12_{train,dev,test}.npz
    (由 extract_features.py --task cluster_sklearn 生成)

运行:
  cd paper/scripts
  # Baseline (τ=0.5, 20k pairs, drop=0.2)
  uv run --project ../../scripts python 06_cluster_sklearn.py --device cuda
  # 最优 (τ=0.3, 80k pairs, drop=0.1)
  uv run --project ../../scripts python 06_cluster_sklearn.py --device cuda --temperature 0.3 --n-pairs 80000 --dropout 0.1
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
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

LAYER = 12
N_CLUSTERS = 20


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

    def forward(self, x):
        x = self.input_norm(x)
        return F.normalize(self.net(x), p=2, dim=-1)


def angle_loss(emb1, emb2, scores, temperature=0.5):
    """AnglE loss: 基于角度的对比损失"""
    cos_sim = (emb1 * emb2).sum(dim=-1)
    # 分数映射到 [0, 1]: score=1 (同类) → 正样本, score=0 (不同类) → 负样本
    labels = scores.float()
    # BCE on cos/temperature
    logits = cos_sim / temperature
    loss = -(labels * torch.log(torch.sigmoid(logits) + 1e-8) +
             (1 - labels) * torch.log(1 - torch.sigmoid(logits) + 1e-8))
    return loss.mean()


def kmeans_eval(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = N_CLUSTERS, n_init: int = 10) -> dict:
    """KMeans 评估，返回 v_measure/NMI/ARI (n_init 次的均值)"""
    v_scores = []
    nmi_scores = []
    ari_scores = []
    for rs in range(n_init):
        pred = KMeans(n_clusters=n_clusters, random_state=rs, n_init=1).fit_predict(embeddings)
        v_scores.append(v_measure_score(labels, pred))
        nmi_scores.append(normalized_mutual_info_score(labels, pred))
        ari_scores.append(adjusted_rand_score(labels, pred))
    return {
        "v_measure": float(np.mean(v_scores)),
        "v_std": float(np.std(v_scores)),
        "nmi": float(np.mean(nmi_scores)),
        "ari": float(np.mean(ari_scores)),
    }


def standardize(x: np.ndarray) -> np.ndarray:
    """标准化特征 (zero mean, unit variance)"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def make_supervised_pairs(hiddens, labels, n_pairs=None, seed=42):
    """构造监督 pair: 50% 同类 (score=1), 50% 不同类 (score=0)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n_pairs is None:
        n_pairs = n * 2

    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos

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


def train_one(seed, train_h, train_y, dev_h, dev_y,
              temperature=0.5, n_epochs=30, device="cpu", n_pairs_per_epoch=20000,
              dropout=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_h.shape[1]
    model = MlpProj(input_dim=input_dim, hidden_dim=512, output_dim=128, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_v = -1.0
    best_state = None
    batch_size = 256

    for epoch in range(n_epochs):
        s1, s2, scores = make_supervised_pairs(train_h, train_y, n_pairs=n_pairs_per_epoch, seed=seed * 1000 + epoch)
        s1_t = torch.from_numpy(s1).float().to(device)
        s2_t = torch.from_numpy(s2).float().to(device)
        scores_t = torch.from_numpy(scores).float().to(device)

        model.train()
        perm = torch.randperm(len(scores_t), device=device)
        for i in range(0, len(scores_t), batch_size):
            idx = perm[i:i + batch_size]
            emb1 = model(s1_t[idx])
            emb2 = model(s2_t[idx])
            loss = angle_loss(emb1, emb2, scores_t[idx], temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # dev 评估
        model.eval()
        with torch.no_grad():
            dev_t = torch.from_numpy(dev_h).float().to(device)
            dev_emb = model(dev_t).cpu().numpy()
        dev_v = kmeans_eval(dev_emb, dev_y)["v_measure"]
        if dev_v > best_v:
            best_v = dev_v
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch={epoch+1}/{n_epochs} loss={loss.item():.4f} dev_v={dev_v:.4f} best={best_v:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model, best_v


def main():
    parser = argparse.ArgumentParser(description="任务六: sklearn 全文版 20NG 监督投影聚类")
    parser.add_argument("--cache-dir", type=Path, default=Path("../cache_python"))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务六: sklearn 全文版 20NG 监督投影聚类", flush=True)
    print(f"  数据: sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))")
    print(f"  18,253 样本, 分层 70/15/15 split")
    print(f"  τ={args.temperature}, dropout={args.dropout}, pairs/epoch={args.n_pairs}")
    print(f"  seeds={args.seeds}, epochs={args.n_epochs}, device={args.device}")
    print("=" * 60, flush=True)

    # 1. 加载预 split 特征
    splits = {}
    for split in ["train", "dev", "test"]:
        cache_path = args.cache_dir / f"cluster_sklearn_20ng_l{LAYER}_{split}.npz"
        with np.load(cache_path) as data:
            hiddens = data["hiddens"].astype(np.float32, copy=False)
            labels = data["labels"].astype(np.int32, copy=False)
        splits[split] = {"hiddens": hiddens, "labels": labels}
        print(f"  {split}: {hiddens.shape}, classes: {len(set(labels))}", flush=True)

    train_h, train_y = splits["train"]["hiddens"], splits["train"]["labels"]
    dev_h, dev_y = splits["dev"]["hiddens"], splits["dev"]["labels"]
    test_h, test_y = splits["test"]["hiddens"], splits["test"]["labels"]

    # 2. 无监督 baseline
    print("\n-- 无监督 baseline --", flush=True)
    base_res = kmeans_eval(standardize(test_h), test_y)
    print(f"  Hidden + standardize + KMeans: v={base_res['v_measure']:.4f} ± {base_res['v_std']:.4f}", flush=True)

    # 3. 监督投影训练 (5 seed)
    print(f"\n-- 监督投影训练 (τ={args.temperature}, pairs={args.n_pairs}, drop={args.dropout}) --", flush=True)
    all_test_emb = []
    results = []
    t_total = time.time()
    for seed in args.seeds:
        t0 = time.time()
        model, best_dev_v = train_one(
            seed, train_h, train_y, dev_h, dev_y,
            temperature=args.temperature, n_epochs=args.n_epochs,
            device=args.device, n_pairs_per_epoch=args.n_pairs,
            dropout=args.dropout,
        )
        model.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(test_h).float().to(args.device)
            test_emb = model(test_t).cpu().numpy()
        all_test_emb.append(test_emb)
        single_res = kmeans_eval(test_emb, test_y)
        # 集成 (embedding 平均)
        emb_avg = np.mean(all_test_emb, axis=0)
        emb_avg = emb_avg / (np.linalg.norm(emb_avg, axis=1, keepdims=True) + 1e-12)
        ens_res = kmeans_eval(emb_avg, test_y)
        print(f"  seed={seed} dev={best_dev_v:.4f} test={single_res['v_measure']:.4f} ens={ens_res['v_measure']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        results.append({
            "seed": seed, "dev": best_dev_v,
            "test_v": single_res["v_measure"], "test_nmi": single_res["nmi"], "test_ari": single_res["ari"],
            "ens_v": ens_res["v_measure"],
        })

    # 4. 最终结果
    final_v = results[-1]["ens_v"]
    single_mean = np.mean([r["test_v"] for r in results])
    print(f"\n{'='*60}")
    print(f"结果汇总 (τ={args.temperature}, pairs={args.n_pairs}, drop={args.dropout})")
    print(f"  无监督 baseline: v={base_res['v_measure']:.4f} ± {base_res['v_std']:.4f}")
    print(f"  监督单模型均值: v={single_mean:.4f}")
    print(f"  {len(args.seeds)} seed 集成:    v={final_v:.4f}")
    print(f"  提升: {(final_v - base_res['v_measure']) / base_res['v_measure'] * 100:.1f}%")
    print(f"  总耗时: {time.time()-t_total:.1f}s")
    print(f"{'='*60}")

    # 5. 保存结果
    import json
    out = {
        "config": {"temperature": args.temperature, "dropout": args.dropout, "n_pairs": args.n_pairs, "n_epochs": args.n_epochs},
        "baseline": base_res,
        "single_mean": float(single_mean),
        "ensemble": float(final_v),
        "results": results,
    }
    out_path = args.cache_dir / f"cluster_sklearn_result_t{args.temperature}_p{args.n_pairs}_d{args.dropout}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
