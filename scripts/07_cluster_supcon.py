#!/usr/bin/env python3
"""任务七: 聚类 - SupCon Loss (Supervised Contrastive Learning)

方案2: 用 SupCon Loss 替代 AnglE Loss
  - AnglE: pair-based (每次只看一对样本的相似度)
  - SupCon: batch-based (一个 batch 内所有同类样本互为正样本)
  - SupCon 专门优化类间分离, 更适合聚类任务

参考: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)

数据: sklearn.fetch_20newsgroups 全文版, 70/15/15 split
依赖: cache_python/cluster_sklearn_20ng_l12_{train,dev,test}.npz

运行:
  cd paper/scripts
  uv run --project ../../scripts python 07_cluster_supcon.py --device cuda
  uv run --project ../../scripts python 07_cluster_supcon.py --device cuda --temperature 0.1 --batch-size 512
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
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 128, dropout: float = 0.1):
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


def supcon_loss(embeddings, labels, temperature=0.07):
    """SupCon Loss (Khosla et al. 2020)

    一个 batch 内:
    - 同类样本 (mask[i,j]=1, i≠j) 互为正样本
    - 不同类样本为负样本
    - 优化: 正样本相似度 / 所有样本相似度

    Args:
        embeddings: (B, D) L2-normalized
        labels: (B,) 类别
        temperature: 温度参数 (原论文 0.07, 聚类任务试 0.07-0.5)
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 相似度矩阵 (B, B), 对角线为 1 (自己和自己)
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # 正样本 mask: labels[i] == labels[j] 且 i != j
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    # 排除对角线 (自己)
    eye = torch.eye(batch_size, device=device)
    pos_mask = pos_mask - eye
    # 排除和自己相似 (logits 对角线设为 -inf, 这样 exp=0)
    logits_mask = 1.0 - eye
    # 每个样本至少有 1 个正样本, 否则跳过
    has_pos = pos_mask.sum(dim=1) > 0
    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # log_prob = sim - log(sum(exp(sim)) over j≠i)
    # 用 logsumexp 提高数值稳定性
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()  # 减最大值提高稳定性
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # 对每个 anchor i, 取所有正样本 j 的 log_prob 均值
    pos_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    # 只取有正样本的 anchor
    loss = -pos_log_prob[has_pos].mean()
    return loss


def class_balanced_batch_sampler(labels, batch_size, n_classes, n_batches, seed=42):
    """Class-balanced batch sampler (PK sampler)
    每个 batch: 选 P 个类, 每类 K 个样本, batch_size = P * K
    确保 batch 内每类有足够正样本
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in classes}

    n_per_class = max(1, batch_size // n_classes)
    for _ in range(n_batches):
        # 随机选 P 个类
        selected_classes = rng.choice(classes, size=min(n_classes, len(classes)), replace=False)
        batch_idx = []
        batch_labels = []
        for c in selected_classes:
            if len(class_indices[c]) >= n_per_class:
                chosen = rng.choice(class_indices[c], n_per_class, replace=False)
            else:
                chosen = rng.choice(class_indices[c], n_per_class, replace=True)
            batch_idx.extend(chosen)
            batch_labels.extend([c] * n_per_class)
        yield np.array(batch_idx), np.array(batch_labels)


def kmeans_eval(embeddings, labels, n_clusters=N_CLUSTERS, n_init=10):
    v_scores, nmi_scores, ari_scores = [], [], []
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


def standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def train_one(seed, train_h, train_y, dev_h, dev_y,
              temperature=0.07, n_epochs=30, device="cpu", batch_size=320,
              dropout=0.1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_h.shape[1]
    model = MlpProj(input_dim=input_dim, hidden_dim=512, output_dim=128, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # 一次性把训练数据移到 GPU
    train_h_t = torch.from_numpy(train_h).float().to(device)
    train_y_t = torch.from_numpy(train_y).long().to(device)

    best_v = -1.0
    best_state = None
    n_classes = len(np.unique(train_y))
    # 每个 epoch 的 batch 数
    n_batches = max(1, len(train_y) // batch_size)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_done = 0
        sampler = class_balanced_batch_sampler(
            train_y, batch_size, n_classes, n_batches, seed=seed * 1000 + epoch
        )
        for batch_idx, batch_labels in sampler:
            h = train_h_t[batch_idx]
            y = train_y_t[batch_idx]
            emb = model(h)
            loss = supcon_loss(emb, y, temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_done += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_done)

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
            print(f"    epoch={epoch+1}/{n_epochs} loss={avg_loss:.4f} dev_v={dev_v:.4f} best={best_v:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model, best_v


def main():
    parser = argparse.ArgumentParser(description="任务七: SupCon Loss 聚类")
    parser.add_argument("--cache-dir", type=Path, default=Path("../cache_python"))
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="SupCon 温度 (原论文 0.07, 聚类可试 0.1-0.5)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=320,
                        help="PK sampler: 20类×16=320, 确保每类有16个正样本")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务七: SupCon Loss 聚类 (sklearn 全文版 20NG)", flush=True)
    print(f"  Loss: Supervised Contrastive (Khosla 2020)")
    print(f"  τ={args.temperature}, dropout={args.dropout}, batch={args.batch_size} (PK sampler)")
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

    # 3. SupCon 训练 (5 seed)
    print(f"\n-- SupCon 训练 (τ={args.temperature}, batch={args.batch_size}, drop={args.dropout}) --", flush=True)
    all_test_emb = []
    results = []
    t_total = time.time()
    for seed in args.seeds:
        t0 = time.time()
        model, best_dev_v = train_one(
            seed, train_h, train_y, dev_h, dev_y,
            temperature=args.temperature, n_epochs=args.n_epochs,
            device=args.device, batch_size=args.batch_size, dropout=args.dropout,
        )
        model.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(test_h).float().to(args.device)
            test_emb = model(test_t).cpu().numpy()
        all_test_emb.append(test_emb)
        single_res = kmeans_eval(test_emb, test_y)
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
    print(f"SupCon 结果汇总 (τ={args.temperature}, batch={args.batch_size}, drop={args.dropout})")
    print(f"  无监督 baseline: v={base_res['v_measure']:.4f} ± {base_res['v_std']:.4f}")
    print(f"  SupCon 单模型均值: v={single_mean:.4f}")
    print(f"  {len(args.seeds)} seed 集成:    v={final_v:.4f}")
    print(f"  提升: {(final_v - base_res['v_measure']) / base_res['v_measure'] * 100:.1f}%")
    print(f"  总耗时: {time.time()-t_total:.1f}s")
    print(f"{'='*60}")

    # 5. 保存
    import json
    out = {
        "loss": "supcon",
        "config": {"temperature": args.temperature, "dropout": args.dropout,
                    "batch_size": args.batch_size, "n_epochs": args.n_epochs},
        "baseline": base_res,
        "single_mean": float(single_mean),
        "ensemble": float(final_v),
        "results": results,
    }
    out_path = args.cache_dir / f"cluster_supcon_result_t{args.temperature}_b{args.batch_size}_d{args.dropout}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
