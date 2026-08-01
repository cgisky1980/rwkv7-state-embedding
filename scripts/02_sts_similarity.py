"""
任务二：语义相似度 - Universal Sentence Embedding (多数据集训练)

方法：
1. 特征：RWKV-7 最后一层 hidden state (1024 维 mean-pooled)
2. 模型：2层 MLP (1024→512→128) + BatchNorm + Dropout + L2 Normalize
3. Loss：AnglE（基于角度的对比损失，缓解各向异性，τ=0.50）
4. 集成：5 个不同 seed 模型 embedding 平均
5. 训练数据：STS-B train + NLI + extra_train + SICK-R = 47.6k pairs
   (相比仅用 STS-B train 5.7k, 数据量提升 8x, 大幅缓解过拟合)

数据集：
  - 训练: sts_train (5.7k) + nli_train (10k) + extra_train (22k) + sickr (9.9k)
  - 评估: STS-Benchmark dev/test

评估：Spearman 相关系数

特征提取:
    run_with_msvc.bat extract_features.py --task sts
    run_with_msvc.bat extract_features.py --task sts_extra
    生成 cache_python/sts_pair_l{LAYER}_{train,dev,test,nli_train,extra_train,sickr}.npz

运行:
    cd paper/scripts
    uv run --project ../../scripts python 02_sts_similarity.py
    uv run --project ../../scripts python 02_sts_similarity.py --no-extra  # 仅 STS-B train (baseline)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from cache import load_npz  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)

LAYER = 12


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================
# 模型
# ============================================================
class MlpProj(nn.Module):
    """2层 MLP 投影器: input → BatchNorm → Linear → GELU → LayerNorm → Dropout
                      → Linear → GELU → LayerNorm → Dropout → Linear → L2 Norm
    """

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


# ============================================================
# AnglE Loss
# ============================================================
def angle_loss(emb1: torch.Tensor, emb2: torch.Tensor, scores: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """AnglE loss: 基于角度的对比损失 (albatross 路径最优 τ=0.50)"""
    cos_sim = (emb1 * emb2).sum(dim=-1)
    cos_sim_scaled = cos_sim / temperature
    s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    loss = -torch.mean(
        s * torch.log(torch.sigmoid(cos_sim_scaled) + 1e-8)
        + (1 - s) * torch.log(1 - torch.sigmoid(cos_sim_scaled) + 1e-8)
    )
    return loss


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return spearmanr(x, y).correlation


# ============================================================
# 训练
# ============================================================
def train_one(seed, train_data, dev_data, test_data, temperature=0.5, n_epochs=50, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    s1_train, s2_train, scores_train = train_data
    s1_dev, s2_dev, scores_dev = dev_data
    s1_test, s2_test, scores_test = test_data

    model = MlpProj(input_dim=s1_train.shape[1], hidden_dim=512, output_dim=128, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_dev_sp = -1.0
    best_state = None
    n_train = len(scores_train)
    batch_size = 256

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            emb1 = model(s1_train[idx].to(device))
            emb2 = model(s2_train[idx].to(device))
            loss = angle_loss(emb1, emb2, scores_train[idx].to(device), temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            emb1_dev = model(s1_dev.to(device)).cpu().numpy()
            emb2_dev = model(s2_dev.to(device)).cpu().numpy()
        dev_sp = spearman_corr((emb1_dev * emb2_dev).sum(axis=1), scores_dev.numpy())
        if dev_sp > best_dev_sp:
            best_dev_sp = dev_sp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb1_test = model(s1_test.to(device)).cpu().numpy()
        emb2_test = model(s2_test.to(device)).cpu().numpy()

    cos_test = (emb1_test * emb2_test).sum(axis=1)
    test_sp = spearman_corr(cos_test, scores_test.numpy())
    return best_dev_sp, test_sp, emb1_test, emb2_test, best_state


# ============================================================
# 无监督 baseline
# ============================================================
def unsupervised_baseline(hiddens_test, scores_test):
    """无监督 baseline: hidden cosine similarity"""
    s1 = hiddens_test[0::2]
    s2 = hiddens_test[1::2]
    # L2 normalize
    s1 = s1 / (np.linalg.norm(s1, axis=1, keepdims=True) + 1e-8)
    s2 = s2 / (np.linalg.norm(s2, axis=1, keepdims=True) + 1e-8)
    cos = (s1 * s2).sum(axis=1)
    return spearman_corr(cos, scores_test)


def main():
    parser = argparse.ArgumentParser(description="任务二: 语义相似度")
    parser.add_argument("--cache-dir", type=Path, default=Path("../cache_python"))
    parser.add_argument("--data-dir", type=Path, default=Path("../../data/sts"))
    parser.add_argument("--temperature", type=float, default=0.50,
                        help="AnglE loss 温度 (albatross 路径最优 0.50, 非 Rust 路径的 0.1)")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-extra", action="store_true",
                        help="仅用 STS-B train (baseline), 不用 NLI/extra_train/sickr")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务二: 语义相似度 - Universal Sentence Embedding", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载 STS-B dev/test (评估用) - 始终从原 data/sts 目录读取
    sts_orig_dir = args.data_dir.parent / "sts" if args.data_dir.name != "sts" else args.data_dir
    splits = {}
    for split in ["dev", "test"]:
        cache_path = args.cache_dir / f"sts_pair_l{LAYER}_{split}.npz"
        states, hiddens = load_npz(cache_path)
        records = read_jsonl(sts_orig_dir / f"sts_{split}.jsonl")
        scores = np.array([r["score"] for r in records], dtype=np.float32)
        splits[split] = {"hiddens": hiddens, "scores": scores}
        print(f"  {split}: {len(scores)} pairs, hiddens {hiddens.shape}", flush=True)

    # 2. 加载训练数据
    # 训练集: sts_train (必有) + nli_train + extra_train + sickr (可选)
    train_hiddens_list = []
    train_scores_list = []

    # STS-B train
    cache_path = args.cache_dir / f"sts_pair_l{LAYER}_train.npz"
    states, hiddens = load_npz(cache_path)
    records = read_jsonl(args.data_dir / "sts_train.jsonl")
    scores = np.array([r["score"] for r in records], dtype=np.float32)
    train_hiddens_list.append(hiddens)
    train_scores_list.append(scores)
    print(f"  train (STS-B): {len(scores)} pairs", flush=True)

    if not args.no_extra:
        # 额外训练数据: nli_train, extra_train, sickr
        extra_datasets = ["nli_train", "extra_train", "sickr"]
        for name in extra_datasets:
            cache_path = args.cache_dir / f"sts_pair_l{LAYER}_{name}.npz"
            if not cache_path.exists():
                print(f"  [skip] {name}: {cache_path} 不存在", flush=True)
                continue
            states, hiddens = load_npz(cache_path)
            data_path = args.data_dir / f"{name}.jsonl"
            records = read_jsonl(data_path)
            scores = np.array([r["score"] for r in records], dtype=np.float32)
            train_hiddens_list.append(hiddens)
            train_scores_list.append(scores)
            print(f"  train ({name}): {len(scores)} pairs", flush=True)

    # 合并训练数据
    train_hiddens = np.concatenate(train_hiddens_list, axis=0)
    train_scores = np.concatenate(train_scores_list, axis=0)
    print(f"\n  总训练数据: {len(train_scores)} pairs", flush=True)

    # 3. 无监督 baseline
    print(f"\n-- 无监督 baseline --", flush=True)
    unsup_sp = unsupervised_baseline(splits["test"]["hiddens"], splits["test"]["scores"])
    print(f"  Hidden cosine: Spearman = {unsup_sp:.4f}", flush=True)

    # 4. 构造句子对 (交错: [s1_p0, s2_p0, s1_p1, ...])
    def make_pairs(hiddens_arr, scores_arr):
        s1 = hiddens_arr[0::2]
        s2 = hiddens_arr[1::2]
        return (
            torch.from_numpy(np.ascontiguousarray(s1)).float(),
            torch.from_numpy(np.ascontiguousarray(s2)).float(),
            torch.from_numpy(np.ascontiguousarray(scores_arr)).float(),
        )

    train_data = make_pairs(train_hiddens, train_scores)
    dev_data = make_pairs(splits["dev"]["hiddens"], splits["dev"]["scores"])
    test_data = make_pairs(splits["test"]["hiddens"], splits["test"]["scores"])
    print(f"\n训练: {len(train_data[2])} pairs, 特征维度: {train_data[0].shape[1]}", flush=True)

    # 5. 训练 5 seed 集成
    print(f"\n-- 训练 {len(args.seeds)} seeds 集成 --", flush=True)
    all_emb_test = []
    saved_projections = []  # 保存每个 seed 的 state_dict, 用于聚类等其他任务
    for seed in args.seeds:
        t0 = time.time()
        dev_sp, test_sp, emb1_test, emb2_test, proj_state = train_one(
            seed, train_data, dev_data, test_data, args.temperature, args.n_epochs, args.device
        )
        all_emb_test.append((emb1_test, emb2_test))
        saved_projections.append(proj_state)
        # 当前集成
        emb1_avg = np.mean([e[0] for e in all_emb_test], axis=0)
        emb2_avg = np.mean([e[1] for e in all_emb_test], axis=0)
        emb1_avg = emb1_avg / (np.linalg.norm(emb1_avg, axis=1, keepdims=True) + 1e-12)
        emb2_avg = emb2_avg / (np.linalg.norm(emb2_avg, axis=1, keepdims=True) + 1e-12)
        ens_sp = spearman_corr((emb1_avg * emb2_avg).sum(axis=1), test_data[2].numpy())
        print(f"  seed={seed} dev={dev_sp:.4f} test={test_sp:.4f} ens={ens_sp:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # 保存 projection 模型 (供聚类等其他任务使用)
    proj_save_path = args.cache_dir / f"universal_projection_l{LAYER}.pt"
    torch.save({
        "seeds": args.seeds,
        "state_dicts": saved_projections,
        "config": {
            "input_dim": train_data[0].shape[1],
            "hidden_dim": 512,
            "output_dim": 128,
            "temperature": args.temperature,
            "train_pairs": len(train_data[2]),
        },
    }, proj_save_path)
    print(f"\n  保存 projection: {proj_save_path}", flush=True)

    # 6. 最终结果
    emb1_avg = np.mean([e[0] for e in all_emb_test], axis=0)
    emb2_avg = np.mean([e[1] for e in all_emb_test], axis=0)
    emb1_avg = emb1_avg / (np.linalg.norm(emb1_avg, axis=1, keepdims=True) + 1e-12)
    emb2_avg = emb2_avg / (np.linalg.norm(emb2_avg, axis=1, keepdims=True) + 1e-12)
    final_sp = spearman_corr((emb1_avg * emb2_avg).sum(axis=1), test_data[2].numpy())
    single_mean = np.mean([spearman_corr((e[0]*e[1]).sum(axis=1), test_data[2].numpy()) for e in all_emb_test])

    print(f"\n{'='*60}", flush=True)
    print(f"结论:", flush=True)
    print(f"  无监督 Hidden cosine:  Spearman = {unsup_sp:.4f}", flush=True)
    print(f"  单模型均值:            Spearman = {single_mean:.4f}", flush=True)
    print(f"  {len(all_emb_test)}seed 集成:             Spearman = {final_sp:.4f}", flush=True)
    print(f"  训练数据:              {len(train_data[2])} pairs", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
