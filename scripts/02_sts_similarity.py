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


# ============================================================
# 缓存加载 (含 scores)
# ============================================================
def load_npz_with_scores(path: Path):
    """加载 .npz 缓存 float32。

    sts_pair 缓存含 scores; states 为可选 (去泄露 clean 缓存仅存 hiddens+scores)。
    """
    path = Path(path)
    with np.load(path) as data:
        states = data["states"].astype(np.float32, copy=False) if "states" in data.files else None
        hiddens = data["hiddens"].astype(np.float32, copy=False)
        scores = data["scores"].astype(np.float32, copy=False)
    return states, hiddens, scores


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
def train_one(seed, train_data, dev_data, test_data, temperature=0.5, n_epochs=50, device="cpu",
              hidden_dim=512, output_dim=128, dropout=0.2, init_state=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    feature_dim = train_data[0].shape[1]
    # 大特征 (如 state 49152 维) 一次性移 GPU 会爆显存, 故训练数据保持 CPU, 按 batch 移 GPU.
    # dev/test 较小, 一次性移 GPU 以加速评估.
    is_big_feature = feature_dim >= 8192
    if is_big_feature:
        s1_train = train_data[0]
        s2_train = train_data[1]
        scores_train = train_data[2]  # 留在 CPU, 用 CPU perm 索引后移 GPU
    else:
        s1_train = train_data[0].to(device)
        s2_train = train_data[1].to(device)
        scores_train = train_data[2].to(device)
    s1_dev = dev_data[0].to(device)
    s2_dev = dev_data[1].to(device)
    scores_dev_cpu = dev_data[2]
    s1_test = test_data[0].to(device)
    s2_test = test_data[1].to(device)
    scores_test_cpu = test_data[2]

    model = MlpProj(input_dim=train_data[0].shape[1], hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout).to(device)
    if init_state is not None:
        # 两阶段: 从预训练投影器 (如 07 InfoNCE) 初始化后微调
        model.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_dev_sp = -1.0
    best_state = None
    n_train = len(scores_train)
    # GPU 用大 batch, CPU 用小 batch
    is_cuda = device != "cpu" and "cuda" in str(device)
    # 大特征 (49k 维) 前向单 batch 计算量大, 需用小 batch, 否则显存爆
    batch_size = 256 if is_big_feature else (4096 if is_cuda else 256)

    for epoch in range(n_epochs):
        model.train()
        # 大特征数据在 CPU, 索引也要在 CPU
        rand_device = "cpu" if is_big_feature else device
        perm = torch.randperm(n_train, device=rand_device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            if is_big_feature:
                s1_t = s1_train[idx].to(device)
                s2_t = s2_train[idx].to(device)
            else:
                s1_t = s1_train[idx]
                s2_t = s2_train[idx]
            sc_t = scores_train[idx].to(device)
            emb1 = model(s1_t)
            emb2 = model(s2_t)
            loss = angle_loss(emb1, emb2, sc_t, temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            emb1_dev = model(s1_dev).cpu().numpy()
            emb2_dev = model(s2_dev).cpu().numpy()
        dev_sp = spearman_corr((emb1_dev * emb2_dev).sum(axis=1), scores_dev_cpu.numpy())
        if dev_sp > best_dev_sp:
            best_dev_sp = dev_sp
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb1_test = model(s1_test).cpu().numpy()
        emb2_test = model(s2_test).cpu().numpy()

    cos_test = (emb1_test * emb2_test).sum(axis=1)
    test_sp = spearman_corr(cos_test, scores_test_cpu.numpy())
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
    parser.add_argument("--data-dir", type=Path, default=Path("../data/sts"))
    parser.add_argument("--temperature", type=float, default=0.50,
                        help="AnglE loss 温度 (albatross 路径最优 0.50, 非 Rust 路径的 0.1)")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-extra", action="store_true",
                        help="仅用 STS-B train (baseline), 不用 NLI/extra_train/sickr")
    parser.add_argument("--extra-datasets", type=str, default="nli_train,extra_train,sickr",
                        help="额外训练数据集名列表 (逗号分隔, 对应缓存 sts_pair_l{layer}_{name}.npz)")
    parser.add_argument("--proj-suffix", type=str, default="",
                        help="投影器文件名后缀 (区分不同训练数据版本, 如 _million)")
    parser.add_argument("--init-from", type=str, default="",
                        help="预训练投影器 .pt (文件名置于 cache-dir 或完整路径), 初始化权重后微调")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--layer", type=int, default=LAYER, help="WKV state 提取层 (文件名层号)")
    parser.add_argument("--feature", choices=["hidden", "state"], default="hidden",
                        help="用于训练投影器的特征: hidden=最后一层 hidden state, state=WKV state")
    parser.add_argument("--top-k-heads", type=int, default=0,
                        help=">0 时启用 Top-K Head 选择 (仅 state 特征, paper §4.3.4 方法): "
                             "逐 head 评估 dev Spearman → 选 Top-K head 拼接 → PCA 降维 → 投影器")
    parser.add_argument("--pca-dim", type=int, default=256,
                        help="Top-K head 拼接后的 PCA 降维维度")
    parser.add_argument("--head-eval-epochs", type=int, default=10,
                        help="逐 head 评估时小投影器的训练轮数")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务二: 语义相似度 - Universal Sentence Embedding", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载 STS-B dev/test (评估用) - scores 直接来自缓存
    def pick_feature(states, hiddens):
        return states if args.feature == "state" else hiddens

    splits = {}
    for split in ["dev", "test"]:
        cache_path = args.cache_dir / f"sts_pair_l{args.layer}_{split}.npz"
        states, hiddens, scores = load_npz_with_scores(cache_path)
        features = pick_feature(states, hiddens)
        splits[split] = {"features": features, "scores": scores}
        print(f"  {split}: {len(scores)} pairs, {args.feature} {features.shape}", flush=True)

    # 2. 加载训练数据
    # 训练集: sts_train (必有) + nli_train + extra_train + sickr (可选)
    train_feats_list = []
    train_scores_list = []

    # STS-B train
    cache_path = args.cache_dir / f"sts_pair_l{args.layer}_train.npz"
    states, hiddens, scores = load_npz_with_scores(cache_path)
    train_feats_list.append(pick_feature(states, hiddens))
    train_scores_list.append(scores)
    print(f"  train (STS-B): {len(scores)} pairs", flush=True)

    if not args.no_extra:
        # 额外训练数据 (缓存缺失的自动跳过)
        extra_datasets = [n.strip() for n in args.extra_datasets.split(",") if n.strip()]
        for name in extra_datasets:
            cache_path = args.cache_dir / f"sts_pair_l{args.layer}_{name}.npz"
            if not cache_path.exists():
                print(f"  [skip] {name}: {cache_path} 不存在", flush=True)
                continue
            states, hiddens, scores = load_npz_with_scores(cache_path)
            train_feats_list.append(pick_feature(states, hiddens))
            train_scores_list.append(scores)
            print(f"  train ({name}): {len(scores)} pairs", flush=True)

    # 合并训练数据
    train_features = np.concatenate(train_feats_list, axis=0)
    train_scores = np.concatenate(train_scores_list, axis=0)
    print(f"\n  总训练数据: {len(train_scores)} pairs", flush=True)

    # ========================================================
    # Top-K Head 选择 (paper §4.3.4 方法, 仅 state 特征)
    # 1. 逐 head 评估: 每 head 4096 维 → PCA64 → 小投影器 → dev Spearman
    # 2. 选 Top-K head 拼接 → PCA pca_dim → 替换特征
    # ========================================================
    head_selection_state = None
    if args.feature == "state" and args.top_k_heads > 0:
        from sklearn.decomposition import PCA

        HEAD_SIZE_SQ = 64 * 64
        n_head = train_features.shape[1] // HEAD_SIZE_SQ
        assert n_head * HEAD_SIZE_SQ == train_features.shape[1], "state 维度不是 64x64 head 的整数倍"
        dev_feats_raw = splits["dev"]["features"]
        test_feats_raw = splits["test"]["features"]

        print(f"\n-- Top-{args.top_k_heads} Head 选择 (n_head={n_head}, 每 head {HEAD_SIZE_SQ} 维) --", flush=True)

        def make_pairs_np(feats_arr, scores_arr):
            s1 = feats_arr[0::2]
            s2 = feats_arr[1::2]
            return (
                torch.from_numpy(np.ascontiguousarray(s1)).float(),
                torch.from_numpy(np.ascontiguousarray(s2)).float(),
                torch.from_numpy(np.ascontiguousarray(scores_arr)).float(),
            )

        # 1. 逐 head 评估 (dev Spearman 排序, head 选择只用 dev 不接触 test)
        head_dev_scores = []
        for h in range(n_head):
            cols = slice(h * HEAD_SIZE_SQ, (h + 1) * HEAD_SIZE_SQ)
            pca_h = PCA(n_components=64, random_state=42)
            Xtr_h = pca_h.fit_transform(train_features[:, cols].astype(np.float32))
            Xdv_h = pca_h.transform(dev_feats_raw[:, cols].astype(np.float32))
            tr_pairs = make_pairs_np(Xtr_h, train_scores)
            dv_pairs = make_pairs_np(Xdv_h, splits["dev"]["scores"])
            dev_sp_h, _, _, _, _ = train_one(
                42, tr_pairs, dv_pairs, dv_pairs, args.temperature,
                args.head_eval_epochs, args.device,
                hidden_dim=256, output_dim=128, dropout=args.dropout,
            )
            head_dev_scores.append((h, dev_sp_h))
            print(f"  H{h:2d}: dev Spearman = {dev_sp_h:.4f}", flush=True)

        head_dev_scores.sort(key=lambda x: x[1], reverse=True)
        top_heads = [h for h, _ in head_dev_scores[: args.top_k_heads]]
        print(f"  Top-{args.top_k_heads} heads (按 dev 选择): {top_heads}", flush=True)

        # 2. Top-K 拼接 + PCA 降维 (PCA 在 train 句子行上 fit)
        def select_heads(feats):
            return np.concatenate(
                [feats[:, h * HEAD_SIZE_SQ:(h + 1) * HEAD_SIZE_SQ] for h in top_heads], axis=1
            )

        print(f"  拼接 {args.top_k_heads} head → {args.top_k_heads * HEAD_SIZE_SQ} 维, PCA → {args.pca_dim} 维...", flush=True)
        t0_pca = time.time()
        pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=42)
        train_features = pca.fit_transform(select_heads(train_features).astype(np.float32)).astype(np.float32)
        splits["dev"]["features"] = pca.transform(select_heads(dev_feats_raw).astype(np.float32)).astype(np.float32)
        splits["test"]["features"] = pca.transform(select_heads(test_feats_raw).astype(np.float32)).astype(np.float32)
        print(f"  PCA 完成 ({time.time()-t0_pca:.1f}s), 特征维度: {train_features.shape[1]}", flush=True)

        head_selection_state = {
            "head_indices": top_heads,
            "head_size": 64,
            "pca_components": pca.components_.astype(np.float32),
            "pca_mean": pca.mean_.astype(np.float32),
            "head_dev_scores": head_dev_scores,
        }

    # 3. 无监督 baseline
    print(f"\n-- 无监督 baseline --", flush=True)
    unsup_sp = unsupervised_baseline(splits["test"]["features"], splits["test"]["scores"])
    print(f"  {args.feature} cosine: Spearman = {unsup_sp:.4f}", flush=True)

    # 4. 构造句子对 (交错: [s1_p0, s2_p0, s1_p1, ...])
    def make_pairs(feats_arr, scores_arr):
        s1 = feats_arr[0::2]
        s2 = feats_arr[1::2]
        return (
            torch.from_numpy(np.ascontiguousarray(s1)).float(),
            torch.from_numpy(np.ascontiguousarray(s2)).float(),
            torch.from_numpy(np.ascontiguousarray(scores_arr)).float(),
        )

    train_data = make_pairs(train_features, train_scores)
    dev_data = make_pairs(splits["dev"]["features"], splits["dev"]["scores"])
    test_data = make_pairs(splits["test"]["features"], splits["test"]["scores"])
    print(f"\n训练: {len(train_data[2])} pairs, 特征维度: {train_data[0].shape[1]}", flush=True)

    # 5. 训练 5 seed 集成
    # 两阶段微调: 加载预训练投影器 (07 InfoNCE 输出), 每个 seed 用相同初始权重
    init_state = None
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            init_path = args.cache_dir / args.init_from
        ckpt = torch.load(init_path, map_location="cpu")
        init_state = ckpt["state_dicts"][0]
        cfg = ckpt.get("config", {})
        print(f"\n-- 从预训练投影器初始化: {init_path} "
              f"(loss={cfg.get('loss', '?')}, best_dev={cfg.get('best_dev_spearman', '?')}) --", flush=True)

    print(f"\n-- 训练 {len(args.seeds)} seeds 集成 --", flush=True)
    all_emb_test = []
    saved_projections = []  # 保存每个 seed 的 state_dict, 用于聚类等其他任务
    for seed in args.seeds:
        t0 = time.time()
        dev_sp, test_sp, emb1_test, emb2_test, proj_state = train_one(
            seed, train_data, dev_data, test_data, args.temperature, args.n_epochs, args.device,
            hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout,
            init_state=init_state,
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
    if args.feature == "state":
        proj_name = (f"universal_projection_state_topk{args.top_k_heads}_l{args.layer}{args.proj_suffix}.pt"
                     if args.top_k_heads > 0 else f"universal_projection_state_l{args.layer}{args.proj_suffix}.pt")
    else:
        proj_name = f"universal_projection_l{args.layer}{args.proj_suffix}.pt"
    proj_save_path = args.cache_dir / proj_name
    save_dict = {
        "seeds": args.seeds,
        "state_dicts": saved_projections,
        "config": {
            "input_dim": train_data[0].shape[1],
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "train_pairs": len(train_data[2]),
        },
    }
    if head_selection_state is not None:
        save_dict["head_selection"] = head_selection_state
        save_dict["config"]["feature"] = "state_topk"
        save_dict["config"]["top_k_heads"] = args.top_k_heads
    torch.save(save_dict, proj_save_path)
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
