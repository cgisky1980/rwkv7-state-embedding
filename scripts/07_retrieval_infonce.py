#!/usr/bin/env python3
"""任务七: 检索式 InfoNCE 预训练 (bge/e5 范式)

数据: collect_retrieval_sts.py 的 hidden 缓存
  - stride 2 (对): (s1, s2)
  - stride 3 (三元组): (anchor, positive, negative 难负例)

损失: InfoNCE + in-batch negatives (MultipleNegativesRankingLoss 同款):
  每个 anchor 的候选 = batch 内全部 positive (in-batch 负例)
                       + batch 内全部显式负例 (难负例)
  缺失显式负例的行, 用 shuffle 后邻位样本的 positive 填充 (合法随机负例)。
  label = 对角线, cross_entropy。

评估: STS-B dev Spearman (投影后 cosine), dev 早停。
输出: 预训练投影器 .pt (格式与 02 的 universal_projection 一致,
      供 02_sts_similarity.py --init-from 两阶段微调)。

用法:
  run_with_msvc.bat 07_retrieval_infonce.py --device cuda \
      --retrieval-dir ../cache_python_0.1b_retrieval --cache-dir ../cache_python_0.1b_clean
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent

# 02 开头的模块名无法常规 import, 用 importlib (复用 MlpProj 保证架构一致)
sts2 = importlib.import_module("02_sts_similarity")
MlpProj = sts2.MlpProj
spearman_corr = sts2.spearman_corr


def load_retrieval_pools(retrieval_dir: Path, layer: int, names: list, feature: str) -> dict:
    """加载检索式缓存 → {name: (A, P, N_or_None)} float16

    feature: hidden=仅 hidden; state=states 槽 (低维 PCA 特征, 由 extract --state-pca-from 生成);
             fusion=hidden ⊕ state 拼接
    """
    out = {}
    for name in names:
        p = retrieval_dir / f"sts_pair_l{layer}_{name}.npz"
        if not p.exists():
            print(f"  [skip] {name}: {p} 不存在", flush=True)
            continue
        data = np.load(p)
        hid = data["hiddens"]  # (2N or 3N, C) fp16
        n_pairs = len(data["scores"])
        stride = hid.shape[0] // n_pairs
        assert stride in (2, 3), f"{name}: stride={stride} 异常"
        if feature == "hidden":
            feats = hid
        elif feature == "state":
            feats = data["states"]  # (2N or 3N, D) 低维 PCA 特征
        else:  # fusion
            feats = np.concatenate([hid, data["states"]], axis=1)
        A = np.ascontiguousarray(feats[0::stride])
        P = np.ascontiguousarray(feats[1::stride])
        N = np.ascontiguousarray(feats[2::stride]) if stride == 3 else None
        out[name] = (A, P, N)
        print(f"  {name}: {n_pairs} 对 (stride {stride}), 特征 {feats.shape[1]} 维", flush=True)
    return out


def load_state_pca_transform(path: Path):
    """从投影器 .pt 读取 (head_indices, pca_components, pca_mean, per_head_dim)"""
    ckpt = torch.load(path, map_location="cpu")
    sel = ckpt.get("head_selection")
    if sel is None:
        raise SystemExit(f"错误: {path} 不含 head_selection")
    per = sel["head_size"] * sel["head_size"]
    return (
        list(sel["head_indices"]),
        np.asarray(sel["pca_components"], dtype=np.float32),
        np.asarray(sel["pca_mean"], dtype=np.float32),
        per,
    )


def transform_state_features(states_full: np.ndarray, transform) -> np.ndarray:
    """全量 state (N, 49152) → Top-K head 选择 + PCA → (N, D) float32"""
    head_idx, comps, mean, per = transform
    cols = np.concatenate([np.arange(h * per, (h + 1) * per) for h in head_idx])
    x = states_full[:, cols].astype(np.float32)
    return (x - mean) @ comps.T


def main() -> None:
    parser = argparse.ArgumentParser(description="任务七: 检索式 InfoNCE 预训练")
    parser.add_argument("--retrieval-dir", type=Path, default=Path("../cache_python_0.1b_retrieval"))
    parser.add_argument("--cache-dir", type=Path, default=Path("../cache_python_0.1b_clean"),
                        help="STS-B dev 缓存目录 (评估用)")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--feature", choices=["hidden", "state", "fusion"], default="hidden",
                        help="hidden=hidden 特征 (默认); state=低维 state PCA 特征 "
                             "(缓存需由 extract --state-pca-from 生成); fusion=hidden⊕state 拼接")
    parser.add_argument("--state-pca-from", type=str, default="",
                        help="feature=state/fusion 时必填: 投影器 .pt (提供 Top-K head 选择+PCA, "
                             "用于变换 STS-B dev 的全量 state)")
    parser.add_argument("--datasets", type=str, default="",
                        help="逗号分隔数据集名 (默认自动发现目录下全部)")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="InfoNCE 温度 (e5/bge 常用 0.01-0.05)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=1000, help="每 N 步评估一次 dev")
    parser.add_argument("--patience", type=int, default=3, help="早停: 连续 N 次评估无提升")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-name", type=str, default="pretrained_infonce_l11.pt")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务七: 检索式 InfoNCE 预训练 (in-batch negatives)", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载 STS-B dev (评估用): 按 feature 构建特征
    dev_path = args.cache_dir / f"sts_pair_l{args.layer}_dev.npz"
    dev_data = np.load(dev_path)
    dev_hiddens = dev_data["hiddens"].astype(np.float32)  # (2N, C)
    dev_scores = dev_data["scores"]
    print(f"  STS-B dev: {len(dev_scores)} pairs", flush=True)
    if args.feature == "hidden":
        dev_feats_np = dev_hiddens
    else:
        # state/fusion: 变换 dev 全量 state (clean dev 缓存含 49152 维 states)
        if not args.state_pca_from:
            raise SystemExit("错误: --feature state/fusion 需要 --state-pca-from")
        sp_path = Path(args.state_pca_from)
        if not sp_path.exists():
            sp_path = args.cache_dir.parent / args.state_pca_from
        transform = load_state_pca_transform(sp_path)
        dev_states_pca = transform_state_features(dev_data["states"], transform)
        dev_feats_np = (dev_hiddens if args.feature == "fusion"
                        else dev_states_pca)
        if args.feature == "fusion":
            dev_feats_np = np.concatenate([dev_hiddens, dev_states_pca], axis=1)
        print(f"  dev 特征: {dev_feats_np.shape[1]} 维 ({args.feature})", flush=True)

    # 2. 加载检索式缓存
    if args.datasets:
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    else:
        names = sorted(
            p.stem.split(f"sts_pair_l{args.layer}_")[1]
            for p in args.retrieval_dir.glob(f"sts_pair_l{args.layer}_*.npz")
        )
    print(f"\n-- 加载检索缓存 ({len(names)} 个, feature={args.feature}) --", flush=True)
    pools = load_retrieval_pools(args.retrieval_dir, args.layer, names, args.feature)
    assert pools, "无可用检索缓存"

    # 3. 合并 + shuffle + 补齐缺失负例
    A = np.concatenate([v[0] for v in pools.values()])
    P = np.concatenate([v[1] for v in pools.values()])
    N_parts, has_neg_flags = [], []
    for v in pools.values():
        if v[2] is not None:
            N_parts.append(v[2])
            has_neg_flags.append(np.ones(len(v[2]), dtype=bool))
        else:
            N_parts.append(np.empty((len(v[1]), P.shape[1]), dtype=P.dtype))
            has_neg_flags.append(np.zeros(len(v[1]), dtype=bool))
    N = np.concatenate(N_parts)
    has_neg = np.concatenate(has_neg_flags)
    M = len(A)
    n_trip = int(has_neg.sum())
    print(f"\n合计: {M} 对 (含难负例 {n_trip}), hidden {A.shape[1]} 维", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(M)
    A, P, N, has_neg = A[perm], P[perm], N[perm], has_neg[perm]
    # 缺失负例 → 邻位样本的 positive (随机合法负例)
    fill = np.roll(P, -1, axis=0)
    N[~has_neg] = fill[~has_neg]
    del fill

    # 4. 训练
    torch.manual_seed(args.seed)
    device = args.device
    model = MlpProj(input_dim=A.shape[1], hidden_dim=args.hidden_dim,
                    output_dim=args.output_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    total_steps = args.n_epochs * ((M + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    dev_feats = torch.from_numpy(np.ascontiguousarray(dev_feats_np)).float()

    def eval_dev() -> float:
        model.eval()
        with torch.no_grad():
            emb = model(dev_feats.to(device)).cpu().numpy()
            e1, e2 = emb[0::2], emb[1::2]
            cos = (e1 * e2).sum(axis=1)
        return spearman_corr(cos, dev_scores)

    print(f"\n-- 训练 (batch {args.batch_size}, τ={args.temperature}, "
          f"~{total_steps} steps, 每 {args.eval_every} 步评估 dev) --", flush=True)
    best_dev, best_state, bad, step = -1.0, None, 0, 0
    t0 = time.time()
    stop = False
    for epoch in range(args.n_epochs):
        if stop:
            break
        order = torch.randperm(M)
        model.train()
        for i in range(0, M, args.batch_size):
            idx = order[i:i + args.batch_size].numpy()
            a = torch.from_numpy(np.ascontiguousarray(A[idx])).float().to(device)
            p = torch.from_numpy(np.ascontiguousarray(P[idx])).float().to(device)
            n = torch.from_numpy(np.ascontiguousarray(N[idx])).float().to(device)

            # InfoNCE: 候选 = [in-batch positives; in-batch negatives], label = 对角线
            ea = model(a)
            ep = model(p)
            en = model(n)
            sim = torch.cat([ea @ ep.t(), ea @ en.t()], dim=1) / args.temperature
            labels = torch.arange(a.size(0), device=device)
            loss = F.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.eval_every == 0 or step == total_steps:
                dev_sp = eval_dev()
                mark = ""
                if dev_sp > best_dev:
                    best_dev = dev_sp
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad = 0
                    mark = " *"
                else:
                    bad += 1
                    mark = f" (bad {bad}/{args.patience})"
                rate = step * args.batch_size / max(time.time() - t0, 1e-6)
                print(f"  step {step}/{total_steps} loss={loss.item():.4f} "
                      f"dev={dev_sp:.4f}{mark} ({rate:.0f} pairs/s)", flush=True)
                model.train()
                if bad >= args.patience:
                    print(f"  早停 (连续 {args.patience} 次无提升)", flush=True)
                    stop = True
                    break

    # 5. 保存 (格式与 02 的 universal_projection 一致)
    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_path = args.retrieval_dir / args.save_name
    save_dict = {
        "seeds": [args.seed],
        "state_dicts": [best_state],
        "config": {
            "input_dim": A.shape[1],
            "feature": args.feature,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "loss": "infonce_inbatch",
            "train_pairs": M,
            "best_dev_spearman": best_dev,
        },
    }
    torch.save(save_dict, save_path)
    print(f"\n{'='*60}", flush=True)
    print(f"最优 dev Spearman: {best_dev:.4f}", flush=True)
    print(f"保存预训练投影器: {save_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
