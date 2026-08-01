"""
任务三：任务分类 - Top-K Head + PCA + MLP

方法：
1. Head 筛选：评估每个 head 单独的分类准确率，选 Top-8 head
2. 特征拼接：Top-8 head 的 WKV state 拼接
3. PCA 降维：→ 256 维（MLP 能学习反旋转）
4. MLP 分类：256 → 256 → num_classes，交叉熵损失

数据集：golden_balanced.jsonl (任务难度 R0-R3 四类)
评估：test_acc (15% held-out test, dev 用于 head 选择和 early stopping)

特征提取:
    run_with_msvc.bat extract_features.py --task classification
    生成 cache_python/classification_l{LAYER}.npz

运行:
    cd paper/scripts
    uv run --project ../../scripts python 03_classification.py
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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from cache import load_npz  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)

NUM_HEADS = 16
HEAD_SIZE = 64
LAYER = 12


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MlpClassifier(nn.Module):
    """MLP 分类器: input → Linear → GELU → LayerNorm → Dropout → Linear"""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(X_train, y_train, X_dev, y_dev, X_test, y_test, input_dim=256, n_epochs=30, batch_size=256, lr=1e-3, device="cpu"):
    """训练分类器: dev 选 best_state, test 最终评估 (held-out)"""
    torch.manual_seed(42)
    np.random.seed(42)

    model = MlpClassifier(input_dim=input_dim, hidden_dim=256, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_dev_t = torch.from_numpy(X_dev).float().to(device)
    y_dev_t = torch.from_numpy(y_dev).long().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    best_dev_acc = 0.0
    best_state = None
    n_train = len(y_train)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(X_train_t[idx])
            loss = F.cross_entropy(logits, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_dev = model(X_dev_t).argmax(dim=-1)
            dev_acc = (pred_dev == y_dev_t).float().mean().item()
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 用 best_state (按 dev 选择) 在 test 上评估
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_t).argmax(dim=-1)
        train_acc = (pred_train == y_train_t).float().mean().item()
        pred_test = model(X_test_t).argmax(dim=-1)
        test_acc = (pred_test == y_test_t).float().mean().item()
    return best_dev_acc, train_acc, test_acc


def extract_head(states, head_idx, head_size=HEAD_SIZE):
    per_head = head_size * head_size
    return states[:, head_idx * per_head : (head_idx + 1) * per_head]


def extract_multi_head(states, head_indices, head_size=HEAD_SIZE):
    return np.concatenate([extract_head(states, h, head_size) for h in head_indices], axis=1)


def main():
    parser = argparse.ArgumentParser(description="任务三: 任务分类")
    parser.add_argument("--cache", type=Path, default=Path("../cache_python/classification_l12.npz"))
    parser.add_argument("--data", type=Path, default=Path("../data/golden_balanced.jsonl"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("任务三: 任务分类 - Top-K Head + PCA + MLP", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载数据
    records = read_jsonl(args.data)
    labels = np.array([r["tier"] for r in records], dtype=np.int64)
    print(f"  总样本: {len(records)}", flush=True)
    print(f"  类别分布: {dict(zip(*np.unique(labels, return_counts=True)))}", flush=True)

    # 2. 加载 state 缓存
    states, hiddens = load_npz(args.cache)
    print(f"  states: {states.shape}", flush=True)
    if states.shape[0] != len(labels):
        n = min(states.shape[0], len(labels))
        states, hiddens, labels = states[:n], hiddens[:n], labels[:n]
        print(f"  截取前 {n} 个样本", flush=True)

    # 3. 划分 train/dev/test (70/15/15, stratified)
    #    dev: head 选择 + early stopping
    #    test: 最终评估 (held-out, 不参与任何选择)
    X_train_idx, X_temp_idx = train_test_split(np.arange(len(labels)), test_size=0.30, random_state=42, stratify=labels)
    X_dev_idx, X_test_idx = train_test_split(X_temp_idx, test_size=0.50, random_state=42, stratify=labels[X_temp_idx])
    states_train, states_dev, states_test = states[X_train_idx], states[X_dev_idx], states[X_test_idx]
    hiddens_train, hiddens_dev, hiddens_test = hiddens[X_train_idx], hiddens[X_dev_idx], hiddens[X_test_idx]
    labels_train, labels_dev, labels_test = labels[X_train_idx], labels[X_dev_idx], labels[X_test_idx]
    print(f"  划分: train={len(X_train_idx)}, dev={len(X_dev_idx)} (head选择+early stopping), test={len(X_test_idx)} (最终评估)", flush=True)

    # 4. Hidden baseline (dev 选 best_state, test 最终评估)
    print(f"\n-- Hidden baseline --", flush=True)
    mean = hiddens_train.mean(axis=0, keepdims=True)
    std = hiddens_train.std(axis=0, keepdims=True) + 1e-6
    h_train = (hiddens_train - mean) / std
    h_dev = (hiddens_dev - mean) / std
    h_test = (hiddens_test - mean) / std
    dev_acc_h, train_acc_h, test_acc_h = train_classifier(h_train, labels_train, h_dev, labels_dev, h_test, labels_test, input_dim=h_train.shape[1], n_epochs=args.n_epochs, device=args.device)
    print(f"  Hidden MLP: train={train_acc_h:.4f} dev={dev_acc_h:.4f} test={test_acc_h:.4f}", flush=True)

    # 5. Head 筛选 (只用 dev 排序, 不接触 test)
    print(f"\n-- 每个 head 的分类准确率 (用 dev 排序) --", flush=True)
    head_accs = []
    for h in range(NUM_HEADS):
        X_h_train = extract_head(states_train, h)
        X_h_dev = extract_head(states_dev, h)
        pca = PCA(n_components=min(64, X_h_train.shape[0], X_h_train.shape[1]), random_state=42)
        X_h_train_pca = pca.fit_transform(X_h_train)
        X_h_dev_pca = pca.transform(X_h_dev)
        m = X_h_train_pca.mean(axis=0, keepdims=True)
        s = X_h_train_pca.std(axis=0, keepdims=True) + 1e-6
        X_h_train_pca = (X_h_train_pca - m) / s
        X_h_dev_pca = (X_h_dev_pca - m) / s
        dev_acc, _, _ = train_classifier(X_h_train_pca, labels_train, X_h_dev_pca, labels_dev, X_h_dev_pca, labels_dev, input_dim=X_h_train_pca.shape[1], n_epochs=10, device=args.device)
        head_accs.append((h, dev_acc))
        print(f"  H{h:2d}: dev_acc={dev_acc:.4f}", flush=True)

    head_accs.sort(key=lambda x: x[1], reverse=True)
    top_heads = [h for h, _ in head_accs[: args.top_k]]
    print(f"\n  Top-{args.top_k} head (按 dev 选择): {top_heads}", flush=True)

    # 6. Top-K head + PCA + MLP (dev 选 best_state, test 最终评估)
    print(f"\n-- Top-{args.top_k} head + PCA{args.pca_dim} + MLP --", flush=True)
    X_train_raw = extract_multi_head(states_train, top_heads)
    X_dev_raw = extract_multi_head(states_dev, top_heads)
    X_test_raw = extract_multi_head(states_test, top_heads)
    pca = PCA(n_components=args.pca_dim, random_state=42)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_dev_pca = pca.transform(X_dev_raw)
    X_test_pca = pca.transform(X_test_raw)
    mean = X_train_pca.mean(axis=0, keepdims=True)
    std = X_train_pca.std(axis=0, keepdims=True) + 1e-6
    X_train_pca = (X_train_pca - mean) / std
    X_dev_pca = (X_dev_pca - mean) / std
    X_test_pca = (X_test_pca - mean) / std

    t0 = time.time()
    dev_acc, train_acc, test_acc = train_classifier(X_train_pca, labels_train, X_dev_pca, labels_dev, X_test_pca, labels_test, input_dim=args.pca_dim, n_epochs=args.n_epochs, device=args.device)
    print(f"  train={train_acc:.4f} dev={dev_acc:.4f} test={test_acc:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # 7. 结论
    print(f"\n{'='*60}", flush=True)
    print(f"结论:", flush=True)
    print(f"  Hidden MLP:               dev={dev_acc_h:.4f} test={test_acc_h:.4f}", flush=True)
    print(f"  Top-{args.top_k} head + PCA + MLP: dev={dev_acc:.4f} test={test_acc:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
