#!/usr/bin/env python3
"""20 Newsgroups (sklearn 全文版) 严格去重 + 分层 70/15/15 split

论文 §4.3.2 规范:
  数据源: sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
  去重: 文本归一化哈希去重 (跨组交叉post 只保留第一次出现)
  Split: 分层 70/15/15 train/dev/test
    - train: 训练监督对比 projector (看标签)
    - dev: early stopping + best_state 选择
    - test: held-out 最终评估 (不参与训练)

输出: data/clustering_sklearn_20ng/{train,dev,test}.jsonl

用法: uv run --project ../../scripts python split_sklearn_20ng.py
"""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def text_hash(text: str) -> str:
    """文本归一化后哈希，用于跨类别去重"""
    norm = " ".join(text.strip().lower().split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent.parent / "data" / "clustering_sklearn_20ng"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 sklearn 全文版 (去 headers/footers/quotes)
    print("加载 sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))...", flush=True)
    bunch = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = list(bunch.data)
    labels = bunch.target.astype(np.int32)
    target_names = bunch.target_names
    print(f"  原始样本: {len(texts)}", flush=True)
    print(f"  类别数: {len(target_names)}", flush=True)
    print(f"  类别: {target_names}", flush=True)

    # 2. 过滤空文档 (去 headers/footers/quotes 后可能只剩空内容)
    valid_mask = np.array([len(t.strip()) > 0 for t in texts])
    texts = [t for t, v in zip(texts, valid_mask) if v]
    labels = labels[valid_mask]
    print(f"  过滤空文档后: {len(texts)} (移除 {int((~valid_mask).sum())})", flush=True)

    # 3. 文本归一化哈希去重 (跨组交叉post)
    seen = set()
    keep_idx = []
    for i, t in enumerate(texts):
        h = text_hash(t)
        if h not in seen:
            seen.add(h)
            keep_idx.append(i)
    texts = [texts[i] for i in keep_idx]
    labels = labels[keep_idx]
    print(f"  哈希去重后: {len(texts)} (移除 {len(valid_mask) - len(keep_idx) if valid_mask.all() else 0})", flush=True)
    print(f"  最终样本: {len(texts)}", flush=True)

    # 4. 分层 70/15/15 split
    indices = np.arange(len(texts))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=labels)
    dev_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=labels[temp_idx])

    print(f"\n分层 split (70/15/15):", flush=True)
    for name, idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        dist = Counter(labels[idx].tolist())
        print(f"  {name}: {len(idx)} ({len(idx)/len(texts)*100:.1f}%)", flush=True)

    # 5. 保存为 jsonl
    import json
    for name, idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        out = out_dir / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps({"text": texts[i], "label": int(labels[i])}, ensure_ascii=False) + "\n")
        print(f"  保存 {len(idx)} 条 -> {out}", flush=True)

    # 6. 报告
    import json
    report = {
        "source": "sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))",
        "original": int(len(bunch.data)),
        "after_filter_empty": int(valid_mask.sum()),
        "after_dedup": len(texts),
        "split": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "split_ratios": "70/15/15",
        "random_state": 42,
        "stratified": True,
        "dedup_method": "text normalization + md5 hash (cross-group crosspost removal)",
        "target_names": list(target_names),
    }
    report_path = out_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {report_path}", flush=True)
    print(f"输出目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
