#!/usr/bin/env python3
"""20 Newsgroups 严格去重 + 分层 train/dev/test split

去重策略（防止数据泄露）:
  1. 文本归一化（strip + lower + 折叠空白）
  2. 文本哈希去重（同一文档跨 newsgroup 交叉post 只保留第一次出现）
  3. 按类别分层 70/15/15 切分 train/dev/test
  4. train 用于训练监督对比 projector（看标签）
  5. dev 用于 early stopping
  6. test 用于最终评估（held-out，不参与训练）

输入: data/clustering/twentynewsgroups.jsonl (59545 samples, MTEB 官方版本)
输出: data/clustering_20ng_split/{train,dev,test}.jsonl

用法: uv run --project ../../scripts python split_20ng.py
"""

import json
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


def text_hash(text: str) -> str:
    """文本归一化后哈希，用于跨类别去重"""
    norm = " ".join(text.strip().lower().split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def main():
    src = Path(__file__).resolve().parent.parent.parent / "data" / "clustering" / "twentynewsgroups.jsonl"
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "clustering_20ng_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载
    print(f"加载: {src}", flush=True)
    records = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  原始样本: {len(records)}", flush=True)
    print(f"  类别分布: {Counter(r['label'] for r in records)}", flush=True)

    # 2. 文本去重（跨 newsgroup 交叉 post）
    seen = set()
    deduped = []
    removed = 0
    for r in records:
        h = text_hash(r["text"])
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        deduped.append(r)
    print(f"\n去重: {len(records)} -> {len(deduped)} (移除 {removed} 重复文档, {removed/len(records)*100:.2f}%)", flush=True)
    print(f"  去重后类别分布: {Counter(r['label'] for r in deduped)}", flush=True)

    # 3. 分层 70/15/15 split
    labels = np.array([r["label"] for r in deduped])
    indices = np.arange(len(deduped))

    # 先分 70% train 和 30% temp
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=labels)
    # 再把 temp 分成 dev/test 各 15%
    dev_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=labels[temp_idx])

    print(f"\n分层 split (70/15/15):", flush=True)
    print(f"  train: {len(train_idx)} ({len(train_idx)/len(deduped)*100:.1f}%)", flush=True)
    print(f"  dev:   {len(dev_idx)} ({len(dev_idx)/len(deduped)*100:.1f}%)", flush=True)
    print(f"  test:  {len(test_idx)} ({len(test_idx)/len(deduped)*100:.1f}%)", flush=True)

    # 验证 split 的类别分布一致性
    for name, idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        dist = Counter(labels[idx].tolist())
        print(f"  {name} 类别分布: {dict(sorted(dist.items()))}", flush=True)

    # 4. 保存
    for name, idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        out = out_dir / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for i in idx:
                r = deduped[i]
                f.write(json.dumps({
                    "_id": r["_id"],
                    "text": r["text"],
                    "label": r["label"],
                }, ensure_ascii=False) + "\n")
        print(f"  保存 {len(idx)} 条 -> {out}", flush=True)

    # 5. 去重报告
    report = {
        "original": len(records),
        "after_dedup": len(deduped),
        "removed_duplicates": removed,
        "removal_rate": f"{removed/len(records)*100:.2f}%",
        "split": {
            "train": len(train_idx),
            "dev": len(dev_idx),
            "test": len(test_idx),
        },
        "split_ratios": "70/15/15",
        "random_state": 42,
        "stratified": True,
    }
    report_path = out_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n去重+split 报告: {report_path}", flush=True)
    print(f"输出目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
