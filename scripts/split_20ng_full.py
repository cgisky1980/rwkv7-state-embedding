#!/usr/bin/env python3
"""20 Newsgroups (sklearn 原始全文版) 严格去重 + 分层 train/dev/test split

数据源: sklearn.datasets.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
  - 去除 headers/footers/quotes 防止模型记忆元信息
  - 全文内容 (vs MTEB 版本的标题级短文本)
  - ~18k 样本

去重策略（防止数据泄露）:
  1. 空文档过滤 (去 headers/footers 后可能为空)
  2. 文本归一化哈希去重 (排除同一文档跨组交叉post)
  3. 按类别分层 70/15/15 切分 train/dev/test
  4. train 用于训练监督对比 projector (看标签)
  5. dev 用于 early stopping
  6. test 用于最终评估 (held-out)

输出: data/clustering_20ng_full/{train,dev,test}.jsonl

用法: uv run --project ../../scripts python split_20ng_full.py
"""

import json
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def text_hash(text: str) -> str:
    """文本归一化后哈希，用于跨类别去重"""
    norm = " ".join(text.strip().lower().split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def main():
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "clustering_20ng_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下载 sklearn 原始全文版 (去 headers/footers/quotes)
    print("下载 20 Newsgroups (sklearn, 去除 headers/footers/quotes)...", flush=True)
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = data.data
    labels = data.target.astype(np.int32)
    target_names = data.target_names
    print(f"  原始样本: {len(texts)}", flush=True)
    print(f"  类别: {target_names}", flush=True)
    print(f"  类别分布: {Counter(labels.tolist())}", flush=True)

    # 2. 过滤空文档 (去 headers/footers 后可能为空)
    non_empty = []
    for i, t in enumerate(texts):
        if t.strip():
            non_empty.append(i)
    print(f"\n过滤空文档: {len(texts)} -> {len(non_empty)} (移除 {len(texts)-len(non_empty)})", flush=True)
    texts = [texts[i] for i in non_empty]
    labels = labels[non_empty]

    # 3. 文本去重 (跨 newsgroup 交叉 post)
    seen = set()
    deduped_idx = []
    removed = 0
    for i, t in enumerate(texts):
        h = text_hash(t)
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        deduped_idx.append(i)
    print(f"去重: {len(texts)} -> {len(deduped_idx)} (移除 {removed} 重复文档, {removed/len(texts)*100:.2f}%)", flush=True)
    texts = [texts[i] for i in deduped_idx]
    labels = labels[deduped_idx]
    print(f"  去重后类别分布: {Counter(labels.tolist())}", flush=True)

    # 4. 分层 70/15/15 split
    indices = np.arange(len(texts))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=labels)
    dev_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=labels[temp_idx])

    print(f"\n分层 split (70/15/15):", flush=True)
    for name, idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        dist = Counter(labels[idx].tolist())
        print(f"  {name}: {len(idx)} ({len(idx)/len(texts)*100:.1f}%) 类别分布: {dict(sorted(dist.items()))}", flush=True)

    # 5. 保存
    for name, idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        out = out_dir / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps({
                    "text": texts[i],
                    "label": int(labels[i]),
                    "label_name": target_names[int(labels[i])],
                }, ensure_ascii=False) + "\n")
        print(f"  保存 {len(idx)} 条 -> {out}", flush=True)

    # 6. 报告
    report = {
        "source": "sklearn.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))",
        "original": len(data.data),
        "after_empty_filter": len(non_empty),
        "after_dedup": len(texts),
        "removed_empty": len(data.data) - len(non_empty),
        "removed_duplicates": removed,
        "split": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "split_ratios": "70/15/15",
        "random_state": 42,
        "stratified": True,
        "target_names": target_names,
    }
    report_path = out_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {report_path}", flush=True)
    print(f"输出目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
