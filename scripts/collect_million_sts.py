#!/usr/bin/env python3
"""百万对级 STS 训练数据收集 + 泄露过滤

目标: 将训练数据从 ~15k 对扩充到百万对级别(合法来源, 排除 MTEB 评估集).

数据源(与 scripts/download_nli_data.py 同源同映射, 全量不做采样):
  - SNLI    (stanfordnlp/snli)     train 550,152 对
  - MultiNLI(nyu-mll/multi_nli)    train 392,702 对
  - 加已有 clean 14,984 对(STS-B train + 旧 NLI 采样) ≈ 95.8 万对

NLI 标签 → 分数映射(沿用既有 nli_train.jsonl 口径):
  entailment=5.0, neutral=3.0, contradiction=2.0

两阶段(小步快跑, 分别运行):
  --stage collect  下载全量 → 转分数 → 过滤无标签/空句 → 对级去重 → *_full.jsonl
  --stage filter   构建 MTEB eng STS eval 句子 blocklist → 过滤 → *_clean.jsonl

输出目录: paper/data/sts_million/
  snli_train_full.jsonl / multinli_train_full.jsonl    (collect 阶段)
  snli_train_clean.jsonl / multinli_train_clean.jsonl  (filter 阶段)
  report.json                                          (两阶段统计)

注意: 新全量数据与旧 data/sts_dedup/nli_train.jsonl(即 SNLI/MultiNLI 前 5k 采样)
天然重叠, 训练时应以本脚本全量版替代旧 nli_train, 不要同时使用.

用法:
  uv run --project ../../scripts python collect_million_sts.py --stage collect
  uv run --project ../../scripts python collect_million_sts.py --stage filter
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
OUT_DIR = PAPER_DIR / "data" / "sts_million"
REPORT_PATH = OUT_DIR / "report.json"

# 与 download_nli_data.py 修正映射一致
LABEL_TO_SCORE = {
    0: 5.0,  # entailment → 高语义相似
    1: 3.0,  # neutral → 中等
    2: 2.0,  # contradiction → 低但非零(共享词汇)
}

DATASETS = [
    # (hf repo, 输出文件名前缀)
    ("stanfordnlp/snli", "snli_train"),
    ("nyu-mll/multi_nli", "multinli_train"),
]


def norm(s: str) -> str:
    return s.strip().lower()


def read_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  保存 {len(records)} 条 -> {path}", flush=True)


def load_report() -> dict:
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- collect

def stage_collect() -> None:
    """全量下载 SNLI + MultiNLI, 转分数, 去重, 保存 *_full.jsonl"""
    from datasets import load_dataset

    print("=" * 60)
    print("阶段 1: 全量下载 + 转换 + 去重")
    print("=" * 60, flush=True)

    report = load_report()
    report["collect"] = {}

    # 跨数据集去重(句子对级, 双向)
    seen_pairs = set()
    total = 0

    for repo, name in DATASETS:
        print(f"\n下载 {repo} (train split, 全量)...", flush=True)
        ds = load_dataset(repo, split="train")
        print(f"  原始: {len(ds)} 对", flush=True)

        records = []
        n_no_label = 0
        n_empty = 0
        n_dup = 0
        for item in ds:
            label = item.get("label", -1)
            if label == -1 or label not in LABEL_TO_SCORE:
                n_no_label += 1
                continue
            s1 = item.get("premise", "").strip()
            s2 = item.get("hypothesis", "").strip()
            if not s1 or not s2:
                n_empty += 1
                continue
            key = (norm(s1), norm(s2))
            rkey = (key[1], key[0])
            if key in seen_pairs or rkey in seen_pairs:
                n_dup += 1
                continue
            seen_pairs.add(key)
            records.append({"sentence1": s1, "sentence2": s2,
                            "score": LABEL_TO_SCORE[label]})

        score_dist = Counter(r["score"] for r in records)
        print(f"  无标签: {n_no_label}, 空句: {n_empty}, 对级重复: {n_dup}")
        print(f"  转换后: {len(records)} 对, 分数分布: {dict(sorted(score_dist.items()))}", flush=True)

        save_jsonl(OUT_DIR / f"{name}_full.jsonl", records)
        report["collect"][name] = {
            "raw": len(ds), "kept": len(records),
            "no_label": n_no_label, "empty": n_empty, "dup": n_dup,
            "score_dist": {str(k): v for k, v in sorted(score_dist.items())},
        }
        total += len(records)

    report["collect"]["total"] = total
    save_report(report)
    print(f"\n合计收集: {total} 对", flush=True)


# ---------------------------------------------------------------- filter

def build_blocklist() -> set:
    """构建 MTEB eng STS 全部任务 eval split 句子 blocklist(句子级精确匹配)"""
    import mteb

    print("构建 MTEB eng STS eval blocklist...", flush=True)
    tasks = mteb.get_tasks(task_types=["STS"], languages=["eng"])
    print(f"  STS 任务数: {len(tasks)}", flush=True)

    block = set()
    for t in tasks:
        try:
            t.load_data()
        except Exception as e:
            print(f"  [WARN] {t.metadata.name} 加载失败: {e}", flush=True)
            continue
        for split in t.metadata.eval_splits:
            if split not in t.dataset:
                continue
            for ex in t.dataset[split]:
                for v in ex.values():
                    if isinstance(v, str) and v.strip():
                        block.add(norm(v))
    print(f"  blocklist 句子数: {len(block)}", flush=True)
    return block


def stage_filter() -> None:
    """用 MTEB blocklist 过滤 *_full.jsonl → *_clean.jsonl"""
    print("=" * 60)
    print("阶段 2: 泄露过滤 (MTEB eval 句子级)")
    print("=" * 60, flush=True)

    block = build_blocklist()

    report = load_report()
    report["filter"] = {"blocklist_size": len(block)}

    total = 0
    for _, name in DATASETS:
        src = OUT_DIR / f"{name}_full.jsonl"
        if not src.exists():
            print(f"[skip] {src} 不存在, 请先运行 --stage collect", flush=True)
            continue
        records = read_jsonl(src)
        kept = []
        removed = 0
        for r in records:
            if norm(r["sentence1"]) in block or norm(r["sentence2"]) in block:
                removed += 1
                continue
            kept.append(r)
        print(f"  {name}: {len(records)} -> {len(kept)} (剔除泄露 {removed})", flush=True)
        save_jsonl(OUT_DIR / f"{name}_clean.jsonl", kept)
        report["filter"][name] = {"raw": len(records), "kept": len(kept), "removed": removed}
        total += len(kept)

    report["filter"]["total"] = total
    # 加上已有 clean 基数
    old_clean = 14984
    report["filter"]["total_with_existing_clean"] = total + old_clean
    save_report(report)
    print(f"\n过滤后新增: {total} 对; 加已有 clean {old_clean} 对 = {total + old_clean} 对", flush=True)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="百万对 STS 训练数据收集")
    parser.add_argument("--stage", choices=["collect", "filter"], required=True)
    args = parser.parse_args()

    if args.stage == "collect":
        stage_collect()
    else:
        stage_filter()

    return 0


if __name__ == "__main__":
    sys.exit(main())
