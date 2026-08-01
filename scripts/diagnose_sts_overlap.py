#!/usr/bin/env python3
"""诊断 STS 训练数据与 STS-B dev/test 的重叠情况

检查三种重叠：
  1. exact pair: (s1, s2) 完全相同（顺序一致）
  2. reverse pair: (s2, s1) 反向出现
  3. 单句级: s1 或 s2 单独出现在 STS-B test/dev 的句子集合中

用法：uv run --project ../../scripts python diagnose_sts_overlap.py
"""

import json
from pathlib import Path
from collections import defaultdict


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def norm(s: str) -> str:
    """归一化：strip + lower，便于跨数据集匹配"""
    return s.strip().lower()


def check_overlap(train_records, eval_records, train_name, eval_name):
    """检查 train 与 eval 的三种重叠"""
    # eval 的 pair 集合和句子集合
    eval_pairs_fwd = set()  # (s1, s2)
    eval_pairs_rev = set()  # (s2, s1)
    eval_sentences = set()  # 所有单独的句子
    for r in eval_records:
        s1 = norm(r["sentence1"])
        s2 = norm(r["sentence2"])
        eval_pairs_fwd.add((s1, s2))
        eval_pairs_rev.add((s2, s1))
        eval_sentences.add(s1)
        eval_sentences.add(s2)

    exact_count = 0
    reverse_count = 0
    sentence_count = 0  # 训练 pair 中至少有一句出现在 eval 句子集合
    overlap_examples = []

    for r in train_records:
        s1 = norm(r["sentence1"])
        s2 = norm(r["sentence2"])
        pair_fwd = (s1, s2)

        if pair_fwd in eval_pairs_fwd:
            exact_count += 1
            if len(overlap_examples) < 5:
                overlap_examples.append(("exact", s1[:80], s2[:80]))
        elif pair_fwd in eval_pairs_rev:
            reverse_count += 1
            if len(overlap_examples) < 5:
                overlap_examples.append(("reverse", s1[:80], s2[:80]))

        # 单句级：s1 或 s2 在 eval 句子集合中
        if s1 in eval_sentences or s2 in eval_sentences:
            sentence_count += 1

    total = len(train_records)
    print(f"\n[{train_name}] vs [{eval_name}] ({total} train pairs, {len(eval_records)} eval pairs):")
    print(f"  exact pair 重叠:   {exact_count:6d} ({exact_count / total * 100:.2f}%)")
    print(f"  reverse pair 重叠: {reverse_count:6d} ({reverse_count / total * 100:.2f}%)")
    print(f"  单句级重叠:        {sentence_count:6d} ({sentence_count / total * 100:.2f}%)")
    if overlap_examples:
        print(f"  示例:")
        for kind, s1, s2 in overlap_examples:
            print(f"    [{kind}] {s1} | {s2}")

    return {
        "train": train_name,
        "eval": eval_name,
        "train_total": total,
        "eval_total": len(eval_records),
        "exact": exact_count,
        "reverse": reverse_count,
        "sentence": sentence_count,
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sts"
    print(f"数据目录: {data_dir}\n")

    # 加载 eval 数据
    sts_dev = read_jsonl(data_dir / "sts_dev.jsonl")
    sts_test = read_jsonl(data_dir / "sts_test.jsonl")
    print(f"STS-B dev:  {len(sts_dev)} pairs")
    print(f"STS-B test: {len(sts_test)} pairs")

    # 加载训练数据
    train_sets = {
        "sts_train (STS-B train)": read_jsonl(data_dir / "sts_train.jsonl"),
        "nli_train": read_jsonl(data_dir / "nli_train.jsonl"),
        "extra_train (STS12-16+SICK-R)": read_jsonl(data_dir / "extra_train.jsonl"),
        "sickr": read_jsonl(data_dir / "sickr.jsonl"),
    }

    # 单独的 STS12-16
    for year in [12, 13, 14, 15, 16]:
        recs = read_jsonl(data_dir / f"sts{year}.jsonl")
        if recs:
            train_sets[f"sts{year}"] = recs

    print("\n训练数据集:")
    for name, recs in train_sets.items():
        print(f"  {name}: {len(recs)} pairs")

    # 检查每个训练集与 dev/test 的重叠
    print("\n" + "=" * 80)
    print("重叠诊断")
    print("=" * 80)

    all_results = []
    for train_name, train_recs in train_sets.items():
        if not train_recs:
            continue
        for eval_name, eval_recs in [("STS-B dev", sts_dev), ("STS-B test", sts_test)]:
            r = check_overlap(train_recs, eval_recs, train_name, eval_name)
            all_results.append(r)

    # 汇总表
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"{'训练集':<35} {'eval':<12} {'exact':>7} {'reverse':>8} {'sentence':>9}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['train']:<35} {r['eval']:<12} {r['exact']:>7} {r['reverse']:>8} {r['sentence']:>9}")

    # 结论
    print("\n结论:")
    total_exact_test = sum(r["exact"] + r["reverse"] for r in all_results if r["eval"] == "STS-B test")
    total_sentence_test = sum(r["sentence"] for r in all_results if r["eval"] == "STS-B test")
    print(f"  与 STS-B test 的 pair 级总重叠 (exact+reverse): {total_exact_test}")
    print(f"  与 STS-B test 的单句级重叠 pair 数: {total_sentence_test}")
    if total_exact_test > 0:
        print(f"  ⚠ 存在 pair 级泄露！STS-B test 结果可能虚高")
    if total_sentence_test > 0:
        print(f"  ⚠ 存在单句级泄露（部分重叠）")


if __name__ == "__main__":
    main()
