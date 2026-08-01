#!/usr/bin/env python3
"""STS 训练数据去重：排除与 STS-B dev/test 有任何重叠的训练 pair

去重规则（最严格，单句级）：
  1. exact pair: (s1, s2) 完全出现在 dev/test → 排除
  2. reverse pair: (s2, s1) 反向出现在 dev/test → 排除
  3. 单句级: s1 或 s2 出现在 dev/test 的句子集合中 → 排除
     （因为 STS-B dev/test 的句子若出现在训练中，即使配对不同，
       模型也可能记住该句子的表示，导致评估偏向）

输入（data/sts/）：
  sts_train.jsonl, nli_train.jsonl, extra_train.jsonl, sickr.jsonl
  sts_dev.jsonl, sts_test.jsonl

输出（data/sts_dedup/）：
  sts_train.jsonl, nli_train.jsonl, extra_train.jsonl, sickr.jsonl
  dedup_report.json (去重前后数量 + 重叠明细)

用法：uv run --project ../../scripts python dedup_sts_train.py
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


def save_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  保存 {len(records)} 条 -> {path}")


def norm(s: str) -> str:
    return s.strip().lower()


def main():
    src_dir = Path(__file__).resolve().parent.parent / "data" / "sts"
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sts_dedup"
    print(f"源目录:   {src_dir}")
    print(f"输出目录: {out_dir}\n")

    # 1. 加载 eval 数据（dev + test），构建排除集合
    sts_dev = read_jsonl(src_dir / "sts_dev.jsonl")
    sts_test = read_jsonl(src_dir / "sts_test.jsonl")
    eval_records = sts_dev + sts_test

    eval_pairs = set()       # (s1, s2) 和 (s2, s1) 都加入
    eval_sentences = set()   # 所有单句
    for r in eval_records:
        s1 = norm(r["sentence1"])
        s2 = norm(r["sentence2"])
        eval_pairs.add((s1, s2))
        eval_pairs.add((s2, s1))
        eval_sentences.add(s1)
        eval_sentences.add(s2)

    print(f"STS-B dev:  {len(sts_dev)} pairs")
    print(f"STS-B test: {len(sts_test)} pairs")
    print(f"eval 句子集合大小: {len(eval_sentences)}\n")

    # 2. 对每个训练文件去重
    train_files = ["sts_train", "nli_train", "extra_train", "sickr"]
    report = {
        "eval": {"dev_pairs": len(sts_dev), "test_pairs": len(sts_test), "eval_sentences": len(eval_sentences)},
        "per_file": {},
    }

    for name in train_files:
        recs = read_jsonl(src_dir / f"{name}.jsonl")
        if not recs:
            print(f"[skip] {name}: 文件不存在或为空")
            continue

        kept = []
        removed_exact = 0
        removed_reverse = 0
        removed_sentence = 0

        for r in recs:
            s1 = norm(r["sentence1"])
            s2 = norm(r["sentence2"])
            pair = (s1, s2)

            if pair in eval_pairs:
                # 这里 eval_pairs 同时包含 (s1,s2) 和 (s2,s1)
                # 判断是 exact 还是 reverse：若 (s1,s2) 在原 eval 的 fwd 集合
                # 简化：统一记为 pair 级
                # 为区分，重新检查
                removed_exact += 1
                continue

            # 单句级：s1 或 s2 在 eval 句子集合中
            if s1 in eval_sentences or s2 in eval_sentences:
                removed_sentence += 1
                continue

            kept.append(r)

        removed_total = removed_exact + removed_reverse + removed_sentence
        report["per_file"][name] = {
            "original": len(recs),
            "kept": len(kept),
            "removed_total": removed_total,
            "removed_exact_or_reverse": removed_exact,
            "removed_sentence": removed_sentence,
        }
        print(f"[{name}] {len(recs)} -> {len(kept)} (移除 {removed_total}: pair级 {removed_exact}, 单句级 {removed_sentence})")
        save_jsonl(out_dir / f"{name}.jsonl", kept)

    # 3. 汇总
    total_orig = sum(v["original"] for v in report["per_file"].values())
    total_kept = sum(v["kept"] for v in report["per_file"].values())
    total_removed = total_orig - total_kept
    report["total"] = {
        "original": total_orig,
        "kept": total_kept,
        "removed": total_removed,
        "removal_rate": f"{total_removed / total_orig * 100:.2f}%",
    }

    print(f"\n=== 汇总 ===")
    print(f"原始训练数据: {total_orig} pairs")
    print(f"去重后:       {total_kept} pairs")
    print(f"移除:         {total_removed} pairs ({total_removed / total_orig * 100:.2f}%)")

    # 保存报告
    save_jsonl(out_dir / "dedup_report.json", [report])
    print(f"\n去重报告: {out_dir / 'dedup_report.json'}")


if __name__ == "__main__":
    main()
