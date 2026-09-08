#!/usr/bin/env python3
"""检索式 STS 训练数据收集 + 泄露过滤 (bge/e5 同款范式)

数据源: sentence-transformers/embedding-training-data (HF Hub)
  格式: jsonl.gz, 每行一个 JSON 对象:
    - Pairs:    ["text1", "text2"]
    - Triplets: ["anchor", "positive", "negative"]  (含难负例)
    - Sets:     {"set": ["text1", "text2", ...]}     (同图多 caption → 全组合对)

选择 ~2.5M 对 (含 ~600k 带难负例三元组), 覆盖: 检索(MSMARCO) / 重复问题
(StackExchange, Quora) / QA(GooAQ, ELI5, SQuAD, Yahoo) / 摘要(WikiHow,
SimpleWiki) / 图像 caption(Flickr30k, COCO — 与 MTEB STS12-16 同源, 必须过滤)。

两阶段:
  --stage collect  下载 → 解析 → 跨数据集对级去重 → {name}_full.jsonl
  --stage filter   MTEB eng STS eval 句子 blocklist 过滤 → {name}_clean.jsonl

输出目录: paper/data/sts_retrieval/
  统一格式: {"sentence1": ..., "sentence2": ..., "negative": ... (可选)}

注意: quora_duplicates(对) 与 quora_triplets(三元组) 同源重叠, 去重保留三元组版。

用法:
  uv run --project ../../scripts python collect_retrieval_sts.py --stage collect
  uv run --project ../../scripts python collect_retrieval_sts.py --stage filter
"""

import argparse
import gzip
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
OUT_DIR = PAPER_DIR / "data" / "sts_retrieval"
REPORT_PATH = OUT_DIR / "report.json"

REPO_ID = "sentence-transformers/embedding-training-data"

# (文件名, 输出名, 采样上限 0=全量)
# 注: wikihow.jsonl.gz 在 README 中列出但仓库未上传 (404), 用 agnews 替代
DATASETS = [
    ("msmarco-triplets.jsonl.gz", "msmarco_triplets", 0),
    ("stackexchange_duplicate_questions_title_title.jsonl.gz", "stackexchange_title", 0),
    ("quora_duplicates_triplets.jsonl.gz", "quora_triplets", 0),
    ("quora_duplicates.jsonl.gz", "quora_pairs", 0),
    ("eli5_question_answer.jsonl.gz", "eli5_qa", 0),
    ("squad_pairs.jsonl.gz", "squad_qa", 0),
    ("agnews.jsonl.gz", "agnews", 300_000),
    ("SimpleWiki.jsonl.gz", "simplewiki", 0),
    ("flickr30k_captions.jsonl.gz", "flickr30k_captions", 0),
    ("coco_captions.jsonl.gz", "coco_captions", 0),
    ("yahoo_answers_title_answer.jsonl.gz", "yahoo_ta", 300_000),
    ("gooaq_pairs.jsonl.gz", "gooaq_qa", 300_000),
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


def parse_line(obj) -> list:
    """解析一行 → [(s1, s2, neg_or_None), ...]

    支持格式:
      ["t1", "t2"]                     → 1 对
      ["a", "p", "n"]                  → 1 三元组
      {"set": [t1, t2, ...]}           → C(n,2) 对
      {"query": q, "pos": [...], "neg": [...]} → (q, pos[0], neg[0] 若有)
    """
    out = []
    if isinstance(obj, list):
        if len(obj) == 2:
            s1, s2 = obj[0].strip(), obj[1].strip()
            if s1 and s2:
                out.append((s1, s2, None))
        elif len(obj) == 3:
            s1, s2, neg = obj[0].strip(), obj[1].strip(), obj[2].strip()
            if s1 and s2 and neg:
                out.append((s1, s2, neg))
    elif isinstance(obj, dict):
        if "set" in obj:
            texts = [t.strip() for t in obj["set"] if isinstance(t, str) and t.strip()]
            for a, b in combinations(texts, 2):
                out.append((a, b, None))
        elif "query" in obj and "pos" in obj:
            q = obj["query"].strip()
            pos = [t.strip() for t in obj["pos"] if isinstance(t, str) and t.strip()]
            negs = [t.strip() for t in obj.get("neg", []) if isinstance(t, str) and t.strip()]
            if q and pos:
                neg = negs[0] if negs else None
                if neg:
                    out.append((q, pos[0], neg))
                else:
                    out.append((q, pos[0], None))
    return out


# ---------------------------------------------------------------- collect

def stage_collect() -> None:
    from huggingface_hub import hf_hub_download

    print("=" * 60)
    print("阶段 1: 下载检索式数据集 + 解析 + 去重")
    print("=" * 60, flush=True)

    report = load_report()
    report["collect"] = {}
    seen_pairs = set()  # (norm_s1, norm_s2) 双向, 跨数据集对级去重
    total = 0

    for fname, name, limit in DATASETS:
        print(f"\n-- {name} ({fname}) --", flush=True)
        try:
            local = hf_hub_download(repo_id=REPO_ID, filename=fname,
                                    repo_type="dataset")
        except Exception as e:
            print(f"  [skip] 下载失败 ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        records = []
        n_lines = 0
        n_parsed = 0
        n_dup = 0
        stride = 0  # 采样步长
        if limit > 0:
            # 先数行数决定步长 (保留原文件不动)
            with gzip.open(local, "rt", encoding="utf-8") as f:
                for _ in f:
                    n_lines += 1
            stride = max(1, (n_lines + limit - 1) // limit)
            print(f"  总行数 {n_lines}, 步长采样 stride={stride} → 目标 ~{limit}", flush=True)

        i = 0
        with gzip.open(local, "rt", encoding="utf-8") as f:
            for line in f:
                if stride and (i % stride != 0):
                    i += 1
                    continue
                i += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for s1, s2, neg in parse_line(obj):
                    key, rkey = (norm(s1), norm(s2)), (norm(s2), norm(s1))
                    if key in seen_pairs or rkey in seen_pairs:
                        n_dup += 1
                        continue
                    seen_pairs.add(key)
                    rec = {"sentence1": s1, "sentence2": s2}
                    if neg:
                        rec["negative"] = neg
                    records.append(rec)
                    n_parsed += 1

        n_trip = sum(1 for r in records if "negative" in r)
        print(f"  解析 {n_parsed} 对 (三元组 {n_trip}, 重复剔除 {n_dup})", flush=True)
        save_jsonl(OUT_DIR / f"{name}_full.jsonl", records)
        report["collect"][name] = {"kept": len(records), "triplets": n_trip,
                                   "dup": n_dup, "stride": stride}
        total += len(records)

    report["collect"]["total"] = total
    save_report(report)
    print(f"\n合计收集: {total} 对", flush=True)


# ---------------------------------------------------------------- filter

def stage_filter() -> None:
    from collect_million_sts import build_blocklist

    print("=" * 60)
    print("阶段 2: 泄露过滤 (MTEB eval 句子级, 含 negative)")
    print("=" * 60, flush=True)

    block = build_blocklist()

    report = load_report()
    report["filter"] = {"blocklist_size": len(block)}

    total = 0
    for _, name, _ in DATASETS:
        src = OUT_DIR / f"{name}_full.jsonl"
        if not src.exists():
            print(f"[skip] {src} 不存在", flush=True)
            continue
        records = read_jsonl(src)
        kept, removed = [], 0
        for r in records:
            texts = [r["sentence1"], r["sentence2"], r.get("negative", "")]
            if any(norm(t) in block for t in texts if t):
                removed += 1
                continue
            kept.append(r)
        print(f"  {name}: {len(records)} -> {len(kept)} (剔除 {removed})", flush=True)
        save_jsonl(OUT_DIR / f"{name}_clean.jsonl", kept)
        report["filter"][name] = {"raw": len(records), "kept": len(kept), "removed": removed}
        total += len(kept)

    report["filter"]["total"] = total
    save_report(report)
    print(f"\n过滤后合计: {total} 对", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索式 STS 训练数据收集")
    parser.add_argument("--stage", choices=["collect", "filter"], required=True)
    args = parser.parse_args()

    if args.stage == "collect":
        stage_collect()
    else:
        stage_filter()

    return 0


if __name__ == "__main__":
    sys.exit(main())
