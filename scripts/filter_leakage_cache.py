"""缓存级泄露过滤: 用 MTEB eng STS eval 句子 blocklist 过滤已提取的 npz 缓存。

背景: clean 缓存 (cache_python_0.1b_clean) 由临时脚本 filter_leakage_state.py 生成
(已删), 本脚本是它的正式版, 用于过滤新提取的缓存 (如 PromptEOL 的
cache_python_0.1b_clean_eol)。过滤后 train/nli_train 对数应与 mean clean 缓存一致
(train 5033 / nli_train 9951), 可交叉验证。

逻辑 (与 collect_million_sts.stage_filter 一致, 句子级精确匹配):
  pair 保留条件: norm(sentence1) ∉ block 且 norm(sentence2) ∉ block
  npz 行为交错 s1,s2 → 保留偶/奇行对

用法:
  uv run --project ../../scripts python filter_leakage_cache.py \
      --cache-dir ../cache_python_0.1b_clean_eol --layer 11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from collect_million_sts import build_blocklist, norm, read_jsonl  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent / "data"

# split -> (jsonl 路径, npz 文件名中的 split 名)
SPLIT_FILES = {
    "train": DATA_DIR / "sts_dedup" / "sts_train.jsonl",
    "nli_train": DATA_DIR / "sts_dedup" / "nli_train.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True, help="要过滤的缓存目录")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--splits", type=str, default="train,nli_train")
    args = ap.parse_args()

    block = build_blocklist()

    for split in args.splits.split(","):
        jsonl = SPLIT_FILES[split]
        npz_path = args.cache_dir / f"sts_pair_l{args.layer}_{split}.npz"
        if not npz_path.exists():
            print(f"[skip] {npz_path} 不存在", flush=True)
            continue

        records = read_jsonl(jsonl)
        with np.load(npz_path) as z:
            states = z["states"].astype(np.float16, copy=True) if "states" in z.files else None
            hiddens = z["hiddens"].astype(np.float16, copy=True)
            scores = z["scores"].astype(np.float32, copy=True)
        n_pairs = len(scores)
        assert hiddens.shape[0] == 2 * n_pairs, \
            f"hiddens 行数 {hiddens.shape[0]} != 2×pairs {2 * n_pairs}"
        assert len(records) == n_pairs, \
            f"jsonl 记录数 {len(records)} != npz pairs {n_pairs} (jsonl 与缓存不同源?)"

        keep = np.array([
            norm(r["sentence1"]) not in block and norm(r["sentence2"]) not in block
            for r in records
        ], dtype=bool)
        removed = int((~keep).sum())

        row_idx = np.repeat(np.where(keep)[0] * 2, 2) + np.tile([0, 1], keep.sum())
        out = {"hiddens": hiddens[row_idx], "scores": scores[keep]}
        if states is not None:
            out["states"] = states[row_idx]
        np.savez_compressed(npz_path, **out)

        print(f"  {split}: {n_pairs} -> {int(keep.sum())} pairs (剔除泄露 {removed}) → {npz_path.name}",
              flush=True)


if __name__ == "__main__":
    main()
