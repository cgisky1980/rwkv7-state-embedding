"""使用官方 albatross 推理引擎并发提取 RWKV-7 特征。

albatross (BlinkDL) 是官方验证过的 RWKV-7 PyTorch 推理实现：
  - WKV kernel: CUDA 扩展（编译时自动加载）
  - 其余逻辑: 纯 PyTorch
  - forward_seq_batch: 支持多序列并发推理

输出 (states, hiddens) 二元组，保存为 .npz (float16 节省空间):
  - states:  (N, 65536)  WKV state at layer 12 (16 heads × 64 × 64)
  - hiddens: (N, 1024)   mean-pooled hidden state (last layer FFN output)

三个任务:
  cluster:        twentynewsgroups.jsonl (字段 text, label)
  sts:            sts_{train,dev,test}.jsonl (字段 sentence1, sentence2, score)
  classification: golden_balanced.jsonl (字段 text, tier)

用法 (Windows 需先激活 MSVC 环境):
  run_with_msvc.bat extract_features.py --task all
  run_with_msvc.bat extract_features.py --task cluster
  run_with_msvc.bat extract_features.py --task sts --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from albatross_wrapper import load_model, extract_features, extract_features_batch  # noqa: E402
from cache import save_npz  # noqa: E402

# ============================================================
# 路径
# ============================================================
PAPER_DIR = SCRIPT_DIR.parent
MODEL_PATH = PAPER_DIR / "models" / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
VOCAB_PATH = SCRIPT_DIR / "lib" / "rwkv_vocab_v20230424.txt"
DATA_DIR = PAPER_DIR / "data"
OUTPUT_DIR = PAPER_DIR / "cache_python"

# ============================================================
# 配置 (0.4B RWKV-7)
# ============================================================
LAYER = 12  # 提取 WKV state 的层


# ============================================================
# 工具
# ============================================================
def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================
# 三个任务
# ============================================================
def run_cluster(model, tokenizer, args) -> None:
    """任务一: 主题聚类 - twentynewsgroups.jsonl

    采样策略：按类别均匀采样（每类 100 个，seed=42），
    类别均匀使 KMeans 评估更稳定。
    """
    print("\n" + "=" * 60, flush=True)
    print("任务一: 主题聚类 (twentynewsgroups)", flush=True)
    print("=" * 60, flush=True)

    data_path = DATA_DIR / "clustering" / "twentynewsgroups.jsonl"
    records = read_jsonl(data_path)
    print(f"  全量样本: {len(records)}", flush=True)

    # 按类别均匀采样（每类 100，seed=42），与 Rust 实验一致
    import random
    from collections import defaultdict
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)
    random.seed(42)
    per_class = 100  # 每类 100，共 2000
    sampled = []
    for cls in sorted(by_class.keys()):
        recs = by_class[cls][:]
        random.shuffle(recs)
        sampled.extend(recs[:per_class])
    records = sampled
    print(f"  按类别采样: 每类 {per_class}, 共 {len(records)} 样本", flush=True)

    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int32)

    states, hiddens = extract_features_batch(
        model, tokenizer, texts, args.batch_size, args.max_length, layer=args.layer
    )

    out_path = OUTPUT_DIR / f"cluster_l{args.layer}.npz"
    save_npz(out_path, states, hiddens, extra={"labels": labels})


def run_cluster_full(model, tokenizer, args) -> None:
    """任务五: 提取全量 twentynewsgroups 特征 (59545 样本, 用于监督聚类 projection 训练)."""
    print("\n" + "=" * 60, flush=True)
    print("任务五: 全量聚类特征提取 (59545 samples)", flush=True)
    print("=" * 60, flush=True)

    data_path = DATA_DIR / "clustering" / "twentynewsgroups.jsonl"
    records = read_jsonl(data_path)
    print(f"  全量样本: {len(records)}", flush=True)

    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int32)

    states, hiddens = extract_features_batch(
        model, tokenizer, texts, args.batch_size, args.max_length, layer=args.layer
    )

    out_path = OUTPUT_DIR / f"cluster_full_l{args.layer}.npz"
    save_npz(out_path, states, hiddens, extra={"labels": labels})


def run_cluster_20ng_full(model, tokenizer, args) -> None:
    """任务: 提取 sklearn 全文版 20NG 严格去重 split 的特征

    数据: data/clustering_sklearn_20ng/{train,dev,test}.jsonl
    输出: cache_python/cluster_sklearn_20ng_l{layer}_{split}.npz
    """
    print("\n" + "=" * 60, flush=True)
    print("任务: 20NG (sklearn 全文, 严格去重 split) 特征提取", flush=True)
    print("=" * 60, flush=True)

    data_dir = DATA_DIR / "clustering_sklearn_20ng"
    for split in ["train", "dev", "test"]:
        data_path = data_dir / f"{split}.jsonl"
        if not data_path.exists():
            print(f"  [skip] {data_path} 不存在", flush=True)
            continue
        records = read_jsonl(data_path)
        print(f"\n  -- {split} -- ({len(records)} samples)", flush=True)

        texts = [r["text"] for r in records]
        labels = np.array([r["label"] for r in records], dtype=np.int32)

        states, hiddens = extract_features_batch(
            model, tokenizer, texts, args.batch_size, args.max_length, layer=args.layer
        )

        out_path = OUTPUT_DIR / f"cluster_sklearn_20ng_l{args.layer}_{split}.npz"
        save_npz(out_path, states, hiddens, extra={"labels": labels})


def run_sts(model, tokenizer, args) -> None:
    """任务二: 语义相似度 - STS-Benchmark"""
    print("\n" + "=" * 60, flush=True)
    print("任务二: 语义相似度 (STS-Benchmark)", flush=True)
    print("=" * 60, flush=True)

    # 支持 --sts-subdir 切换到去重后的数据目录
    sts_subdir = getattr(args, "sts_subdir", "sts")
    sts_dir = DATA_DIR / sts_subdir
    print(f"  STS 数据目录: {sts_dir}", flush=True)

    for split in ["train", "dev", "test"]:
        # train 可能来自 dedup 目录（sts_train.jsonl），dev/test 始终从原 sts 目录读取（eval 不去重）
        if split == "train" and sts_subdir != "sts":
            data_path = sts_dir / f"sts_{split}.jsonl"
        else:
            data_path = DATA_DIR / "sts" / f"sts_{split}.jsonl"
        if not data_path.exists():
            print(f"  [skip] {data_path} 不存在", flush=True)
            continue

        records = read_jsonl(data_path)
        print(f"\n  -- {split} -- ({len(records)} pairs)", flush=True)

        # 收集所有句子 (交错: s1, s2, s1, s2, ...)
        sentences = []
        for r in records:
            sentences.append(r["sentence1"])
            sentences.append(r["sentence2"])

        states, hiddens = extract_features_batch(
            model, tokenizer, sentences, args.batch_size, args.max_length, layer=args.layer
        )

        scores = np.array([r["score"] for r in records], dtype=np.float32)
        out_path = OUTPUT_DIR / f"sts_pair_l{args.layer}_{split}.npz"
        save_npz(out_path, states, hiddens, extra={"scores": scores})


def _load_state_pca_transform(path: Path):
    """从投影器 .pt 读取 (head_indices, pca_components, pca_mean, per_head_dim)"""
    import torch
    ckpt = torch.load(path, map_location="cpu")
    sel = ckpt.get("head_selection")
    if sel is None:
        raise SystemExit(f"错误: {path} 不含 head_selection, 无法用于 state 降维")
    per = sel["head_size"] * sel["head_size"]
    return (
        list(sel["head_indices"]),
        np.asarray(sel["pca_components"], dtype=np.float32),  # (D, K*per)
        np.asarray(sel["pca_mean"], dtype=np.float32),         # (K*per,)
        per,
    )


def _extract_state_pca_chunked(model, tokenizer, sentences, args, transform, n_embd: int):
    """分块提取 + 实时 head 选择/PCA 降维 (避免全量 49152 维 state 驻留内存/磁盘)。

    返回 (states_pca (N, D) fp16, hiddens (N, C) fp16)。
    """
    head_idx, comps, mean, per = transform
    cols = np.concatenate([np.arange(h * per, (h + 1) * per) for h in head_idx])
    n = len(sentences)
    out_states = np.zeros((n, comps.shape[0]), dtype=np.float16)
    out_hiddens = np.zeros((n, n_embd), dtype=np.float16)

    CHUNK = 100_000  # 每块 transient state 内存 ~9.8GB fp16
    n_chunks = (n + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        c0 = ci * CHUNK
        chunk = sentences[c0:c0 + CHUNK]
        print(f"\n  [chunk {ci+1}/{n_chunks}] {len(chunk)} texts", flush=True)
        states, hiddens = extract_features_batch(
            model, tokenizer, chunk, args.batch_size, args.max_length,
            layer=args.layer, need_state=True,
        )
        x = states[:, cols].astype(np.float32)
        x = (x - mean) @ comps.T
        out_states[c0:c0 + len(chunk)] = x.astype(np.float16)
        out_hiddens[c0:c0 + len(chunk)] = hiddens
        del states, hiddens, x
    return out_states, out_hiddens


def run_sts_extra(model, tokenizer, args) -> None:
    """任务四: 提取额外 STS 训练数据 (用于训练 universal projection).

    数据集:
      - nli_train.jsonl: 10k NLI pairs (score 1-5)
      - extra_train.jsonl: 22k STS pairs (score 0-5)
      - sickr.jsonl: 9.9k SICK-R pairs (score 1-5)

    合计 ~42k pairs, 配合 STS-B train (5.7k) 共 ~47.6k 训练数据.
    使用 extract_features_batch (按长度分桶并发推理), 速度提升 5-10x.
    """
    print("\n" + "=" * 60, flush=True)
    print("任务四: 额外 STS 训练数据特征提取 (batch 并发)", flush=True)
    print("=" * 60, flush=True)

    datasets = args.sts_names.split(",") if args.sts_names else ["nli_train", "extra_train", "sickr"]
    sts_subdir = getattr(args, "sts_subdir", "sts")
    sts_dir = DATA_DIR / sts_subdir
    print(f"  STS 数据目录: {sts_dir}", flush=True)

    # state 实时降维变换 (可选)
    state_pca_transform = None
    if getattr(args, "state_pca_from", ""):
        sp_path = Path(args.state_pca_from)
        if not sp_path.exists():
            sp_path = PAPER_DIR / args.state_pca_from
        state_pca_transform = _load_state_pca_transform(sp_path)
        h_idx, comps, _, _ = state_pca_transform
        print(f"  state 降维: heads={h_idx} → PCA {comps.shape[0]} 维 (源: {sp_path})", flush=True)

    for name in datasets:
        data_path = sts_dir / f"{name}.jsonl"
        if not data_path.exists():
            print(f"  [skip] {name}: {data_path} 不存在", flush=True)
            continue

        records = read_jsonl(data_path)
        print(f"\n  -- {name} -- ({len(records)} pairs)", flush=True)

        # 检索式数据 (collect_retrieval_sts.py 输出) 含 negative 字段 → 三元组 stride 3
        has_neg = bool(records) and "negative" in records[0]
        if has_neg:
            print(f"  含 negative 字段, 按三元组提取 (stride 3)", flush=True)

        sentences = []
        for r in records:
            sentences.append(r["sentence1"])
            sentences.append(r["sentence2"])
            if has_neg:
                sentences.append(r["negative"])

        if state_pca_transform is not None:
            # 分块提取 + 实时降维: states 槽存低维 PCA 特征
            t0 = time.time()
            states, hiddens = _extract_state_pca_chunked(
                model, tokenizer, sentences, args, state_pca_transform, model.n_embd)
            print(f"  {name} 提取+降维完成 ({time.time()-t0:.1f}s): "
                  f"states {states.shape}, hiddens {hiddens.shape}", flush=True)
        else:
            states, hiddens = extract_features_batch(
                model, tokenizer, sentences, args.batch_size, args.max_length,
                layer=args.layer, need_state=not args.hidden_only,
            )

        # InfoNCE 训练无需分数, 检索式数据无 score 字段 → 填充 1.0 (保持缓存格式)
        scores = np.array([r.get("score", 1.0) for r in records], dtype=np.float32)
        out_path = OUTPUT_DIR / f"sts_pair_l{args.layer}_{name}.npz"
        save_npz(out_path, states, hiddens, extra={"scores": scores})


def run_classification(model, tokenizer, args) -> None:
    """任务三: 任务分类 - golden_balanced.jsonl"""
    print("\n" + "=" * 60, flush=True)
    print("任务三: 任务分类 (golden_balanced)", flush=True)
    print("=" * 60, flush=True)

    data_path = DATA_DIR / "golden_balanced.jsonl"
    records = read_jsonl(data_path)
    if args.limit > 0:
        np.random.seed(42)
        idx = np.random.choice(len(records), min(args.limit, len(records)), replace=False)
        records = [records[i] for i in sorted(idx)]
    print(f"  样本数: {len(records)}", flush=True)

    texts = [r["text"] for r in records]
    labels = np.array([r["tier"] for r in records], dtype=np.int32)

    states, hiddens = extract_features(
        model, tokenizer, texts, args.batch_size, args.max_length, layer=args.layer
    )

    out_path = OUTPUT_DIR / f"classification_l{args.layer}.npz"
    save_npz(out_path, states, hiddens, extra={"labels": labels})


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="批量并发特征提取 (albatross 官方推理)")
    parser.add_argument(
        "--task",
        choices=["cluster", "cluster_full", "cluster_sklearn", "sts", "sts_extra", "classification", "all"],
        default="all",
        help="运行哪个任务",
    )
    parser.add_argument("--limit", type=int, default=0, help="样本上限 (0=全部)")
    parser.add_argument("--batch-size", type=int, default=8, help="并发 batch 大小")
    parser.add_argument("--max-length", type=int, default=512, help="单序列最大 token 数")
    parser.add_argument("--layer", type=int, default=LAYER, help="提取 WKV state 的层")
    parser.add_argument("--sts-subdir", type=str, default="sts", help="STS 数据子目录 (sts 或 sts_dedup)")
    parser.add_argument("--model-path", type=str, default="", help="模型 .pth 路径 (默认 0.4B)")
    parser.add_argument("--out-dir", type=str, default="", help="输出缓存目录 (默认 cache_python，可传独立目录避免覆盖)")
    parser.add_argument("--hidden-only", action="store_true",
                        help="只提取 hidden 不存 state (百万级数据避免 100GB+ 内存/磁盘)")
    parser.add_argument("--state-pca-from", type=str, default="",
                        help="state 实时降维: 从该投影器 .pt 读取 Top-K head 选择 + PCA 分量, "
                             "提取时将 49152 维 state 变换为低维 (如 256) 后存入 states 槽, "
                             "避免全量 state 落盘 (~700GB→15GB)。与 02/07 的 --state-pca-from 同源")
    parser.add_argument("--sts-names", type=str, default="",
                        help="sts_extra 任务的数据集名列表 (逗号分隔, 默认 nli_train,extra_train,sickr)")
    args = parser.parse_args()

    global MODEL_PATH, OUTPUT_DIR
    if args.model_path:
        MODEL_PATH = Path(args.model_path)
    if args.out_dir:
        OUTPUT_DIR = Path(args.out_dir)

    print("=" * 60, flush=True)
    print("批量并发特征提取 (albatross 官方推理)", flush=True)
    print(f"  model: {MODEL_PATH}", flush=True)
    print(f"  batch_size: {args.batch_size} (并发推理)", flush=True)
    print(f"  max_length: {args.max_length} tokens", flush=True)
    print(f"  layer: {args.layer}", flush=True)
    print(f"  output: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)

    # 加载 albatross 模型
    print("\n加载 albatross RWKV-7 0.4B 模型...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(MODEL_PATH, VOCAB_PATH)
    print(f"  加载完成 ({time.time()-t0:.1f}s)", flush=True)
    print(f"  n_layer={model.n_layer}, n_embd={model.n_embd}, n_head={model.n_head}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 运行任务
    if args.task in ("cluster", "all"):
        run_cluster(model, tokenizer, args)
    if args.task in ("cluster_full",):
        run_cluster_full(model, tokenizer, args)
    if args.task in ("cluster_sklearn",):
        run_cluster_20ng_full(model, tokenizer, args)
    if args.task in ("sts", "all"):
        run_sts(model, tokenizer, args)
    if args.task in ("sts_extra", "all"):
        run_sts_extra(model, tokenizer, args)
    if args.task in ("classification", "all"):
        run_classification(model, tokenizer, args)

    print("\n" + "=" * 60, flush=True)
    print("[DONE] 全部特征提取完成", flush=True)
    print(f"  输出目录: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
