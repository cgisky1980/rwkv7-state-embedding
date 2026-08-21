"""批量导出多个层的 WKV state + hidden 到 .bin 缓存（RWKVSHP 格式）。

跨层诊断显示：L6 的 PHS/hidden 显著高于现有 L12，L21 hidden 最高。本脚本为
train/dev/test/extra 全量导出指定层的 state + hidden，供监督训练验证「换层」。

复用 albatross extract_features_batch 的按长度分桶并发推理，导出指定层。

用法:
  run_with_msvc.bat extract_multi_layer_cache.py --layers 6 21 [--batch-size 16]
输出: saved/cache/sts_pair_l{L}_{split}.bin  (RWKVSHP, 与现有 L12 缓存同格式)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from albatross_wrapper import load_model  # noqa: E402

PAPER_DIR = SCRIPT_DIR.parent
MODEL_PATH = PAPER_DIR / "models" / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
VOCAB_PATH = SCRIPT_DIR / "lib" / "rwkv_vocab_v20230424.txt"
DATA_DIR = PAPER_DIR / "data"
CACHE_DIR = Path("saved/cache")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_bin(path: Path, states: np.ndarray, hiddens: np.ndarray) -> None:
    """保存 RWKVSHP 格式。header: magic(7)+n(4)+state_dim(4)+hidden_dim(4)+pad(4)=23B"""
    states = states.astype(np.float32, copy=False)
    hiddens = hiddens.astype(np.float32, copy=False)
    n, sd, hd = states.shape[0], states.shape[1], hiddens.shape[1]
    header = b"RWKVSHP" + struct.pack("<III", n, sd, hd) + b"\x00" * 4
    body = np.concatenate([states, hiddens], axis=1).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header + body)
    print(f"  saved: {path} ({len(body)/1e6:.1f} MB)", flush=True)


def extract_batch_multi_layer(model, tokenizer, texts, batch_size, max_length, layers):
    from rwkv7 import RWKV_x070_TMix_seq_batch, RWKV_x070_CMix_seq_batch

    z = model.z
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size
    n_embd = model.n_embd
    device = z["emb.weight"].device

    tokens_list = []
    for text in texts:
        tokens = tokenizer.encode(text)[:max_length]
        if len(tokens) == 0:
            tokens = [0]
        tokens_list.append(tokens)

    from collections import defaultdict
    buckets = defaultdict(list)
    for i, tokens in enumerate(tokens_list):
        buckets[len(tokens)].append((i, tokens))

    n = len(texts)
    state_dim = n_head * head_size * head_size
    states = {L: np.zeros((n, state_dim), dtype=np.float32) for L in layers}
    hiddens = {L: np.zeros((n, n_embd), dtype=np.float32) for L in layers}

    print(f"  分桶数: {len(buckets)}, batch_size={batch_size}", flush=True)
    t0 = time.time()
    processed = 0
    for length, items in sorted(buckets.items()):
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            bsz = len(batch_items)
            batch_tokens = [t for _, t in batch_items]
            orig_indices = [i for i, _ in batch_items]
            state = model.generate_zero_state(bsz)
            x = z["emb.weight"][torch.tensor(batch_tokens, device=device)]
            v_first = torch.empty_like(x)
            for i in range(n_layer):
                bbb = f"blocks.{i}."
                att = f"blocks.{i}.att."
                ffn = f"blocks.{i}.ffn."
                xx = F.layer_norm(x, (n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])
                xx, v_first = RWKV_x070_TMix_seq_batch(
                    i, n_head, head_size, xx, state[0][i], v_first, state[1][i],
                    z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"], z[att + "x_a"], z[att + "x_g"],
                    z[att + "w0"], z[att + "w1"], z[att + "w2"], z[att + "a0"], z[att + "a1"], z[att + "a2"],
                    z[att + "v0"], z[att + "v1"], z[att + "v2"],
                    z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
                    z[att + "receptance.weight"], z[att + "key.weight"], z[att + "value.weight"], z[att + "output.weight"],
                    z[att + "ln_x.weight"], z[att + "ln_x.bias"], state[2],
                )
                x = x + xx
                layer_no = i + 1  # 1-indexed
                if layer_no in layers:
                    states[layer_no][orig_indices] = state[1][i].reshape(bsz, -1).cpu().numpy()
                xx = F.layer_norm(x, (n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
                xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn + "x_k"], z[ffn + "key.weight"], z[ffn + "value.weight"])
                x = x + xx
                if layer_no in layers:
                    hiddens[layer_no][orig_indices] = x.float().mean(dim=1).cpu().numpy()
            processed += bsz
            if processed % 1000 < batch_size or processed == n:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                eta = (n - processed) / max(rate, 1e-6)
                print(f"    [{processed}/{n}] {rate:.0f} sps, ETA {eta:.0f}s", flush=True)
    print(f"  推理完成 ({time.time()-t0:.1f}s)", flush=True)
    return states, hiddens


SPLIT_CONF = {
    "train": ("sts_dedup", "sts_train.jsonl"),
    "dev": ("sts", "sts_dev.jsonl"),
    "test": ("sts", "sts_test.jsonl"),
    "extra": ("sts_dedup", "extra_train.jsonl"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    layers = sorted(set(args.layers))

    print("加载 albatross 0.4B 模型...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(MODEL_PATH, VOCAB_PATH)
    print(f"  加载完成 ({time.time()-t0:.1f}s)", flush=True)

    for split, (subdir, fname) in SPLIT_CONF.items():
        data_path = DATA_DIR / subdir / fname
        if not data_path.exists():
            print(f"  [skip] {data_path} 不存在", flush=True)
            continue
        records = read_jsonl(data_path)
        sentences = []
        for r in records:
            sentences.append(r["sentence1"])
            sentences.append(r["sentence2"])
        print(f"\n-- {split} -- ({len(records)} pairs)", flush=True)
        states, hiddens = extract_batch_multi_layer(
            model, tokenizer, sentences, args.batch_size, args.max_length, layers
        )
        for L in layers:
            out_path = CACHE_DIR / f"sts_pair_l{L}_{split}.bin"
            save_bin(out_path, states[L], hiddens[L])

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()