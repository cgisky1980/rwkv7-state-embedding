"""提取全量 twentynewsgroups 的多层 hidden state (59545 样本).

用于无监督聚类实验: 多层拼接 + 非线性变换.

层选择: L0, L4, L8, L12, L16, L20, L23 (与 cluster_multilayer_hidden.npz 一致)

输出: cache_python/cluster_full_multilayer.npz
  - L0: (59545, 1024) float16
  - L4: (59545, 1024) float16
  - ...
  - L23: (59545, 1024) float16
  - labels: (59545,) int32
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from albatross_wrapper import load_model  # noqa: E402
from utils import TRIE_TOKENIZER  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "cache_python"

# 提取的层索引 (与 cluster_multilayer_hidden.npz 一致)
EXTRACT_LAYERS = [0, 4, 8, 12, 16, 20, 23]


def read_jsonl(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_multilayer_batch(model, tokenizer, texts, batch_size=16, max_length=128, pad_token=0):
    """批量并发提取多层 hidden state.

    Returns:
        multilayer_hiddens: dict {layer_idx: (N, C) float16}
    """
    from rwkv7 import RWKV_x070_TMix_seq_batch, RWKV_x070_CMix_seq_batch

    n = len(texts)
    z = model.z
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size
    n_embd = model.n_embd

    # tokenize + 截断
    print(f"  Tokenize {n} texts (max_length={max_length})...", flush=True)
    tokens_list = []
    for text in texts:
        tokens = tokenizer.encode(text)[:max_length]
        if len(tokens) == 0:
            tokens = [pad_token]
        tokens_list.append(tokens)

    # 按长度分桶
    buckets = defaultdict(list)
    for i, tokens in enumerate(tokens_list):
        buckets[len(tokens)].append((i, tokens))

    # 多层 hidden 存储
    multilayer_hiddens = {l: np.zeros((n, n_embd), dtype=np.float16) for l in EXTRACT_LAYERS}

    print(f"  分桶数: {len(buckets)}, 总样本: {n}", flush=True)
    print(f"  批量并发推理 (batch_size={batch_size})...", flush=True)
    t0 = time.time()
    processed = 0
    device = z['emb.weight'].device

    for length, items in sorted(buckets.items()):
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            bsz = len(batch_items)
            batch_tokens = [t for _, t in batch_items]
            orig_indices = [i for i, _ in batch_items]

            state = model.generate_zero_state(bsz)
            x = z['emb.weight'][torch.tensor(batch_tokens, device=device)]
            v_first = torch.empty_like(x)

            for i in range(n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = RWKV_x070_TMix_seq_batch(
                    i, n_head, head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2],
                )
                x = x + xx

                xx = F.layer_norm(x, (n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

                # 保存目标层的 hidden (mean pooling)
                if i in EXTRACT_LAYERS:
                    hidden = x.float().mean(dim=1).half()  # (B, C)
                    for k, orig_idx in enumerate(orig_indices):
                        multilayer_hiddens[i][orig_idx] = hidden[k].cpu().numpy()

            processed += bsz
            if processed % 500 < batch_size or processed == n:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                eta = (n - processed) / max(rate, 1e-6)
                print(f"    [{processed}/{n}] {rate:.1f} samples/s, ETA {eta:.0f}s", flush=True)

    print(f"  推理完成 ({time.time()-t0:.1f}s)", flush=True)
    return multilayer_hiddens


def main():
    import argparse
    parser = argparse.ArgumentParser(description="提取全量多层 hidden state")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("提取全量多层 hidden state (59545 samples)", flush=True)
    print(f"  层: {EXTRACT_LAYERS}", flush=True)
    print("=" * 60, flush=True)

    model_path = SCRIPT_DIR.parent / "models" / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
    vocab_path = SCRIPT_DIR / "lib" / "rwkv_vocab_v20230424.txt"
    model, tokenizer = load_model(model_path, vocab_path)

    data_path = DATA_DIR / "clustering" / "twentynewsgroups.jsonl"
    records = read_jsonl(data_path)
    print(f"  全量样本: {len(records)}", flush=True)

    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int32)

    multilayer_hiddens = extract_multilayer_batch(
        model, tokenizer, texts, args.batch_size, args.max_length
    )

    # 保存
    out_path = OUTPUT_DIR / "cluster_full_multilayer.npz"
    save_dict = {f"L{l}": multilayer_hiddens[l] for l in EXTRACT_LAYERS}
    save_dict["labels"] = labels
    np.savez_compressed(out_path, **save_dict)
    print(f"\n  保存: {out_path}", flush=True)
    for l in EXTRACT_LAYERS:
        h = multilayer_hiddens[l]
        print(f"  L{l}: {h.shape} std={h.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
