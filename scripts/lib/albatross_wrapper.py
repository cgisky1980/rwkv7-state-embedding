"""Albatross wrapper: 用官方 albatross 推理引擎提取 RWKV state + hidden。

albatross (BlinkDL/faster_251101) 是官方验证过的 RWKV-7 PyTorch 推理实现，
WKV kernel 用 CUDA 扩展（编译时自动加载），其余逻辑纯 PyTorch。

本 wrapper 提供:
  - load_model(): 加载 albatross RWKV_x070 模型 + tokenizer
  - extract_features(): 单序列推理，提取 WKV state + hidden

state 格式:
  - state[0]: (n_layer, 2, B, C)  = [att_x_prev, ffn_x_prev]
  - state[1]: (n_layer, B, H, N, N) = WKV state  ← 我们要的
  - state[2]: (B,) = token 计数

hidden 提取:
  - 最后一层 FFN 输出 (x + xx 后)
  - mean pooling: 对所有 token 取均值
"""
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent.parent

# 必须先 stub flag_gems (Windows 无 triton)
sys.path.insert(0, str(SCRIPT_DIR))
import flag_gems_stub  # noqa: F401

# albatross reference 路径 (faster_251101, sm_75 兼容，无 cp.async)
ALBATROSS_REF = PAPER_DIR / "albatross_src" / "faster_251101" / "reference"
sys.path.insert(0, str(ALBATROSS_REF))


def load_model(model_path: Path, vocab_path: Path):
    """加载 albatross 模型和 tokenizer。

    Args:
        model_path: .pth 模型文件路径
        vocab_path: rwkv_vocab_v20230424.txt 路径

    Returns:
        model: RWKV_x070 实例
        tokenizer: TRIE_TOKENIZER 实例
    """
    import types
    from rwkv7 import RWKV_x070
    from utils import TRIE_TOKENIZER

    args = types.SimpleNamespace()
    args.MODEL_NAME = str(model_path.with_suffix(""))  # 去掉 .pth 后缀
    model = RWKV_x070(args)
    tokenizer = TRIE_TOKENIZER(str(vocab_path))
    return model, tokenizer


def extract_features(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 8,
    max_length: int = 512,
    layer: int = 12,
    pad_token: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """提取 WKV state + mean hidden state。

    使用 albatross 的 forward_seq 单序列推理（无 padding），确保 state/hidden
    不被 padding token 污染。虽然比并发慢，但结果与 Rust 完全一致。

    Args:
        model: RWKV_x070 实例
        tokenizer: TRIE_TOKENIZER 实例
        texts: 文本列表
        batch_size: 未使用（保留接口兼容）
        max_length: 最大 token 长度（截断）
        layer: 提取 WKV state 的层索引
        pad_token: 空 token id

    Returns:
        states: (N, H*N*N) float16 - WKV state（指定层）
        hiddens: (N, C) float16 - mean pooling hidden state（最后一层 FFN 输出）

    Notes:
        - 单序列推理（forward_seq），无 padding 污染
        - hidden = mean(x_last_layer_ffn) - 与 Rust PostFfn hook 一致
        - state 维度 (B, H, N, N) 与 Rust 一致（k 在前，v 在后）
    """
    # 延迟 import: 模块级函数
    from rwkv7 import RWKV_x070_TMix_seq, RWKV_x070_CMix_seq

    n = len(texts)
    z = model.z
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size
    n_embd = model.n_embd

    # 1. tokenize + 截断
    print(f"  Tokenize {n} texts (max_length={max_length})...", flush=True)
    tokens_list = []
    for text in texts:
        tokens = tokenizer.encode(text)[:max_length]
        if len(tokens) == 0:
            tokens = [pad_token]
        tokens_list.append(tokens)

    # 2. 单序列推理
    state_dim = n_head * head_size * head_size
    all_states = np.zeros((n, state_dim), dtype=np.float16)
    all_hiddens = np.zeros((n, n_embd), dtype=np.float16)

    print(f"  单序列推理 {n} samples...", flush=True)
    t0 = time.time()

    device = z['emb.weight'].device

    for idx in range(n):
        tokens = tokens_list[idx]
        T = len(tokens)

        # 生成 zero state (单序列)
        state = model.generate_zero_state(0)  # bsz=0 → 单序列格式

        # === 复制 forward_seq 的循环，同时拿 hidden ===
        x = z['emb.weight'][torch.tensor(tokens, device=device)]  # (T, C)
        v_first = torch.empty_like(x)

        for i in range(n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            xx = F.layer_norm(x, (n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
            xx, v_first = RWKV_x070_TMix_seq(
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
            xx = RWKV_x070_CMix_seq(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx

        # x: (T, C) - 最后一层 FFN 输出
        # hidden = mean pooling over tokens (与 Rust PostFfn hook 一致)
        hidden = x.float().mean(dim=0).half()  # (C,)

        # WKV state: state[1][layer] shape (H, N, N) for single sequence
        wkv = state[1][layer]  # (H, N, N) - half

        all_states[idx] = wkv.reshape(-1).cpu().numpy()
        all_hiddens[idx] = hidden.cpu().numpy()

        if (idx + 1) % 200 == 0 or idx + 1 == n:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            eta = (n - idx - 1) / max(rate, 1e-6)
            print(f"    [{idx+1}/{n}] {rate:.1f} samples/s, ETA {eta:.0f}s", flush=True)

    print(f"  推理完成 ({time.time()-t0:.1f}s)", flush=True)
    return all_states, all_hiddens


def extract_features_batch(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 16,
    max_length: int = 512,
    layer: int = 12,
    pad_token: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """批量并发提取 WKV state + mean hidden state (按长度分桶, 无 padding 污染).

    相比 extract_features (单序列), 本函数按 token 长度分桶后用 forward_seq_batch
    并发推理, 速度提升 5-10x. 每桶内序列长度相同, 无 padding, state/hidden 不被污染.

    Args:
        model: RWKV_x070 实例
        tokenizer: TRIE_TOKENIZER 实例
        texts: 文本列表
        batch_size: 每桶最大序列数
        max_length: 最大 token 长度（截断）
        layer: 提取 WKV state 的层索引
        pad_token: 空 token id

    Returns:
        states: (N, H*N*N) float16 - WKV state（指定层, 按原 texts 顺序）
        hiddens: (N, C) float16 - mean pooling hidden state（按原 texts 顺序）
    """
    from rwkv7 import RWKV_x070_TMix_seq_batch, RWKV_x070_CMix_seq_batch

    n = len(texts)
    z = model.z
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size
    n_embd = model.n_embd

    # 1. tokenize + 截断
    print(f"  Tokenize {n} texts (max_length={max_length})...", flush=True)
    tokens_list = []
    for text in texts:
        tokens = tokenizer.encode(text)[:max_length]
        if len(tokens) == 0:
            tokens = [pad_token]
        tokens_list.append(tokens)

    # 2. 按长度分桶 (桶内序列长度相同, 无 padding)
    from collections import defaultdict
    buckets = defaultdict(list)  # length -> [(orig_idx, tokens), ...]
    for i, tokens in enumerate(tokens_list):
        buckets[len(tokens)].append((i, tokens))

    state_dim = n_head * head_size * head_size
    all_states = np.zeros((n, state_dim), dtype=np.float16)
    all_hiddens = np.zeros((n, n_embd), dtype=np.float16)

    print(f"  分桶数: {len(buckets)}, 总样本: {n}", flush=True)
    print(f"  批量并发推理 (batch_size={batch_size})...", flush=True)
    t0 = time.time()
    processed = 0
    device = z['emb.weight'].device

    for length, items in sorted(buckets.items()):
        # 桶内按 batch_size 切分
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            bsz = len(batch_items)
            # 提取 tokens (所有序列相同长度)
            batch_tokens = [t for _, t in batch_items]
            orig_indices = [i for i, _ in batch_items]

            # 生成 zero state (batch)
            state = model.generate_zero_state(bsz)

            # 复制 forward_seq_batch 的循环, 同时提取 hidden
            # 输入: (B, T, C)
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

            # x: (B, T, C) - 最后一层 FFN 输出
            # hidden = mean pooling over tokens
            hidden = x.float().mean(dim=1).half()  # (B, C)
            # WKV state: state[1][layer] shape (B, H, N, N)
            wkv = state[1][layer]  # (B, H, N, N) - half

            for k, orig_idx in enumerate(orig_indices):
                all_states[orig_idx] = wkv[k].reshape(-1).cpu().numpy()
                all_hiddens[orig_idx] = hidden[k].cpu().numpy()

            processed += bsz
            if processed % 500 < batch_size or processed == n:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                eta = (n - processed) / max(rate, 1e-6)
                print(f"    [{processed}/{n}] {rate:.1f} samples/s, ETA {eta:.0f}s", flush=True)

    print(f"  推理完成 ({time.time()-t0:.1f}s)", flush=True)
    return all_states, all_hiddens
