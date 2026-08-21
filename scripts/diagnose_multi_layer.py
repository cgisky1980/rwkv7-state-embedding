"""跨层特征拼接诊断（门控路线否定后的新方向）

背景：现有特征只用 L12 的 state + L23 的 hidden mean-pool。GOAL 文档明确列出
「多层 hidden 拼接 (L10-L14)」为待探索方向但从未做。本诊断快速判断：
不同层的 state/hidden 是否携带互补信息，跨层拼接能否超过单层最佳。

方法：对 dev 前 N 对句子，用 albatross 推理循环导出多层 (L6/L9/L12/L15/L18/L21)
的 state 与 hidden (每层 FFN 输出后 x)，对比：
  - 单层 hidden mean-pool 余弦-Spearman（找最优层）
  - 相邻层 hidden 拼接（验证互补）
  - 单层 PHS_V3 余弦-Spearman（找最优层）
  - 最优 hidden 层 + PHS 层 拼接（与现有 L12+S 对齐）

若相邻/跨层拼接显著 > 单层最佳 → 支持投入全量多层重推理。
若无增益 → 单层已足够，方向也否定。

用法:
  run_with_msvc.bat diagnose_multi_layer.py [--n-pairs 400] [--max-length 128]
"""
from __future__ import annotations

import argparse
import json
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
DATA_PATH = PAPER_DIR / "data" / "sts" / "sts_dev.jsonl"
OUT_PATH = Path("参考/diagnose_multi_layer_results.json")

N_EMBD = 1024
NUM_HEADS = 16
HEAD_SIZE = 64
LAYERS = [6, 9, 12, 15, 18, 21]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def spearman_corr(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def compute_per_head_stats_v3(states: np.ndarray) -> np.ndarray:
    n = states.shape[0]
    per_head_dim = HEAD_SIZE * 2 + HEAD_SIZE + 6
    out = np.zeros((n, NUM_HEADS * per_head_dim), dtype=np.float32)
    for h in range(NUM_HEADS):
        s_h = states[:, h * HEAD_SIZE * HEAD_SIZE : (h + 1) * HEAD_SIZE * HEAD_SIZE].reshape(
            n, HEAD_SIZE, HEAD_SIZE
        )
        row_sum = s_h.sum(axis=2)
        col_sum = s_h.sum(axis=1)
        diag = np.diagonal(s_h, axis1=1, axis2=2)
        frob_norm = np.sqrt((s_h**2).sum(axis=(1, 2))).reshape(n, 1)
        energy = (s_h**2).sum(axis=(1, 2)).reshape(n, 1)
        mean = s_h.mean(axis=(1, 2)).reshape(n, 1)
        std = s_h.std(axis=(1, 2)).reshape(n, 1)
        max_val = s_h.max(axis=(1, 2)).reshape(n, 1)
        min_val = s_h.min(axis=(1, 2)).reshape(n, 1)
        offset = h * per_head_dim
        out[:, offset : offset + HEAD_SIZE] = row_sum
        out[:, offset + HEAD_SIZE : offset + HEAD_SIZE * 2] = col_sum
        out[:, offset + HEAD_SIZE * 2 : offset + HEAD_SIZE * 3] = diag
        out[:, offset + HEAD_SIZE * 3 : offset + HEAD_SIZE * 3 + 1] = frob_norm
        out[:, offset + HEAD_SIZE * 3 + 1 : offset + HEAD_SIZE * 3 + 2] = energy
        out[:, offset + HEAD_SIZE * 3 + 2 : offset + HEAD_SIZE * 3 + 3] = mean
        out[:, offset + HEAD_SIZE * 3 + 3 : offset + HEAD_SIZE * 3 + 4] = std
        out[:, offset + HEAD_SIZE * 3 + 4 : offset + HEAD_SIZE * 3 + 5] = max_val
        out[:, offset + HEAD_SIZE * 3 + 5 : offset + HEAD_SIZE * 3 + 6] = min_val
    return out


def extract_multi_layer(model, tokenizer, texts, max_length, layers):
    """导出多个层的 state 与 hidden (每层 FFN 后 x)。"""
    from rwkv7 import RWKV_x070_TMix_seq, RWKV_x070_CMix_seq

    z = model.z
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size
    n_embd = model.n_embd
    device = z["emb.weight"].device

    n = len(texts)
    state_dim = n_head * head_size * head_size
    # states_by_layer[layer] shape (n, state_dim)
    states = {L: np.zeros((n, state_dim), dtype=np.float32) for L in layers}
    # hiddens_by_layer[layer] shape (n, n_embd)  mean-pool FFN 输出
    hiddens = {L: np.zeros((n, n_embd), dtype=np.float32) for L in layers}

    for idx, text in enumerate(texts):
        tokens = tokenizer.encode(text)[:max_length]
        if len(tokens) == 0:
            tokens = [0]
        state = model.generate_zero_state(0)
        x = z["emb.weight"][torch.tensor(tokens, device=device)]
        v_first = torch.empty_like(x)
        for i in range(n_layer):
            bbb = f"blocks.{i}."
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."
            xx = F.layer_norm(x, (n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])
            xx, v_first = RWKV_x070_TMix_seq(
                i, n_head, head_size, xx, state[0][i], v_first, state[1][i],
                z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"], z[att + "x_a"], z[att + "x_g"],
                z[att + "w0"], z[att + "w1"], z[att + "w2"], z[att + "a0"], z[att + "a1"], z[att + "a2"],
                z[att + "v0"], z[att + "v1"], z[att + "v2"],
                z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
                z[att + "receptance.weight"], z[att + "key.weight"], z[att + "value.weight"], z[att + "output.weight"],
                z[att + "ln_x.weight"], z[att + "ln_x.bias"], state[2],
            )
            x = x + xx
            # TMix 已更新 state[1][i]（第 i+1 层, 1-indexed）
            if (i + 1) in layers:
                states[i + 1][idx] = state[1][i].reshape(-1).cpu().numpy()
            xx = F.layer_norm(x, (n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
            xx = RWKV_x070_CMix_seq(xx, state[0][i], z[ffn + "x_k"], z[ffn + "key.weight"], z[ffn + "value.weight"])
            x = x + xx
            # 第 i+1 层完整前向后的 FFN 输出（等价 wrapper 的 hidden 口径）
            if (i + 1) in layers:
                hiddens[i + 1][idx] = x.float().mean(dim=0).cpu().numpy()
        if (idx + 1) % 200 == 0 or idx + 1 == n:
            print(f"    [{idx+1}/{n}]", flush=True)
    return states, hiddens


def cos_spearman(feat_list, records):
    n = len(records)
    cos = []
    for i in range(n):
        a = feat_list[2 * i]
        b = feat_list[2 * i + 1]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos.append(c)
    scores = np.array([r["score"] for r in records], dtype=np.float32)
    return spearman_corr(cos, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    records = read_jsonl(DATA_PATH)[: args.n_pairs]
    print(f"pair 数: {len(records)}", flush=True)

    print("加载 albatross 0.4B 模型...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(MODEL_PATH, VOCAB_PATH)
    print(f"  加载完成 ({time.time()-t0:.1f}s)", flush=True)

    sentences = []
    for r in records:
        sentences.append(r["sentence1"])
        sentences.append(r["sentence2"])

    print(f"推理 {len(sentences)} 句, 导出层 {LAYERS}...", flush=True)
    t0 = time.time()
    states, hiddens = extract_multi_layer(model, tokenizer, sentences, args.max_length, LAYERS)
    print(f"  推理完成 ({time.time()-t0:.1f}s)", flush=True)

    # 单层 hidden mean-pool
    print("\n单层 hidden mean-pool 余弦-Spearman:", flush=True)
    h_layer_sp = {}
    for L in LAYERS:
        sp = cos_spearman(hiddens[L].tolist(), records)
        h_layer_sp[L] = sp
        print(f"  L{L:2d} hidden: {sp:.4f}", flush=True)

    # 单层 PHS_V3
    print("\n单层 PHS_V3 余弦-Spearman:", flush=True)
    s_layer_sp = {}
    phs_by_layer = {}
    for L in LAYERS:
        phs = compute_per_head_stats_v3(states[L])
        phs_by_layer[L] = phs
        sp = cos_spearman(phs.tolist(), records)
        s_layer_sp[L] = sp
        print(f"  L{L:2d} PHS: {sp:.4f}", flush=True)

    # 相邻层 hidden 拼接
    print("\n相邻层 hidden 拼接:", flush=True)
    concat_sp = {}
    for i in range(len(LAYERS) - 1):
        La, Lb = LAYERS[i], LAYERS[i + 1]
        feat = np.concatenate([hiddens[La], hiddens[Lb]], axis=1)
        sp = cos_spearman(feat.tolist(), records)
        concat_sp[f"{La}+{Lb}"] = sp
        print(f"  L{La}+L{Lb} hidden: {sp:.4f}", flush=True)

    # hidden + PHS 同层拼接
    print("\n同层 hidden+PHS 拼接:", flush=True)
    best_h = max(h_layer_sp, key=h_layer_sp.get)
    best_s = max(s_layer_sp, key=s_layer_sp.get)
    hp_sp = {}
    for L in LAYERS:
        feat = np.concatenate([hiddens[L], phs_by_layer[L]], axis=1)
        sp = cos_spearman(feat.tolist(), records)
        hp_sp[L] = sp
        print(f"  L{L:2d} hidden+PHS: {sp:.4f}", flush=True)

    results = {
        "n_pairs": len(records),
        "hidden_single": h_layer_sp,
        "phs_single": s_layer_sp,
        "hidden_concat": concat_sp,
        "hidden_phs_concat": hp_sp,
        "best_hidden_layer": best_h,
        "best_phs_layer": best_s,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()