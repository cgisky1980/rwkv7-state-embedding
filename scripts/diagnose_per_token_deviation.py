"""诊断 D：量化「逐 token 残差偏差」是否携带 mean-pool 丢失的标签相关信号。

背景（负结果）：固定 tanh 偏差门控作用在 mean-pool 后的 hidden/state 上无增益。
失败归因之一是「池化破坏了逐 token 偏差信号」。本诊断直接检验该归因：
对每个句子导出*逐 token* 最后一层 FFN 残差 x_t (T, C)，计算 J-lens 偏差门控
g_t = tanh((LN(x_t) - x_t)/2)，对比不同池化策略在句子对余弦-Spearman 上的表现。

若「逐 token 门控后池化」(g_pool) 显著优于「直接 mean-pool」(h_pool)，
说明偏差信号确实存在但被池化破坏 → 值得投入方向 A（逐 token 门控 + 全量重推理）。
若两者接近或 g_pool 更差 → 偏差信号在 0.4B 上不成立，方向 A 不值得投入。

用法 (Windows, 需 MSVC + CUDA):
  run_with_msvc.bat diagnose_per_token_deviation.py [--n-pairs 200] [--max-length 256]

输出: 各池化策略的句子对余弦-Spearman 对照（打印 + 保存 json）。
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
DATA_PATH = PAPER_DIR / "data" / "sts" / "sts_test.jsonl"
OUT_PATH = Path("参考/diagnose_per_token_deviation_results.json")

N_EMBD = 1024


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def spearman_corr(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def layer_norm_vec(x: np.ndarray) -> np.ndarray:
    """对最后一个维度 (C) 做 LN: (x-mean)/std"""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return (x - mean) / std


def extract_per_token(model, tokenizer, texts, max_length):
    """逐 token 导出最后一层 FFN 残差 x_t (T, C)，单序列循环（诊断用）。"""
    from rwkv7 import RWKV_x070_TMix_seq, RWKV_x070_CMix_seq

    z = model.z
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size
    n_embd = model.n_embd
    device = z["emb.weight"].device

    per_token = []  # list of (T, C) float32
    for text in texts:
        tokens = tokenizer.encode(text)[:max_length]
        if len(tokens) == 0:
            tokens = [0]
        T = len(tokens)
        state = model.generate_zero_state(0)

        x = z["emb.weight"][torch.tensor(tokens, device=device)]  # (T, C)
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
            xx = F.layer_norm(x, (n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
            xx = RWKV_x070_CMix_seq(xx, state[0][i], z[ffn + "x_k"], z[ffn + "key.weight"], z[ffn + "value.weight"])
            x = x + xx
        per_token.append(x.float().cpu().numpy())  # (T, C)
    return per_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=1379)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    records = read_jsonl(DATA_PATH)[: args.n_pairs]
    print(f"pair 数: {len(records)}", flush=True)

    print("加载 albatross 0.4B 模型...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(MODEL_PATH, VOCAB_PATH)
    print(f"  加载完成 ({time.time()-t0:.1f}s)", flush=True)

    # 收集所有句子（交错 s1,s2,s1,s2,...）
    sentences = []
    for r in records:
        sentences.append(r["sentence1"])
        sentences.append(r["sentence2"])
    scores = np.array([r["score"] for r in records], dtype=np.float32)

    print(f"逐 token 推理 {len(sentences)} 句 (max_length={args.max_length})...", flush=True)
    t0 = time.time()
    per_token = extract_per_token(model, tokenizer, sentences, args.max_length)
    print(f"  推理完成 ({time.time()-t0:.1f}s)", flush=True)

    # 逐句特征
    h_pool_list = []   # mean pool hidden
    g_pool_list = []   # mean over tokens of g_t (先逐token门控再pool)
    ng_pool_list = []  # 对照: 先 pool 再对 pooled 向量门控 (约等于旧 E1)
    g_last_list = []   # last token gated
    g_absmax_list = []  # absmax over tokens of g_t
    g_shufc_list = []   # 反证: 每 token 内部 C 维打乱再 pool (破坏偏差内部结构)
    rng = np.random.default_rng(42)
    for x in per_token:
        x = x.astype(np.float32)
        g = np.tanh((layer_norm_vec(x) - x) / 2.0)  # (T, C)
        h_pool = x.mean(axis=0)
        h_pool_list.append(h_pool)
        g_pool_list.append(g.mean(axis=0))
        ng_pool_list.append(np.tanh((layer_norm_vec(h_pool.reshape(1, -1))[0] - h_pool) / 2.0))
        g_last_list.append(g[-1])
        g_absmax_list.append(g[np.abs(g).sum(axis=-1).argmax()])
        g_shufc_list.append(np.take_along_axis(g, rng.permutation(g.shape[1])[None, :], axis=1).mean(axis=0).tolist())
        g_shufc_list[-1] = np.asarray(g_shufc_list[-1])

    def cos_spearman(feat_list):
        n = len(records)
        cos = []
        for i in range(n):
            a = feat_list[2 * i]
            b = feat_list[2 * i + 1]
            c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            cos.append(c)
        return spearman_corr(cos, scores)

    results = {
        "n_pairs": len(records),
        "h_pool_mean": cos_spearman(h_pool_list),
        "g_pool_mean": cos_spearman(g_pool_list),
        "g_last_token": cos_spearman(g_last_list),
        "g_absmax": cos_spearman(g_absmax_list),
        "ng_pool": cos_spearman(ng_pool_list),
        "g_shufc": cos_spearman(g_shufc_list),
    }
    labels = {
        "h_pool_mean": "baseline: mean pool hidden",
        "g_pool_mean": "先逐token门控再 mean pool",
        "ng_pool": "对照: 先 mean pool 再门控 (≈旧E1)",
        "g_last_token": "最后 token 门控",
        "g_absmax": "|g| 最大 token 门控",
        "g_shufc": "反证: 每token内部C维打乱再 pool",
    }
    print("\n诊断结果（句子对余弦-Spearman vs 标签）:", flush=True)
    for k, v in results.items():
        if k == "n_pairs":
            continue
        print(f"  {k:15s} ({labels[k]}): {v:.4f}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()