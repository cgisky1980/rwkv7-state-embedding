"""
任务八：Whitening 后处理验证（零训练）

检验 BERT-whitening（中心化 + 去相关）后处理能否提升现有 fusion 投影器的 STS 表现。
whitening 参数仅 fit 在 clean 训练语料（STS-B train + NLI train）上，不接触 dev/test。

变体：
  - baseline : 原始 embedding（L2 归一化）
  - center   : 仅减均值（中心化）
  - whiten-k : 中心化 + 去相关，保留前 k 主成分（k=512/256/128/64）

原理 (Su et al. 2021, BERT-whitening):
  cov = W Λ W^T  →  变换核 = W_k Λ_k^{-1/2}，取特征值最大的前 k 列

运行:
  cd paper/scripts
  uv run --project ../../scripts python 08_whitening.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

torch.set_num_threads(os.cpu_count() or 4)


class MlpProj(nn.Module):
    """与 02_sts_similarity.py 一致 (支持 n_layers)。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.2, n_layers: int = 2):
        super().__init__()
        assert n_layers >= 2
        self.input_norm = nn.BatchNorm1d(input_dim)
        blocks = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            blocks += [nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                       nn.Linear(hidden_dim, hidden_dim)]
        blocks += [nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                   nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return F.normalize(self.net(x), p=2, dim=-1)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return spearmanr(x, y).correlation


def load_split(cache_dir: Path, layer: int, name: str):
    """加载 sts_pair 缓存 (states 可选, clean 缓存 train/nli 含 states)。"""
    p = cache_dir / f"sts_pair_l{layer}_{name}.npz"
    with np.load(p) as z:
        states = z["states"].astype(np.float32, copy=False) if "states" in z.files else None
        hiddens = z["hiddens"].astype(np.float32, copy=False)
        scores = z["scores"].astype(np.float32, copy=False)
    return states, hiddens, scores


def infer_n_layers(sd: dict) -> int:
    """从 state dict 推断 Linear 层数 (旧 checkpoint config 无 n_layers 字段)。

    原版结构为 3 个 Linear (in→h→h→out)，n_layers 参数化下等于 3。
    """
    n_linear = sum(1 for k in sd
                   if k.startswith("net.") and k.endswith(".weight") and sd[k].dim() == 2)
    return max(2, n_linear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection", type=str,
                    default=r"..\cache_python_0.1b_clean\universal_projection_fusion_topk8_l11_infonce_ft.pt")
    ap.add_argument("--cache-dir", type=str, default=r"..\cache_python_0.1b_clean")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--fit-splits", type=str, default="train,nli_train",
                    help="whitening 参数的 fit 集（不得含 dev/test）")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    proj_path = Path(args.projection)
    ckpt = torch.load(proj_path, map_location="cpu")
    cfg = ckpt["config"]
    sel = ckpt["head_selection"]
    heads = list(sel["head_indices"])
    pca_components = np.asarray(sel["pca_components"], dtype=np.float32)
    pca_mean = np.asarray(sel["pca_mean"], dtype=np.float32)
    per = int(sel["head_size"]) ** 2

    n_layers = cfg.get("n_layers") or infer_n_layers(ckpt["state_dicts"][0])
    models = []
    for sd in ckpt["state_dicts"]:
        m = MlpProj(cfg["input_dim"], cfg["hidden_dim"], cfg["output_dim"],
                    cfg["dropout"], n_layers)
        m.load_state_dict(sd)
        m.eval()
        models.append(m.to(args.device))
    print(f"投影器: {proj_path.name} ({len(models)} seeds, {cfg['input_dim']}→{cfg['output_dim']}, "
          f"n_layers={n_layers}, heads={heads})", flush=True)

    def transform(states: np.ndarray, hiddens: np.ndarray) -> np.ndarray:
        """state → Top-K head 拼接 → PCA → 与 hidden 拼接 (fusion)。"""
        st = np.concatenate([states[:, h * per:(h + 1) * per] for h in heads], axis=1)
        st = ((st - pca_mean) @ pca_components.T).astype(np.float32)
        return np.concatenate([hiddens, st], axis=1)

    def embed(feats: np.ndarray) -> np.ndarray:
        """5 seed 集成平均 (与 MTEB 评估路径一致)。"""
        x = torch.from_numpy(np.ascontiguousarray(feats)).to(args.device)
        with torch.no_grad():
            embs = torch.stack([m(x) for m in models]).mean(dim=0)
        return embs.cpu().numpy().astype(np.float32)

    # 各 split embedding
    splits = {}
    for name in ["train", "nli_train", "dev", "test"]:
        states, hiddens, scores = load_split(cache_dir, args.layer, name)
        feats = transform(states, hiddens)
        splits[name] = {"emb": embed(feats), "scores": scores}
        print(f"  {name}: {len(scores)} pairs, emb {splits[name]['emb'].shape}", flush=True)

    # whitening 参数 fit（仅训练语料）
    fit_names = [s.strip() for s in args.fit_splits.split(",")]
    fit_emb = np.concatenate([splits[n]["emb"] for n in fit_names], axis=0)
    mu = fit_emb.mean(axis=0)
    cov = np.cov(fit_emb.T)
    eigval, eigvec = np.linalg.eigh(cov)          # 升序
    order = np.argsort(eigval)[::-1]              # 降序
    eigval = np.clip(eigval[order], 1e-8, None)
    eigvec = eigvec[:, order]
    print(f"whitening fit: {'+'.join(fit_names)} = {fit_emb.shape[0]} 句", flush=True)

    def evaluate(name: str, variant: str) -> float:
        e = splits[name]["emb"].copy()
        if variant == "center":
            e = e - mu
        elif variant.startswith("whiten-"):
            k = int(variant.split("-")[1])
            W = eigvec[:, :k] / np.sqrt(eigval[:k])
            e = (e - mu) @ W
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        s1, s2 = e[0::2], e[1::2]
        cos = (s1 * s2).sum(axis=1)
        return spearman_corr(cos, splits[name]["scores"])

    out_dim = cfg["output_dim"]
    variants = ["baseline", "center", f"whiten-{out_dim}", "whiten-256", "whiten-128", "whiten-64"]
    print("\n结果表 (Spearman):")
    print(f"{'变体':<16}{'dev':>10}{'test':>10}")
    for v in variants:
        print(f"{v:<16}{evaluate('dev', v):>10.4f}{evaluate('test', v):>10.4f}", flush=True)


if __name__ == "__main__":
    main()
