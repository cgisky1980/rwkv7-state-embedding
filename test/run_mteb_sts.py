"""用 0.1B RWKV 跑 mteb 全部英语 STS 任务。

支持裸特征 (hidden / state) 与训练好的投影器 (universal_projection) 两条路径。

用法 (需 MSVC 环境):
  # 裸 hidden 特征
  run_with_msvc.bat run_mteb_sts.py --feature hidden --layer 11
  # 用训练好的 hidden 投影器
  run_with_msvc.bat run_mteb_sts.py --feature hidden --layer 11 --projection universal_projection_l11.pt
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper" / "scripts" / "lib"))
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from albatross_wrapper import load_model, extract_features_batch

MODEL = r"C:\work\niceui\rwkv-router\paper\models\rwkv7-g1d-0.1b-20260129-ctx8192.pth"
VOCAB = r"C:\work\niceui\rwkv-router\paper\scripts\lib\rwkv_vocab_v20230424.txt"


class MlpProj(nn.Module):
    """与 02_sts_similarity.py 一致的 MLP 投影器 (hidden → 语义空间, L2 归一化)."""

    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=512, dropout=0.1, n_layers=2):
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

    def forward(self, x):
        x = self.input_norm(x)
        return F.normalize(self.net(x), p=2, dim=-1)


def load_projection(path: Path):
    """加载 universal_projection .pt, 返回 (models, config, head_selection). 多 seed 集成平均.

    head_selection (Top-K Head 模式): {head_indices, head_size, pca_components, pca_mean}
    应用顺序: state → 选 head 列拼接 → (x - mean) @ components.T → MLP
    """
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    models = []
    for sd in ckpt["state_dicts"]:
        m = MlpProj(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"],
                    output_dim=config["output_dim"], dropout=config["dropout"],
                    n_layers=config.get("n_layers", 2))
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    head_sel = ckpt.get("head_selection")
    if head_sel is not None:
        head_sel = {
            "head_indices": head_sel["head_indices"],
            "head_size": head_sel["head_size"],
            "pca_components": torch.from_numpy(head_sel["pca_components"]).float(),
            "pca_mean": torch.from_numpy(head_sel["pca_mean"]).float(),
        }
    return models, config, head_sel


class RWKVEmbedder:
    def __init__(self, feature="hidden", layer=11, batch_size=8, max_length=512, projection_path=None, device="cpu", prompt_style=""):
        t0 = time.time()
        self.model, self.tok = load_model(Path(MODEL), Path(VOCAB))
        self.feature = feature
        self.layer = layer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.prompt_style = prompt_style
        self.projection = None
        self.head_selection = None
        self.tag = f"{feature}-{layer}" + (f"-{prompt_style}" if prompt_style else "")

        if projection_path:
            proj_path = Path(projection_path)
            if not proj_path.is_absolute():
                proj_path = Path(r"C:\work\niceui\rwkv-router\paper\cache_python_0.1b") / proj_path
            self.projection, proj_config, self.head_selection = load_projection(proj_path)
            self.projection = [m.to(self.device) for m in self.projection]
            if self.head_selection is not None:
                self.head_selection["pca_components"] = self.head_selection["pca_components"].to(self.device)
                self.head_selection["pca_mean"] = self.head_selection["pca_mean"].to(self.device)
            self.embed_dim = proj_config["output_dim"]
            self.tag = f"proj_{proj_path.stem}"
            sel = (f", heads={self.head_selection['head_indices']}"
                   if self.head_selection else "")
            print(f"[RWKVEmbedder] 加载投影器: {proj_path} "
                  f"({len(self.projection)} seeds, {proj_config['input_dim']}→{self.embed_dim}{sel})", flush=True)
        else:
            n_head = self.model.n_head
            hs = self.model.head_size
            self.embed_dim = n_head * hs * hs if feature == "state" else self.model.n_embd

        from mteb.models.model_meta import ModelMeta
        self.mteb_model_meta = ModelMeta.create_empty({
            "name": f"rwkv7-g1d-0.1b-{self.tag}",
            "similarity_fn_name": "cosine",
            "embed_dim": self.embed_dim,
            "revision": "0",
            "release_date": "2026-01-29",
            "framework": ["PyTorch"],
            "languages": ["eng-Latn"],
            "open_weights": True,
            "license": "apache-2.0",
            "n_parameters": 107_000_000,
        })
        self.similarity_fn_name = "cosine"
        print(f"[RWKVEmbedder] feature={feature} layer={layer} tag={self.tag} "
              f"dim={self.embed_dim} load={time.time()-t0:.1f}s", flush=True)

    def encode(self, inputs, **kwargs):
        out = []
        for batch in inputs:
            texts = list(batch["text"])
            states, hiddens = extract_features_batch(
                self.model, self.tok, texts,
                batch_size=self.batch_size, max_length=self.max_length, layer=self.layer,
                prompt_style=self.prompt_style,
            )
            hd = torch.from_numpy(np.ascontiguousarray(hiddens)).float().to(self.device)
            if self.feature == "hidden":
                feats = hd
            else:
                st = torch.from_numpy(np.ascontiguousarray(states)).float().to(self.device)
                if self.head_selection is not None:
                    # Top-K Head / 整层 PCA: 选列拼接 → PCA 变换
                    hs = self.head_selection
                    per = hs["head_size"] * hs["head_size"]
                    st = torch.cat(
                        [st[:, h * per:(h + 1) * per] for h in hs["head_indices"]], dim=1
                    )
                    st = (st - hs["pca_mean"]) @ hs["pca_components"].t()
                # fusion: hidden ⊕ state 变换特征 拼接
                feats = st if self.feature == "state" else torch.cat([hd, st], dim=1)
            if self.projection:
                with torch.no_grad():
                    emb = torch.stack([m(feats) for m in self.projection]).mean(dim=0)
            else:
                emb = feats
            out.append(emb.cpu().numpy().astype(np.float32))
        if not out:
            return np.zeros((0, self.embed_dim), np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    def similarity(self, e1, e2):
        a = torch.from_numpy(np.asarray(e1, np.float32))
        b = torch.from_numpy(np.asarray(e2, np.float32))
        import torch.nn.functional as F
        a = F.normalize(a, p=2, dim=-1); b = F.normalize(b, p=2, dim=-1)
        return a @ b.t()

    def similarity_pairwise(self, e1, e2):
        a = torch.from_numpy(np.asarray(e1, np.float32))
        b = torch.from_numpy(np.asarray(e2, np.float32))
        import torch.nn.functional as F
        a = F.normalize(a, p=2, dim=-1); b = F.normalize(b, p=2, dim=-1)
        return (a * b).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["hidden", "state", "fusion"], default="hidden",
                    help="fusion = hidden ⊕ Top-K state PCA 变换后拼接 (需投影器含 head_selection)")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--projection", type=str, default="",
                    help="投影器 .pt 文件名 (置于 cache_python_0.1b, 如 universal_projection_l11.pt)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--prompt-style", type=str, default="", choices=["", "eol", "io", "qa", "ins"],
                    help="文本模板 (与 extract_features.py 一致; ins=RWKV 官方 Instruction 格式)")
    args = ap.parse_args()

    import mteb
    tasks = mteb.get_tasks(task_types=["STS"], languages=["eng"])
    names = [t.metadata.name for t in tasks]
    print("待跑 STS 任务:", names, flush=True)

    emb = RWKVEmbedder(feature=args.feature, layer=args.layer,
                       batch_size=args.batch_size, max_length=args.max_length,
                       projection_path=args.projection, device=args.device,
                       prompt_style=args.prompt_style)
    cache = mteb.cache.ResultCache(rf"C:\work\niceui\rwkv-router\test\mteb_out\{emb.tag}")
    results = mteb.evaluate(emb, tasks, cache=cache)

    summary = {"model": f"rwkv7-g1d-0.1b-{emb.tag}", "feature": args.feature, "layer": args.layer,
               "projection": args.projection, "tasks": {}}
    for r in results:
        for split, scores_list in r.scores.items():
            for sc in scores_list:
                main = sc.get("main_score")
                summary["tasks"][r.task_name + "/" + split] = {
                    "main_score": main,
                    "spearman_cosine": sc.get("cosine_spearman"),
                    "pearson_cosine": sc.get("cosine_pearson"),
                    "spearman_euclidean": sc.get("euclidean_spearman"),
                }
    out = rf"C:\work\niceui\rwkv-router\test\mteb_out\summary_{emb.tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("[DONE] saved:", out, flush=True)


if __name__ == "__main__":
    main()