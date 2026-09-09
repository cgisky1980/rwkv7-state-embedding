"""最小验证: 0.1B 模型加载 + 单次特征提取, 确认 MTEB embedder 通路的先决条件。"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper" / "scripts" / "lib"))
import numpy as np

from albatross_wrapper import load_model, extract_features_batch

MODEL = r"C:\work\niceui\rwkv-router\paper\models\rwkv7-g1d-0.1b-20260129-ctx8192.pth"
VOCAB = r"C:\work\niceui\rwkv-router\paper\scripts\lib\rwkv_vocab_v20230424.txt"

t0 = time.time()
m, tok = load_model(Path(MODEL), Path(VOCAB))
print("model params:", m.n_layer, m.n_embd, m.n_head, m.head_size, flush=True)
print("device:", m.z['emb.weight'].device, "load", round(time.time()-t0, 1), "s", flush=True)

texts = [
    "A man is playing guitar.",
    "A musician is playing an instrument.",
    "The stock market went up today.",
]
states, hiddens = extract_features_batch(m, tok, texts, batch_size=8, max_length=256, layer=11)
print("states:", states.shape, states.dtype, flush=True)
print("hiddens:", hiddens.shape, hiddens.dtype, flush=True)
print("state_dim = n_head*head_size^2 =", m.n_head * m.head_size ** 2, flush=True)
print("OK", flush=True)