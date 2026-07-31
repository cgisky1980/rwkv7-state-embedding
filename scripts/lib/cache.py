"""特征缓存读写 helper。

支持两种格式:
1. .npz (Python, 推荐): 由 extract_features.py 生成，纯 Python 路径
2. .bin  (Rust legacy): 由 Rust 端 forward_sts_train.rs 等生成

两种格式都返回 (states, hiddens) 元组，states 形状 (N, state_dim)，hiddens 形状 (N, hidden_dim)。
STS 任务中 states/hiddens 按句子对交错排列: [s1_p0, s2_p0, s1_p1, s2_p1, ...]。
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np


# ============================================================
# .npz 格式 (Python 路径)
# ============================================================
def save_npz(
    path: Path,
    states: np.ndarray,
    hiddens: np.ndarray,
    extra: dict | None = None,
) -> None:
    """保存为 .npz 格式 (float16 节省空间)。

    Args:
        path: 输出路径 (.npz)
        states: (N, state_dim) float32 / float16
        hiddens: (N, hidden_dim) float32 / float16
        extra: 额外数组 (如 labels, scores)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "states": states.astype(np.float16, copy=False),
        "hiddens": hiddens.astype(np.float16, copy=False),
    }
    if extra:
        for k, v in extra.items():
            arrays[k] = np.asarray(v)
    np.savez(path, **arrays)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  saved: {path} ({size_mb:.1f} MB)")


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载 .npz 缓存，返回 (states, hiddens) float32。"""
    path = Path(path)
    with np.load(path) as data:
        states = data["states"].astype(np.float32, copy=False)
        hiddens = data["hiddens"].astype(np.float32, copy=False)
    return states, hiddens


# ============================================================
# .bin 格式 (Rust legacy, 只读)
# ============================================================
def load_bin_pair(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载 Rust 生成的 pair 缓存 (magic=b'RWKVSHP')。

    Header (23 bytes): magic(7) + n(u32) + state_dim(u32) + hidden_dim(u32) + padding(4)
    Body: n × (state_dim + hidden_dim) × f32

    Returns:
        states: (N, state_dim) float32
        hiddens: (N, hidden_dim) float32
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    if data[0:7] != b"RWKVSHP":
        raise ValueError(f"缓存 magic 错误: {data[0:7]} (期望 b'RWKVSHP')")
    n, state_dim, hidden_dim = struct.unpack("<III", data[7:19])
    body = data[23:]  # body 在 offset 23 (4 字节 padding)
    expected = n * (state_dim + hidden_dim) * 4
    if len(body) != expected:
        raise ValueError(f"缓存 body 大小不匹配: 期望 {expected}, 实际 {len(body)}")
    arr = np.frombuffer(body, dtype=np.float32).reshape(n, state_dim + hidden_dim)
    return arr[:, :state_dim].copy(), arr[:, state_dim:].copy()


# ============================================================
# 统一入口
# ============================================================
def load_cache(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """根据后缀自动选择加载方式。

    .npz → load_npz (Python 路径)
    .bin → load_bin_pair (Rust legacy)
    """
    path = Path(path)
    if path.suffix == ".npz":
        return load_npz(path)
    elif path.suffix == ".bin":
        return load_bin_pair(path)
    else:
        raise ValueError(f"不支持的缓存格式: {path.suffix} (仅支持 .npz / .bin)")
