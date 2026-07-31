#!/usr/bin/env python3
"""纯 PyTorch 实现的 RWKV-7 推理，不依赖 CUDA 扩展。

从 albatross CUDA 代码推导的 WKV 公式:
    w = exp(-0.6065306597 * w_raw)  # 0.6065306597 = exp(-0.5)
    sa[k] = sum_v state[k, v] * a[v]
    state[k, v] = state[k, v] * w[v] + (sa[k] * b[v] + k[v] * v[k])
    y[k] = sum_v state[k, v] * r[v]

支持:
    - 加载 BlinkDL .pth 模型
    - 批量并发推理 (forward_batch)
    - 提取 WKV state (任意层)
    - 提取 hidden state (mean pooling 或最后 token)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional

DTYPE = torch.float16
HEAD_SIZE = 64
DECAY_CONST = 0.6065306597  # exp(-0.5)


class RWKV7Pure:
    """纯 PyTorch RWKV-7 模型，支持 state 和 hidden 提取。"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        torch.set_grad_enabled(False)

        # 加载权重
        print(f"加载模型: {model_path}")
        z = torch.load(model_path, map_location="cpu", mmap=True)

        self.n_head, self.head_size = z["blocks.0.att.r_k"].shape
        self.n_embd = self.n_head * self.head_size
        assert self.head_size == HEAD_SIZE

        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            kk = k.split(".")
            if "att.g1" in k or "att.g2" in k or "att.a1" in k or "att.a2" in k:
                z[k] = z[k].t()
            if "att.w1" in k or "att.w2" in k:
                z[k] = z[k].t()
            if "att.v1" in k or "att.v2" in k:
                z[k] = z[k].t()
            if "ffn.value.weight" in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE, device=device)
            if k.endswith("att.r_k"):
                z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            if kk[0] == "blocks":
                max_layer = max(max_layer, int(kk[1]))

        self.n_layer = max_layer + 1
        self.z = z

        # 预处理 embedding (layer norm with ln0)
        z["emb.weight"] = F.layer_norm(
            z["emb.weight"], (self.n_embd,),
            weight=z["blocks.0.ln0.weight"],
            bias=z["blocks.0.ln0.bias"]
        )
        z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
        z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
        z["blocks.0.att.v2"] = z["blocks.0.att.a2"]

        print(f"模型加载完成: n_layer={self.n_layer}, n_embd={self.n_embd}, "
              f"n_head={self.n_head}, head_size={self.head_size}")

    def zero_state(self, batch_size: int = 1) -> List[torch.Tensor]:
        """创建初始空 state。

        Returns:
            state list: [att_x_prev, att_kv, ffn_x_prev] * n_layer + [elapsed_t]
            - att_kv: (B, H, N, N) WKV state [k_index, v_index] — float32 避免溢出
            - att_x_prev: (B, n_embd) shift state
            - ffn_x_prev: (B, n_embd) FFN shift state
        """
        B = batch_size
        state = []
        for i in range(self.n_layer):
            state.append(torch.zeros(B, self.n_embd, dtype=DTYPE, device=self.device))  # att_x_prev
            state.append(torch.zeros(B, self.n_head, self.head_size, self.head_size, dtype=torch.float32, device=self.device))  # att_kv (float32!)
            state.append(torch.zeros(B, self.n_embd, dtype=DTYPE, device=self.device))  # ffn_x_prev
        state.append(torch.zeros(B, dtype=torch.int32, device=self.device))  # elapsed_t
        return state

    def _wkv_forward(self, r, w, k, v, a, b, state):
        """纯 PyTorch WKV 计算 (Delta Rule)。

        Args:
            r, w, k, v, a, b: (B, T, H, N)
            state: (B, H, N, N) -- [k_index, v_index]

        Returns:
            y: (B, T, H, N) -- output
            state: (B, H, N, N) -- updated state
        """
        B, T, H, N = r.shape
        y_list = []

        for t in range(T):
            r_t = r[:, t]  # (B, H, N)
            w_t = w[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            a_t = a[:, t]
            b_t = b[:, t]

            # w[k] = exp(-0.6065306597 * sigmoid(w_raw[k]))
            w_dec = torch.exp(-DECAY_CONST * w_t.float()).to(DTYPE)  # (B, H, N) = w[k]

            # sa[v] = sum_k state[k, v] * a[k]  (对 dim=-2 即 k 维求和)
            sa = (state.float() * a_t[:, :, :, None].float()).sum(dim=-2).to(DTYPE)  # (B, H, N) = sa[v]

            # state[k, v] = state[k, v] * w[k] + (sa[v] * b[k] + k[k] * v[v])
            state = state * w_dec[:, :, :, None] + (
                sa[:, :, None, :] * b_t[:, :, :, None] +
                k_t[:, :, :, None] * v_t[:, :, None, :]
            )

            # y[v] = sum_k state[k, v] * r[k]  (对 dim=-2 即 k 维求和)
            y_t = (state.float() * r_t[:, :, :, None].float()).sum(dim=-2).to(DTYPE)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # (B, T, H, N)
        return y, state

    def _tmix(self, layer_id, x, x_prev, v_first, state, B, T):
        """Time mixing (attention) for one layer。

        Args:
            x: (B, T, C) input
            x_prev: (B, C) previous input (shift state)
            v_first: (B, T, C) first layer value
            state: (B, H, N, N) WKV state

        Returns:
            xx: (B, T, C) output
            v_first: (B, T, C) updated v_first
            x_prev: (B, C) updated x_prev
            state: (B, H, N, N) updated state
        """
        z = self.z
        H, N = self.n_head, self.head_size
        C = self.n_embd
        att = f"blocks.{layer_id}.att."

        # Shift
        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1, :]), dim=1) - x
        x_prev = x[:, -1, :].clone()

        # Projections
        xr = x + xx * z[att + "x_r"]
        xw = x + xx * z[att + "x_w"]
        xk = x + xx * z[att + "x_k"]
        xv = x + xx * z[att + "x_v"]
        xa = x + xx * z[att + "x_a"]
        xg = x + xx * z[att + "x_g"]

        r = F.linear(xr, z[att + "receptance.weight"])  # (B, T, H*N)
        # NOTE: w 必须经过 sigmoid，与 albatross CUDA 参考一致 (rwkv7.py:354)
        # 未 sigmoid 的 w 取值范围未受控，exp(-0.6065 * w) 会溢出到 inf，使 state 变 NaN
        w_pre = F.linear(torch.tanh(F.linear(xw, z[att + "w1"])), z[att + "w2"])
        w = torch.sigmoid(z[att + "w0"] + w_pre)
        k = F.linear(xk, z[att + "key.weight"])
        v = F.linear(xv, z[att + "value.weight"])
        a = torch.sigmoid(F.linear(F.linear(xa, z[att + "a1"]), z[att + "a2"], bias=z[att + "a0"]))
        g = F.linear(torch.sigmoid(F.linear(xg, z[att + "g1"])), z[att + "g2"])

        # v_first (residual value) — 在 reshape 之前操作，保持 (B, T, H*N)
        if layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                F.linear(F.linear(xv, z[att + "v1"]), z[att + "v2"], bias=z[att + "v0"])
            )

        # Reshape to (B, T, H, N)
        r = r.view(B, T, H, N)
        w = w.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        a = a.view(B, T, H, N)

        # Reshape weight vectors to (H, N) for broadcasting
        k_k = z[att + "k_k"].view(H, N)
        k_a = z[att + "k_a"].view(H, N)
        r_k = z[att + "r_k"].view(H, N)

        # Normalize k
        kk = F.normalize(k * k_k, dim=-1, p=2.0)  # (B, T, H, N)
        k = k * (1 + (a - 1) * k_a)
        kka = kk * a
        neg_kk = -kk

        # WKV forward (pure PyTorch)
        xx_wkv, state = self._wkv_forward(r, w, k, v, neg_kk, kka, state)
        xx_wkv = xx_wkv.view(B, T, H * N)

        # Group norm
        xx_wkv = F.group_norm(
            xx_wkv.view(B * T, H * N), num_groups=H,
            weight=z[att + "ln_x.weight"], bias=z[att + "ln_x.bias"], eps=64e-5
        ).view(B, T, H * N)

        # Add residual: (r * k * r_k) * v
        rk_term = ((r * k * r_k).sum(dim=-1, keepdim=True) * v).view(B, T, H * N)
        xx_wkv = xx_wkv + rk_term

        # Output projection with gating
        xx = F.linear(xx_wkv * g, z[att + "output.weight"])
        return xx, v_first, x_prev, state

    def _cmix(self, x, x_prev, layer_id, B, T):
        """Channel mixing (FFN) for one layer."""
        z = self.z
        ffn = f"blocks.{layer_id}.ffn."

        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1, :]), dim=1) - x
        x_prev = x[:, -1, :].clone()

        k = x + xx * z[ffn + "x_k"]
        k = torch.relu(F.linear(k, z[ffn + "key.weight"])) ** 2
        return k @ z[ffn + "value.weight"], x_prev

    def forward_batch(
        self,
        tokens: torch.Tensor,
        state: List[torch.Tensor],
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """批量并发推理。

        Args:
            tokens: (B, T) token IDs
            state: from zero_state()
            return_hidden: 如果 True，返回 hidden state (mean pooling)

        Returns:
            logits: (B, V) 或 (B, T, V) 如果 full_output
            hidden: (B, C) mean pooling hidden state (如果 return_hidden=True)

        Notes:
            hidden 提取与 Rust real_backbone.rs 的 forward_layer_with_hidden 一致:
            - 只累积最后一层 (PostFfn) 的 x (即 FFN 输出 x+xx 后的结果)
            - 累积所有 token 的 x，最后除以 num_tokens 得到 mean pooling
            - 不累积所有层！只最后一层！
        """
        z = self.z
        B, T = tokens.shape
        C = self.n_embd
        H, N = self.n_head, self.head_size

        # Embedding
        x = z["emb.weight"][tokens]  # (B, T, C)

        v_first = torch.empty_like(x)
        # 累积最后一层的 hidden (sum over tokens, 最后除以 T)
        hidden_sum = torch.zeros(B, C, dtype=torch.float32, device=self.device) if return_hidden else None

        for i in range(self.n_layer):
            bbb = f"blocks.{i}."

            # Layer norm
            xx = F.layer_norm(x, (C,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])

            # Time mixing
            xx, v_first, state[3 * i], state[3 * i + 1] = self._tmix(
                i, xx, state[3 * i], v_first, state[3 * i + 1], B, T
            )
            x = x + xx

            # Channel mixing
            xx = F.layer_norm(x, (C,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
            xx, state[3 * i + 2] = self._cmix(xx, state[3 * i + 2], i, B, T)
            x = x + xx

            # 只在最后一层累积 hidden (与 Rust hook PostFfn(last_layer) 一致)
            # Rust hook 在 PostFfn 处 add frame.buffer.x，累积所有 token
            if return_hidden and i == self.n_layer - 1:
                hidden_sum += x.float().sum(dim=1)  # sum over tokens (最后除以 T)

        # Final layer norm
        x = F.layer_norm(x, (C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])

        # Hidden state (mean pooling over tokens of last layer)
        hidden = None
        if return_hidden:
            hidden = (hidden_sum / T).to(DTYPE)  # (B, C)

        # Logits
        logits = F.linear(x[:, -1, :], z["head.weight"])  # (B, V)
        state[3 * self.n_layer] += T

        return logits, hidden

    def get_wkv_state(self, state: List[torch.Tensor], layer: int) -> torch.Tensor:
        """提取指定层的 WKV state。

        Args:
            state: from forward_batch
            layer: 层索引 (0-based)

        Returns:
            wkv: (B, H, N, N) -- [k_index, v_index]
        """
        return state[3 * layer + 1]

    def forward_extract(
        self,
        tokens: torch.Tensor,
        state: List[torch.Tensor],
        wkv_layer: int = 12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理并提取 hidden state 和 WKV state。

        Args:
            tokens: (B, T) token IDs
            state: from zero_state()
            wkv_layer: 提取哪层的 WKV state

        Returns:
            hidden: (B, C) mean pooling hidden state
            wkv: (B, H, N, N) WKV state at specified layer
        """
        logits, hidden = self.forward_batch(tokens, state, return_hidden=True)
        wkv = self.get_wkv_state(state, wkv_layer)
        return hidden, wkv
