"""flag_gems stub: 让 albatross rwkv7.py 能 import。

albatross faster_251101 的 reference/rwkv7.py 在顶部 import flag_gems，
但只有 CMix_one (单 token 推理) 用了 flag_gems.rwkv_mm_sparsity。
我们只用 forward_seq_batch (并发推理)，所以 stub 掉即可。

triton 在 Windows 上不支持，无法安装真正的 flag_gems。
"""
import sys
import types

if "flag_gems" not in sys.modules:
    mod = types.ModuleType("flag_gems")
    # torch.ops.flag_gems.rwkv_mm_sparsity 的 stub (未被调用)
    mod.rwkv_mm_sparsity = lambda k, V_: k @ V_
    mod.rwkv_ka_fusion = lambda *a, **kw: a
    sys.modules["flag_gems"] = mod
