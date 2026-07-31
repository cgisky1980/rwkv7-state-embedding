#!/usr/bin/env python3
"""下载 RWKV-7 0.4B 模型和 Albatross reference 代码。

用法:
    python 00_setup.py

下载内容:
    1. BlinkDL/rwkv7-g1 仓库中的 rwkv7-g1d-0.4b-20260210-ctx8192.pth (约 0.8GB)
    2. 复制 Albatross reference 代码 (rwkv7.py + cuda/)
    3. 复制 tokenizer (rwkv_vocab_v20230424.txt)
"""
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
MODEL_DIR = PAPER_DIR / "models"
ALBATROSS_SRC = PAPER_DIR / "albatross_src" / "_ref_slower_" / "reference"


def download_model():
    """从 HuggingFace 下载 RWKV-7 0.4B 模型。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_file = MODEL_DIR / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"

    if model_file.exists():
        size_gb = model_file.stat().st_size / (1024 ** 3)
        print(f"[OK] 模型已存在: {model_file} ({size_gb:.2f} GB)")
        return model_file

    print("[1/3] 下载 RWKV-7 0.4B 模型 (BlinkDL/rwkv7-g1)...")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id="BlinkDL/rwkv7-g1",
        filename="rwkv7-g1d-0.4b-20260210-ctx8192.pth",
        local_dir=str(MODEL_DIR),
    )
    print(f"[OK] 模型下载完成: {downloaded}")
    return model_file


def copy_reference_code():
    """复制 Albatross reference 代码到 scripts/lib/。"""
    print("[2/3] 复制 Albatross reference 代码...")
    lib_dir = SCRIPT_DIR / "lib"
    if lib_dir.exists():
        shutil.rmtree(lib_dir)
    lib_dir.mkdir(parents=True)

    if not ALBATROSS_SRC.exists():
        print(f"[ERROR] Albatross 源码不存在: {ALBATROSS_SRC}")
        print("请先运行: git clone --depth 1 https://github.com/BlinkDL/Albatross.git albatross_src")
        sys.exit(1)

    for name in ["rwkv7.py", "__init__.py", "utils.py", "rwkv_vocab_v20230424.txt"]:
        src = ALBATROSS_SRC / name
        if src.exists():
            shutil.copy2(src, lib_dir / name)
            print(f"  复制 {name}")

    cuda_dir = lib_dir / "cuda"
    cuda_dir.mkdir(parents=True)
    src_cuda = ALBATROSS_SRC / "cuda"
    if src_cuda.exists():
        for name in os.listdir(src_cuda):
            shutil.copy2(src_cuda / name, cuda_dir / name)
            print(f"  复制 cuda/{name}")

    print(f"[OK] reference 代码已复制到 {lib_dir}")
    return lib_dir


def check_environment():
    """检查运行环境。"""
    print("[3/3] 检查环境...")
    try:
        import torch
        print(f"  torch: {torch.__version__}")
        print(f"  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[ERROR] 未安装 torch，请先安装: uv pip install torch")
        sys.exit(1)

    import subprocess
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        version_line = [l for l in result.stdout.split("\n") if "release" in l]
        if version_line:
            print(f"  nvcc: {version_line[0].strip()}")
    else:
        print("[WARN] 未找到 nvcc，CUDA 扩展编译可能失败")

    vocab = SCRIPT_DIR / "lib" / "rwkv_vocab_v20230424.txt"
    model = MODEL_DIR / "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
    if vocab.exists():
        print(f"  tokenizer: OK")
    if model.exists():
        size_gb = model.stat().st_size / (1024 ** 3)
        print(f"  model: OK ({size_gb:.2f} GB)")

    print("\n[DONE] 环境配置完成！")
    print("下一步: python extract_features.py")


def main():
    download_model()
    copy_reference_code()
    check_environment()


if __name__ == "__main__":
    main()
