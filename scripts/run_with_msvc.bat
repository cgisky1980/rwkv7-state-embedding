@echo off
:: Activate MSVC environment, then run Python script
:: Usage: run_with_msvc.bat <python_script.py> [args...]
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo MSVC environment activation failed
    exit /b 1
)
set TORCH_EXTENSIONS_DIR=C:\Users\cgisk\AppData\Local\Temp\torch_ext_nodither
uv run --project c:\work\niceui\rwkv-router\scripts python %*
