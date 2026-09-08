# Retrain + test B/C — Windows / local
$env:DATA_DIR = if ($env:DATA_DIR) { $env:DATA_DIR } else { "D:\CAFA6" }
$env:DGL_CUDA = if ($env:DGL_CUDA) { $env:DGL_CUDA } else { "1" }

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python scripts/retrain_fusion_bc.py --profile match_a @args
