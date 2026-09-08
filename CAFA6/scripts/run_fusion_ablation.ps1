# Train + eval ablation fusion — Windows
$env:DATA_DIR = if ($env:DATA_DIR) { $env:DATA_DIR } else { "D:\CAFA6" }
$env:DGL_CUDA = if ($env:DGL_CUDA) { $env:DGL_CUDA } else { "1" }

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python scripts/run_fusion_ablation.py @args
