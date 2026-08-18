# Isaac Lab 環境アクティベーション（PowerShell）
# 使い方:  . D:\IsaacStack\activate_isaac.ps1
# venv を有効化し、Omniverse/Kit のキャッシュを D: に退避して C: を保護する。

$ErrorActionPreference = "Stop"
$Root = "D:\IsaacStack"

# --- venv ---
& "$Root\env_isaaclab\Scripts\Activate.ps1"

# --- Omniverse EULA 非対話受諾（これが無いと import で対話プロンプトが出る）---
$env:OMNI_KIT_ACCEPT_EULA = "YES"

# --- キャッシュ/データを D: へ ---
$env:OMNI_KIT_ALLOW_ROOT = "1"
# Kit / Omniverse shader & asset cache を D: に向ける（C: の AppData 逼迫を回避）
$env:OMNI_CACHE_DIR     = "$Root\cache\ov"
$env:OMNI_DATA_DIR      = "$Root\cache\ov_data"
$env:OMNI_LOGS_DIR      = "$Root\cache\ov_logs"
$env:XDG_CACHE_HOME     = "$Root\cache\xdg"
$env:HF_HOME            = "$Root\cache\huggingface"
$env:TORCH_HOME         = "$Root\cache\torch"

$cacheDirs = @($env:OMNI_CACHE_DIR, $env:OMNI_DATA_DIR, $env:OMNI_LOGS_DIR, $env:XDG_CACHE_HOME, $env:HF_HOME, $env:TORCH_HOME)
foreach ($d in $cacheDirs) {
    if ($d -and -not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

Write-Host "[Isaac] venv active: $((Get-Command python).Source)"
Write-Host "[Isaac] cache -> $Root\cache"
Write-Host "[Isaac] IsaacLab: $Root\IsaacLab"
Write-Host "[Isaac] 検証: python $Root\verify_factory.py --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 16 --steps 200 --headless"
Write-Host "[Isaac] !! pip install するときは必ず -c $Root\constraints.txt を付けて GPU torch / numpy<2 を保護する"
