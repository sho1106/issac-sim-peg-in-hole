#Requires -Version 5.1
<#
.SYNOPSIS
  上流（IsaacLab / TacEx）を clone し、このリポジトリのパッチを当てて Isaac スタックを組む。

.DESCRIPTION
  external\UPSTREAM.lock に固定した commit を checkout し、patches\ 配下を適用する。
  上流は両方とも公開リポジトリなので、このリポジトリを clone するだけで再現できる
  （NAS も追加のリポジトリも要らない）。

  冪等: 既に clone 済み・適用済みなら黙って飛ばす。途中で失敗しても再実行できる。

.PARAMETER StackRoot
  スタックの置き場所。既定 D:\IsaacStack。
  **短いパスにすること** — IsaacLab は相対 144 文字のファイルを含み、深い階層に置くと
  checkout が Filename too long で止まる（README 3.1）。

.PARAMETER SkipVerify
  パッチの sha256 照合を飛ばす。既定では照合し、不一致なら中止する。

.EXAMPLE
  .\scripts\setup_stack.ps1
  .\scripts\setup_stack.ps1 -StackRoot E:\IsaacStack
#>
[CmdletBinding()]
param(
    [string]$StackRoot = "D:\IsaacStack",
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LockFile = Join-Path $RepoRoot "external\UPSTREAM.lock"

function Write-Step($msg) { Write-Host "`n[setup] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  OK   $msg" -ForegroundColor Green }
function Write-Skip($msg) { Write-Host "  skip $msg" -ForegroundColor DarkGray }
function Fail($msg)       { Write-Host "  NG   $msg" -ForegroundColor Red; exit 1 }

# ---- 前提の確認 -------------------------------------------------------------
Write-Step "前提の確認"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git が見つからない" }
if (-not (Test-Path $LockFile)) { Fail "UPSTREAM.lock が無い: $LockFile" }

# 長いパス対策。IsaacLab の checkout がこれ無しで止まった実績がある。
$longPaths = (git config --global core.longpaths)
if ($longPaths -ne "true") {
    Write-Host "  core.longpaths を true にします（IsaacLab の checkout に必要）" -ForegroundColor Yellow
    git config --global core.longpaths true
}
Write-Ok "git / core.longpaths"

# ---- UPSTREAM.lock を読む ---------------------------------------------------
Write-Step "UPSTREAM.lock を読む"
$section = ""
$upstreams = @{}
$patches = @{}
foreach ($line in Get-Content $LockFile -Encoding UTF8) {
    $t = $line.Trim()
    if ($t -match '^\[(.+)\]$') {
        $section = $Matches[1]
        $upstreams[$section] = @{}
        $patches[$section] = New-Object System.Collections.ArrayList
        continue
    }
    if ($t -eq "" -or $t.StartsWith("#")) { continue }
    if ($t -match '^patch\s*=\s*(\S+)') {
        [void]$patches[$section].Add(@{ path = $Matches[1]; sha256 = $null })
        continue
    }
    if ($t -match '^sha256\s+([0-9a-f]{64})') {
        $patches[$section][-1].sha256 = $Matches[1]
        continue
    }
    if ($t -match '^(\w+)\s*=\s*(.+?)(\s+#.*)?$') {
        $upstreams[$section][$Matches[1]] = $Matches[2].Trim()
    }
}
Write-Ok "上流 $($upstreams.Keys.Count) 件"

# ---- パッチの sha256 照合 ---------------------------------------------------
if (-not $SkipVerify) {
    Write-Step "パッチの sha256 照合"
    foreach ($name in $patches.Keys) {
        foreach ($p in $patches[$name]) {
            if (-not $p.sha256) { continue }
            $full = Join-Path $RepoRoot $p.path
            if (-not (Test-Path $full)) { Fail "パッチが無い: $($p.path)" }
            $got = (Get-FileHash $full -Algorithm SHA256).Hash.ToLower()
            if ($got -ne $p.sha256) {
                Fail "sha256 不一致: $($p.path)`n       期待 $($p.sha256)`n       実際 $got"
            }
        }
    }
    Write-Ok "全パッチが UPSTREAM.lock と一致"
}

# ---- clone とパッチ適用 -----------------------------------------------------
function Setup-Upstream($name, $cfg, $patchList) {
    Write-Step "$name : clone とパッチ適用"
    $dest = $cfg["dest"]
    # StackRoot が既定と違うなら差し替える
    if ($StackRoot -ne "D:\IsaacStack") {
        $dest = $dest -replace [regex]::Escape("D:\IsaacStack"), $StackRoot
    }
    $marker = Join-Path $dest ".peg_in_hole_patched"

    if (Test-Path $marker) { Write-Skip "$dest は適用済み"; return }

    if (-not (Test-Path $dest)) {
        Write-Host "  clone $($cfg['url']) -> $dest"
        # --no-checkout で先に取り、固定 commit だけを展開する（無駄な checkout を避ける）
        git clone --no-checkout $cfg["url"] $dest
        if ($LASTEXITCODE -ne 0) { Fail "clone に失敗: $name" }
    } else {
        Write-Skip "$dest は既にある（clone を飛ばす）"
    }

    $rev = if ($cfg.ContainsKey("tag")) { $cfg["tag"] } else { $cfg["commit"] }
    Write-Host "  checkout $rev"
    git -C $dest checkout --detach $rev
    if ($LASTEXITCODE -ne 0) { Fail "checkout に失敗: $name $rev" }

    # libuipc（TacEx の入れ子 submodule）は初期化しない。ソースビルドが走る。
    if ($cfg["submodule"] -eq "none") { Write-Ok "submodule は初期化しない（方針）" }

    foreach ($p in $patchList) {
        $full = Join-Path $RepoRoot $p.path
        $leaf = Split-Path $p.path -Leaf
        # format-patch 形式なら am、素の diff なら apply
        $isMailbox = (Get-Content $full -TotalCount 1) -match '^From [0-9a-f]{40}'
        if ($isMailbox) {
            git -C $dest am --keep-cr "$full"
        } else {
            git -C $dest apply --whitespace=nowarn "$full"
        }
        if ($LASTEXITCODE -ne 0) { Fail "パッチ適用に失敗: $leaf" }
        Write-Ok $leaf
    }

    New-Item -ItemType File -Path $marker -Force | Out-Null
    Set-Content -Path $marker -Value "patched at $(Get-Date -Format o)" -Encoding utf8
}

Setup-Upstream "isaaclab" $upstreams["isaaclab"] $patches["isaaclab"]
Setup-Upstream "tacex"    $upstreams["tacex"]    $patches["tacex"]

# ---- 次にやること -----------------------------------------------------------
Write-Step "clone とパッチ適用は完了"
Write-Host @"

次は venv を作る。README の 3.2 - 3.5 をそのまま実行すること。
（pip の download が数 GB あり、順序を間違えると壊れるので自動化していない）

  1. README 3.2  venv と Isaac Sim   ... torch を先に入れる
  2. README 3.3  Isaac Lab を editable で
  3. README 3.4  TacEx 用 venv（触覚実験を回す場合）
  4. README 3.5  activate_isaac.ps1 を $StackRoot へコピー

そのあと動作確認:

  $StackRoot\env_isaaclab\Scripts\python.exe scripts\doctor.py
  cmd /c "$StackRoot\env_isaaclab\Scripts\python.exe -u scripts\verify_setup.py --profile forge > results\verify_forge.log 2>&1"

"@ -ForegroundColor Gray
