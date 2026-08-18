"""Blackwell (RTX 50 系 / sm_120) 上の Isaac Sim スタックの事前診断。

Isaac Sim を起動しないので数秒で終わる。verify_setup.py（Isaac を実際に起動して
タスク登録を見る・約9秒）の手前に置く軽い関門。

使い方（診断したい venv の python で実行する）:
    D:\IsaacStack\env_isaaclab\Scripts\python.exe scripts\doctor.py
    D:\IsaacStack\env_tacex_isaac\Scripts\python.exe scripts\doctor.py

判定:
    [OK]   問題なし
    [WARN] 用途によっては困る（GUI を使わないなら無視してよい等）
    [NG]   このまま進むと落ちる

NG が1つでもあれば終了コード 1。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTS: list[tuple[str, str, str]] = []


def rec(level: str, name: str, detail: str) -> None:
    RESULTS.append((level, name, detail))


def check_python() -> None:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        rec("OK", "python", f"{got}")
    else:
        rec("NG", "python", f"{got} (Isaac Sim 5.1 は 3.11 系が前提)")


def check_torch() -> None:
    try:
        import torch
    except Exception as e:
        rec("NG", "torch", f"import できない: {e}")
        return

    ver = torch.__version__
    # CPU 版への巻き上げは過去に環境を壊した最頻の事故
    if "+cpu" in ver or "+cu" not in ver:
        rec("NG", "torch build", f"{ver} = CPU版の可能性。GPU版 (+cu128) を入れ直す")
    else:
        rec("OK", "torch build", ver)

    cuda = torch.version.cuda
    if cuda == "12.8":
        rec("OK", "torch CUDA", cuda)
    else:
        rec("WARN", "torch CUDA", f"{cuda} (検証済みは 12.8)")

    # ---- Blackwell の核心: sm_120 を含んでビルドされているか ----
    try:
        arches = torch.cuda.get_arch_list()
    except Exception as e:
        arches = []
        rec("NG", "torch arch list", f"取得できない: {e}")
    if arches:
        if "sm_120" in arches:
            rec("OK", "sm_120 サポート", f"{' '.join(arches)}")
        else:
            rec(
                "NG",
                "sm_120 サポート",
                f"torch が sm_120 を含まない ({' '.join(arches)})。Blackwell では PTX JIT 頼みになるか動かない",
            )

    if not torch.cuda.is_available():
        rec("NG", "cuda 利用可否", "torch.cuda.is_available() が False")
        return
    rec("OK", "cuda 利用可否", "True")

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cc = torch.cuda.get_device_capability(i)
        tag = f"{name} / cc {cc[0]}.{cc[1]}"
        if cc[0] >= 12:
            rec("OK", f"gpu[{i}]", tag + " = Blackwell 世代")
        else:
            rec("WARN", f"gpu[{i}]", tag + " = Blackwell ではない")


def check_pins() -> None:
    import importlib.metadata as md

    # (パッケージ名, 期待値, 外れたときの level, 説明)
    pins = [
        ("numpy", "1.26.0", "NG", "2.x に上げると Isaac が動かない"),
        ("gymnasium", "1.2.1", "WARN", "isaacsim/isaaclab が要求する境界"),
    ]
    for pkg, want, lvl, why in pins:
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            rec("WARN", pkg, f"未インストール ({why})")
            continue
        if got == want:
            rec("OK", pkg, got)
        else:
            rec(lvl, pkg, f"{got} (期待 {want}: {why})")

    # h5py は GUI モードだけで効く。headless しか使わないなら無視してよい。
    try:
        got = md.version("h5py")
        if got == "3.11.0":
            rec("OK", "h5py", got)
        else:
            rec(
                "WARN",
                "h5py",
                f"{got} — GUI モードで DLL 競合により即死する。GUI を使うなら h5py==3.11.0",
            )
    except md.PackageNotFoundError:
        rec("OK", "h5py", "未インストール (GUI を使わないなら問題なし)")


def check_nvidia_wheels() -> None:
    """Isaac の venv に nvidia-*-cu12 wheel が入っていると Kit が起動時にクラッシュする。

    torch が site-packages/nvidia/*/bin を DLL 探索に足すため、torch 同梱 (cu12.8) と
    wheel (cu12.9) が混ざる。uipc が要る DLL は torch\lib に全部ある。
    """
    sp = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not sp.exists():
        rec("OK", "nvidia-*-cu12 wheel", "なし")
        return
    subdirs = sorted(p.name for p in sp.iterdir() if p.is_dir())
    rec(
        "NG",
        "nvidia-*-cu12 wheel",
        f"site-packages/nvidia/ が存在 ({', '.join(subdirs[:6])}...)。"
        " torch 同梱 CUDA と混ざって Kit が起動時にクラッシュする",
    )


def check_env_vars() -> None:
    if os.environ.get("TORCHDYNAMO_DISABLE") == "1":
        rec("OK", "TORCHDYNAMO_DISABLE", "1")
    else:
        rec(
            "NG",
            "TORCHDYNAMO_DISABLE",
            f"{os.environ.get('TORCHDYNAMO_DISABLE')!r} — Windows に Triton が無いので 1 が必須",
        )

    if os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES":
        rec("OK", "OMNI_KIT_ACCEPT_EULA", "YES")
    else:
        rec(
            "NG",
            "OMNI_KIT_ACCEPT_EULA",
            "未設定 — import 時に対話プロンプトで止まる（非対話実行では EOF で落ちる）",
        )


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def check_gpu_free() -> None:
    out = _run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    if out is None:
        rec("WARN", "VRAM", "nvidia-smi を実行できない")
        return
    line = out.strip().splitlines()[0]
    used, total = (int(x.strip()) for x in line.split(","))
    free = total - used
    # Isaac の学習は実測で約 9.6GB 要求した実績がある（plan24 hold-out 学習）
    if free >= 9600:
        rec("OK", "VRAM 空き", f"{free} MiB / {total} MiB")
    elif free >= 6100:
        rec("WARN", "VRAM 空き", f"{free} MiB — 小さい num_envs なら起動できるが学習には不足しうる")
    else:
        rec("NG", "VRAM 空き", f"{free} MiB — Isaac が起動できない（実測で --num_envs 2 でも約 6.1GB 要る）")


def check_ollama() -> None:
    """`ollama ps` の出力が空であることが唯一確実な確認。

    プロセスの有無では判定できない（サービス常駐のままモデルだけ載っている状態を取る）。
    """
    exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    cmd = [str(exe)] if exe.exists() else ["ollama"]
    out = _run(cmd + ["ps"])
    if out is None:
        rec("OK", "ollama", "未起動 or 未インストール")
        return
    rows = [ln for ln in out.strip().splitlines()[1:] if ln.strip()]
    if not rows:
        rec("OK", "ollama", "モデル未ロード")
    else:
        rec(
            "NG",
            "ollama",
            f"モデルがロード中: {rows[0].split()[0]} — 同じ 16GB を奪い合う。`ollama stop <model>` で降ろす",
        )


def main() -> int:
    check_python()
    check_torch()
    check_pins()
    check_nvidia_wheels()
    check_env_vars()
    check_gpu_free()
    check_ollama()

    print("---- doctor (Blackwell / Isaac Sim stack) ----")
    print(f"venv: {sys.prefix}")
    width = max(len(n) for _, n, _ in RESULTS)
    for level, name, detail in RESULTS:
        print(f"[{level:4}] {name:{width}}  {detail}")

    ng = [r for r in RESULTS if r[0] == "NG"]
    warn = [r for r in RESULTS if r[0] == "WARN"]
    print("----")
    if ng:
        print(f"DOCTOR_NG  NG {len(ng)} 件 / WARN {len(warn)} 件")
        return 1
    print(f"DOCTOR_OK  WARN {len(warn)} 件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
