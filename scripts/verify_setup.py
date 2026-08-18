"""環境構築の最小検証。学習は一切せず、以下だけを確かめる。

1. Isaac Sim / Isaac Lab が headless で起動できる
2. 期待するカスタムタスクが gym に登録されている
3. GPU が torch から見えている

使い方（doctor.py が DOCTOR_OK になってから実行する）:
    D:\IsaacStack\env_isaaclab\Scripts\python.exe    -u scripts\verify_setup.py --profile forge
    D:\IsaacStack\env_tacex_isaac\Scripts\python.exe -u scripts\verify_setup.py --profile tacex

全部通れば最後に SETUP_OK を印字して終了コード0。
1つでも欠けると欠落タスク名を出して終了コード1。

出力の取り方に2つ罠がある（どちらも実際に踏んだ）:
- `python -u` を付けないと結果が出ない（stdout がバッファされる）
- PowerShell の `2>&1 | Out-File` は native の stderr を壊す。`cmd /c "... > log 2>&1"` で受ける
"""

import argparse
import sys

# ---- プロファイル定義 --------------------------------------------------------
# forge : IsaacLab フォーク (peg-in-hole/v2.3.2-custom) が提供するカスタムタスク7種
# tacex : TacEx フォーク (peg-in-hole/blackwell) が提供する触覚タスク
PROFILES = {
    "forge": {
        "module": "isaaclab_tasks",
        "enable_cameras": False,
        "tasks": [
            "Isaac-Factory-CylInsert-Direct-v0",
            "Isaac-Forge-PegInsertBlind-Direct-v0",
            "Isaac-Forge-PegInsertBlindDR-Direct-v0",
            "Isaac-Forge-PegInsertRPTilt-Direct-v0",
            "Isaac-Forge-PegInsertOracle-Direct-v0",
            "Isaac-Forge-PegSearchMap-Direct-v0",
            "Isaac-Forge-PegSearchMapDR-Direct-v0",
        ],
        "hint": "IsaacLab が fork の peg-in-hole/v2.3.2-custom ブランチになっているか確認する",
    },
    "tacex": {
        "module": "tacex_tasks",
        # GelSight センサはカメラ経路を使うので有効化しておく
        "enable_cameras": True,
        "tasks": [
            "TacEx-Factory-PegInsert-Direct-v0",
            "TacEx-Factory-GearMesh-Direct-v0",
            "TacEx-Factory-NutThread-Direct-v0",
        ],
        "hint": "TacEx の 3 拡張 (tacex / tacex_assets / tacex_tasks) が editable install されているか確認する",
    },
}

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--profile", choices=sorted(PROFILES), required=True)
_args, _rest = _pre.parse_known_args()
PROFILE = PROFILES[_args.profile]

# ---- ★ TacEx の必須前置き -------------------------------------------------
# tacex_tasks は uipc 版タスクを条件付き登録する。Kit 起動後に uipc を import すると
# PyInit_pyuipc で access violation を起こしてプロセスごと落ちる（実測: 0.7 秒で即死）。
# したがって AppLauncher より **前** に読んでおく必要がある。
if _args.profile == "tacex":
    try:
        import uipc as _uipc_preload  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
a = p.parse_args([])
a.headless = True
if PROFILE["enable_cameras"]:
    a.enable_cameras = True
app_launcher = AppLauncher(a)
app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# ---- ★ 素の import 文で読む -------------------------------------------------
# importlib.import_module() 経由だと tacex が名前空間パッケージとして解決され
# `ImportError: cannot import name 'GelSightSensor' from 'tacex' (unknown location)`
# で落ちる（2026-08-19 実測。torch の import 有無・tacex の先読み有無は無関係と A/B 済み）。
if _args.profile == "forge":
    import isaaclab_tasks  # noqa: F401, E402
else:
    import tacex_tasks  # noqa: F401, E402

registered = set(gym.registry.keys())
missing = [t for t in PROFILE["tasks"] if t not in registered]

print("---- verify_setup ----")
print(f"profile     : {_args.profile}  (module: {PROFILE['module']})")
print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__} (cuda {torch.version.cuda})")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i)
        print(f"  gpu[{i}]    : {torch.cuda.get_device_name(i)} (cc {cc[0]}.{cc[1]})")
for t in PROFILE["tasks"]:
    print(f"  {'OK  ' if t not in missing else 'MISS'} {t}")

if missing:
    print(f"SETUP_NG missing {len(missing)} task(s): {missing}")
    print(f"→ {PROFILE['hint']}")
else:
    print("SETUP_OK")

# app.close() の後に置いた print はプロセスごと落ちて出ないので、判定は必ずこの前に出す。
app.close()
sys.exit(1 if missing else 0)
