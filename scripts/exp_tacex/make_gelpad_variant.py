"""ゲルパッドの取り付けオフセットを変えた USD を作る（元アセットは無改変）。

**⚠ 「対称にすれば直る」ではない。** それは実験で否定された（下記）。
本スクリプトは**任意のオフセットを与える実験用の道具**であって、修正案ではない。

---

## 何を操作するか

ゲルパッドはアーティキュレーションのリンクなので、**PhysX が使うのはリンクの xform ではなく
FixedJoint のローカルフレーム**（`localPos0`）である。同じ非対称量が2箇所に書かれていて、
実効なのはジョイント側。

    /panda/gelsight_mini_case_left/FixedJointCaseLeft   body0=gelpad_left   localPos0.z = -0.024254449
    /panda/gelsight_mini_case_right/FixedJointCaseRight body0=gelpad_right  localPos0.z = -0.023959097
                                                                       差 = 0.2954 mm

**リンクの xform だけ直しても実行時には届かない**（PC1 で実測・出力がビット単位で不変だった）。

## 分かっていること（PC1 実測・2026-08-20・RTX 4060）

`--joint-diff-mm` を変えると、観測される左右差は**ほぼ 1:1 mm/mm で線形**に動く（傾き 0.9963）。

| joint 差 | 高さマップ左右差 | 押し込み比 L/R | 判定 |
|---|---|---|---|
| +0.2954（元アセット） | +0.2498 mm | 1.698 | ASYMMETRIC |
| **0.0（「対称」）** | **+0.5441 mm** | **3.560** | **悪化** |
| −0.5441（数値的に打ち消す値） | −0.0002 mm | 0.9996 | OK |

∴ **系にはゲルパッド配置とは別に約 0.5441mm の非対称が存在し、元アセットの
オフセットはそれを部分的に打ち消していた。** 対称に直すと打ち消しが外れて悪化する。

**−0.5441 は指標を数値的に打ち消しただけの経験値**であって、物理的に正しい修正ではない。
真因の 0.5441mm がどこから来るかは**未特定**。

詳細: `docs/gelsight-right-sensor-asymmetry.md`

---

## 使い方

`pxr`（usd-core）が要る。**Isaac の venv には入っていない**ので、Isaac 非依存の venv を使う。

    py -3.11 -m venv D:\\IsaacStack\\env_tacex
    D:\\IsaacStack\\env_tacex\\Scripts\\python.exe -m pip install usd-core numpy

    # 計測だけ（書き込まない）
    ...\\env_tacex\\Scripts\\python.exe scripts\\exp_tacex\\make_gelpad_variant.py --dry-run

    # ジョイント差を指定して variant を書く
    ... make_gelpad_variant.py --joint-diff-mm 0.0      -> _sym0.usd
    ... make_gelpad_variant.py --joint-diff-mm -0.5441  -> _symm0p5441.usd
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

try:
    from pxr import Gf, Usd, UsdGeom  # noqa: F401
except ModuleNotFoundError:
    sys.exit(
        "pxr (usd-core) が無い。Isaac の venv には入っていないので Isaac 非依存の venv を使う:\n"
        "  py -3.11 -m venv D:\\IsaacStack\\env_tacex\n"
        "  D:\\IsaacStack\\env_tacex\\Scripts\\python.exe -m pip install usd-core numpy"
    )

import numpy as np

TC = Usd.TimeCode.Default()
REL = (r"source\tacex_assets\tacex_assets\data\Robots\Franka\GelSight_Mini"
       r"\Gripper\physx_rigid_gelpads.usd")
JOINT = {"left": "/panda/gelsight_mini_case_left/FixedJointCaseLeft",
         "right": "/panda/gelsight_mini_case_right/FixedJointCaseRight"}


def local_pos0(stage: Usd.Stage, side: str):
    prim = stage.GetPrimAtPath(JOINT[side])
    if not prim.IsValid():
        raise SystemExit(f"ジョイントが無い: {JOINT[side]}")
    attr = prim.GetAttribute("physics:localPos0")
    if not attr or not attr.HasValue():
        raise SystemExit(f"physics:localPos0 が無い: {JOINT[side]}")
    return attr, np.array(attr.Get(), dtype=np.float64)


def report(stage: Usd.Stage, tag: str) -> float:
    print(f"--- {tag} ---")
    vals = {}
    for s in ("left", "right"):
        _, v = local_pos0(stage, s)
        vals[s] = v
        print(f"  {JOINT[s].rsplit('/', 1)[-1]:22s} localPos0 = {np.round(v*1000, 6)} mm")
    diff = float(abs(vals["left"][2]) - abs(vals["right"][2]))
    print(f"  ★ |left.z| - |right.z| = {diff*1000:+.4f} mm")
    return diff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack-root", default=os.environ.get("ISAAC_STACK_ROOT", r"D:\IsaacStack"))
    ap.add_argument("--joint-diff-mm", type=float, default=None,
                    help="目標の |left.z|-|right.z| [mm]。省略時は計測のみ")
    ap.add_argument("--out", default=None, help="出力ファイル名（省略時は自動命名）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.stack_root) / "TacEx" / REL
    if not src.exists():
        print(f"元アセットが無い: {src}", file=sys.stderr)
        return 1

    print("=" * 74)
    print(f"元アセット: {src}")
    print("=" * 74)
    cur = report(Usd.Stage.Open(str(src)), "現在")

    if args.joint_diff_mm is None or args.dry_run:
        print("\n（計測のみ。--joint-diff-mm を指定すると variant を書く）")
        return 0

    target = args.joint_diff_mm / 1000.0
    if args.out:
        dst = src.with_name(args.out)
    else:
        tag = f"{args.joint_diff_mm:+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
        dst = src.with_name(f"physx_rigid_gelpads_jd{tag}.usd")

    shutil.copy2(src, dst)
    out = Usd.Stage.Open(str(dst))
    # right 側だけを動かして目標の差にする（left は基準として動かさない）
    attr_l, vl = local_pos0(out, "left")
    attr_r, vr = local_pos0(out, "right")
    sign = np.sign(vr[2]) or 1.0
    new_abs = abs(vl[2]) - target
    vr_new = vr.copy()
    vr_new[2] = sign * new_abs
    attr_r.Set(Gf.Vec3f(*[float(v) for v in vr_new]))
    out.GetRootLayer().Save()

    print()
    print("=" * 74)
    got = report(Usd.Stage.Open(str(dst)), "変更後")
    print()
    print(f"  目標 {args.joint_diff_mm:+.4f} mm / 実際 {got*1000:+.4f} mm")
    print(f"保存: {dst}")
    print(f"元ファイルは無改変（{src.stat().st_size} bytes）")
    print()
    print("⚠ これは実験用の variant であって修正案ではない。")
    print("  joint 差 0（「対称」）は実測で**悪化**する（+0.5441mm / 比 3.560）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
