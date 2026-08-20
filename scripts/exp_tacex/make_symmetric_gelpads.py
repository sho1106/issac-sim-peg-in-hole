"""左右のゲルパッドを対称に配置し直した USD を作る（元アセットは無改変）。

**なぜ要るか**: TacEx 上流のグリッパアセットは、ゲルパッドの左右配置が対称でない。

    gelpad_left  の左右軸への射影 = -7.4161 mm
    gelpad_right の左右軸への射影 = +7.7115 mm      差 0.2954 mm（対称なら 0）

ゲルパッドは `CollisionAPI` と `RigidBodyAPI` を持ち（`panda_leftfinger`/`rightfinger` は
CollisionAPI を持たない）、**ペグを実際に掴んでいるのはゲルパッド**である。
∴ パッドの中点が左へ 0.1477mm ずれる → ペグもそこに保持される →
左右カメラからの距離が 0.2954mm 違う → 右センサの感度が下がる。

実機の触覚センサペアは左右対称に見えることが期待されるので、これはシミュレータの
妥当性の問題。詳細は docs/gelsight-right-sensor-asymmetry.md。

**なぜアセットを直すのか**: `Articulation()` 生成後に USD の xform を書き換えても、
PhysX は既に剛体を解析済みなので反映されない（実測でビット単位に不変だった）。
スポーンより前＝アセット自体を直す必要がある。

使い方:

    <TacEx用venv>\\Scripts\\python.exe scripts\\exp_tacex\\make_symmetric_gelpads.py

    # 置き場所を変えているなら
    ... make_symmetric_gelpads.py --stack-root E:\\IsaacStack

出力: 元アセットと同じディレクトリに `physx_rigid_gelpads_symmetric.usd`
（`--dry-run` を付けると計測だけして書き込まない）
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom

TC = Usd.TimeCode.Default()
REL = r"source\tacex_assets\tacex_assets\data\Robots\Franka\GelSight_Mini\Gripper\physx_rigid_gelpads.usd"


def m2np(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def world(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise SystemExit(f"プリムが無い: {path}")
    return m2np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(TC))


def grip_axis(stage: Usd.Stage) -> tuple[np.ndarray, np.ndarray]:
    """カメラ中点 M と、左右カメラを結ぶ単位ベクトル（＝指が閉じる方向）。"""
    cl = world(stage, "/panda/gelsight_mini_case_left/Camera")[3, :3]
    cr = world(stage, "/panda/gelsight_mini_case_right/Camera")[3, :3]
    axis = cr - cl
    return (cl + cr) / 2.0, axis / np.linalg.norm(axis)


def projections(stage: Usd.Stage) -> dict[str, float]:
    """各ゲルパッドの、カメラ中点からの左右軸方向の位置 [m]。"""
    mid, axis = grip_axis(stage)
    return {s: float((world(stage, f"/panda/gelpad_{s}")[3, :3] - mid) @ axis)
            for s in ("left", "right")}


def report(stage: Usd.Stage, tag: str) -> dict[str, float]:
    p = projections(stage)
    asym = abs(p["right"]) - abs(p["left"])
    print(f"--- {tag} ---")
    for s in ("left", "right"):
        print(f"  gelpad_{s:5s} 左右軸への射影 = {p[s]*1000:+9.4f} mm")
    print(f"  非対称量 = {asym*1000:+.4f} mm  （対称なら 0）")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack-root", default=os.environ.get("ISAAC_STACK_ROOT", r"D:\IsaacStack"),
                    help="Isaac スタックの置き場所（既定 D:\\IsaacStack / 環境変数 ISAAC_STACK_ROOT）")
    ap.add_argument("--dry-run", action="store_true", help="計測だけして書き込まない")
    args = ap.parse_args()

    src = pathlib.Path(args.stack_root) / "TacEx" / REL
    if not src.exists():
        print(f"元アセットが無い: {src}", file=sys.stderr)
        return 1
    dst = src.with_name("physx_rigid_gelpads_symmetric.usd")

    print("=" * 74)
    print(f"元アセット: {src}")
    print("=" * 74)
    stage = Usd.Stage.Open(str(src))
    before = report(stage, "修正前")

    half = (abs(before["left"]) + abs(before["right"])) / 2.0
    shift = {"left": (-half) - before["left"], "right": (+half) - before["right"]}
    print()
    print(f"  目標（対称）  left {-half*1000:+.4f} / right {+half*1000:+.4f} mm")
    print(f"  移動量        left {shift['left']*1000:+.4f} / right {shift['right']*1000:+.4f} mm（軸方向）")

    if abs(abs(before["right"]) - abs(before["left"])) * 1000 < 1e-3:
        print("\n既に対称。何もしない。")
        return 0
    if args.dry_run:
        print("\n--dry-run のため書き込まない。")
        return 0

    _, axis = grip_axis(stage)
    shutil.copy2(src, dst)
    out = Usd.Stage.Open(str(dst))
    for s in ("left", "right"):
        prim = out.GetPrimAtPath(f"/panda/gelpad_{s}")
        xf = UsdGeom.Xformable(prim)
        op = next(o for o in xf.GetOrderedXformOps() if o.GetOpName().endswith("translate"))
        parent_r = m2np(UsdGeom.Xformable(prim.GetParent())
                        .ComputeLocalToWorldTransform(TC))[:3, :3]
        d_local = (axis * shift[s]) @ np.linalg.inv(parent_r)
        cur = np.array(op.Get(), dtype=np.float64)
        op.Set(Gf.Vec3d(*[float(v) for v in (cur + d_local)]))
    out.GetRootLayer().Save()

    print()
    print("=" * 74)
    report(Usd.Stage.Open(str(dst)), "修正後")
    print()
    print(f"保存: {dst}")
    print(f"元ファイルは無改変（{src.stat().st_size} bytes）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
