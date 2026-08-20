"""左右の GelSight が対称に応答しているかを測る。方策も学習済みモデルも要らない。

**何を測るか**: 環境を作ってリセットし、両センサの生データを読むだけ。
主指標は **高さマップ（カメラ深度 mm）の左右差**で、これは1ステップ目から出る。

    元アセット   : 右のほうが遠い（差 +0.26mm 前後）＝ ペグが左寄りに保持されている
    対称化アセット: 差が約 0 になるはず

**なぜ方策が要らないか**: 非対称はアセットの幾何に由来する静的な性質なので、
掴んだ直後の1フレームで出る。∴ チェックポイントの無いマシンでも検証できる。

使い方（TacEx 用 venv で・headless）:

    # A) 元アセット
    cmd /c "<venv>\\Scripts\\python.exe -u scripts\\exp_tacex\\check_gelsight_symmetry.py ^
        --label A_original --headless > results\\sym_A.log 2>&1"

    # B) 対称化アセット（先に make_symmetric_gelpads.py を実行しておく）
    cmd /c "<venv>\\Scripts\\python.exe -u scripts\\exp_tacex\\check_gelsight_symmetry.py ^
        --label B_symmetric --symmetric --headless > results\\sym_B.log 2>&1"

判定は各ランが自分で出す（`SYMMETRY_OK` / `SYMMETRY_ASYMMETRIC`）。
**同一マシン内の A/B で完結する**ので、他マシンの数値と突き合わせる必要はない。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# ★ uipc は AppLauncher より前に読む。後だと PyInit_pyuipc で access violation。
try:
    import uipc  # noqa: F401
except Exception:  # noqa: BLE001
    pass

ISAAC_ROOT = pathlib.Path(os.environ.get("ISAAC_STACK_ROOT", r"D:\IsaacStack"))
SYM_USD = (ISAAC_ROOT / "TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka"
           / "GelSight_Mini/Gripper/physx_rigid_gelpads_symmetric.usd")

# Kit 起動より前に環境変数を立てる（EULA・キャッシュ）
for _k, _v in {
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "OMNI_CACHE_DIR": str(ISAAC_ROOT / "cache/ov"),
    "OMNI_DATA_DIR": str(ISAAC_ROOT / "cache/ov_data"),
    "OMNI_LOGS_DIR": str(ISAAC_ROOT / "cache/ov_logs"),
    "TORCHDYNAMO_DISABLE": "1",
}.items():
    os.environ.setdefault(_k, _v)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--label", default="sym_check", help="出力名")
parser.add_argument("--symmetric", action="store_true",
                    help="対称化アセット physx_rigid_gelpads_symmetric.usd を使う")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20, help="リセット後に進めるステップ数")
parser.add_argument("--outdir", default="results")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True          # GelSight はカメラを使う＝必須

app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import tacex_tasks  # noqa: F401, E402

TASK = "TacEx-Factory-PegInsert-Direct-v0"
FAR_MM = 29.0          # clippingRange[1]。inf はこの値に丸められる
GP2CAM_M = 0.024       # gelpad_to_camera_min_distance
GELPAD_H_M = 4.5e-3    # gelpad_dimensions.height


def main() -> int:
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
    cfg.tactile_in_obs = False       # 観測に入れる必要はない。センサの生データだけ見る
    cfg.seed = 0

    if args.symmetric:
        if not SYM_USD.exists():
            print(f"SYMMETRY_ERROR 対称化アセットが無い: {SYM_USD}", flush=True)
            print("  先に make_symmetric_gelpads.py を実行すること", flush=True)
            return 2
        cfg.robot.spawn.usd_path = str(SYM_USD)
    print(f"USD: {cfg.robot.spawn.usd_path}", flush=True)

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    u = env.unwrapped
    env.reset()

    hm_l, hm_r, tac = [], [], []
    zero = torch.zeros((u.num_envs, u.cfg.action_space), device=u.device)
    with torch.inference_mode():
        for _ in range(args.steps):
            env.step(zero)
            hm_l.append(u.gsmini_left.data.output["height_map"].clone().cpu().numpy())
            hm_r.append(u.gsmini_right.data.output["height_map"].clone().cpu().numpy())
            tac.append(u._get_tactile_obs().clone().cpu().numpy())

    HL = np.asarray(hm_l, dtype=np.float64)    # (T,N,32,32)
    HR = np.asarray(hm_r, dtype=np.float64)
    TA = np.asarray(tac, dtype=np.float32)     # (T,N,32,32,6)
    T, N = HL.shape[0], HL.shape[1]

    def indent(h):                              # taxim_sim.compute_indentation_depth と同式
        mind = h.reshape(h.shape[0], h.shape[1], -1).min(axis=2) / 1000.0
        dist = np.maximum(mind - GP2CAM_M, 0.0)
        return np.where(dist <= GELPAD_H_M, (GELPAD_H_M - dist) * 1000.0, 0.0)

    lmin = HL.reshape(T, N, -1).min(axis=2)
    rmin = HR.reshape(T, N, -1).min(axis=2)
    il, ir = indent(HL), indent(HR)
    lstd = TA[..., 0:3].std(axis=0).mean()
    rstd = TA[..., 3:6].std(axis=0).mean()
    lvis = (HL < FAR_MM - 1e-6).mean() * 100
    rvis = (HR < FAR_MM - 1e-6).mean() * 100

    hm_diff = float(rmin.mean() - lmin.mean())
    ind_ratio = float(il.mean() / ir.mean()) if ir.mean() > 0 else float("inf")

    print("=" * 74, flush=True)
    print(f"GelSight 左右対称性チェック  label={args.label}  "
          f"asset={'symmetric' if args.symmetric else 'original'}  "
          f"num_envs={N} steps={T}", flush=True)
    print("=" * 74, flush=True)
    print(f"  高さマップ最短距離   left {lmin.mean():.4f} mm   right {rmin.mean():.4f} mm", flush=True)
    print(f"  ★ 左右差 (right-left) = {hm_diff:+.4f} mm       （対称なら 0）", flush=True)
    print(f"  押し込み量           left {il.mean():.4f} mm   right {ir.mean():.4f} mm", flush=True)
    print(f"  ★ 押し込み量の比 L/R  = {ind_ratio:.3f}            （対称なら 1.0）", flush=True)
    print(f"  可視画素             left {lvis:.2f}%      right {rvis:.2f}%", flush=True)
    print(f"  触覚 std             left {lstd:.5f}    right {rstd:.5f}   "
          f"比 {lstd/rstd if rstd>0 else float('nan'):.3f}", flush=True)

    ok = abs(hm_diff) < 0.05 and 0.80 <= ind_ratio <= 1.25
    print("=" * 74, flush=True)
    print("SYMMETRY_OK" if ok else "SYMMETRY_ASYMMETRIC", flush=True)
    print("  判定式: |高さマップ左右差| < 0.05mm かつ 押し込み量の比 が [0.80, 1.25]", flush=True)

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}_symmetry.json").write_text(json.dumps({
        "label": args.label, "asset": "symmetric" if args.symmetric else "original",
        "usd_path": str(cfg.robot.spawn.usd_path), "num_envs": N, "steps": T,
        "hm_min_left_mm": float(lmin.mean()), "hm_min_right_mm": float(rmin.mean()),
        "hm_diff_mm": hm_diff,
        "indentation_left_mm": float(il.mean()), "indentation_right_mm": float(ir.mean()),
        "indentation_ratio_LR": ind_ratio,
        "visible_pct_left": float(lvis), "visible_pct_right": float(rvis),
        "tactile_std_left": float(lstd), "tactile_std_right": float(rstd),
        "verdict": "SYMMETRY_OK" if ok else "SYMMETRY_ASYMMETRIC",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / f"{args.label}_symmetry.npz", hm_left=HL, hm_right=HR)
    print(f"  保存: {out / (args.label + '_symmetry.json')}", flush=True)
    env.close()
    return 0 if ok else 1


try:
    _rc = main()
except Exception:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    _rc = 3
finally:
    # Isaac の終了処理はハングすることがある。判定は上で印字済み。
    app.close()

# app.close() の後の print は出ないことがあるので、判定は main() 内で印字済み。
# 終了コードは自動判定（CI・キュー）から使うので必ず返す。
#   0 = SYMMETRY_OK / 1 = SYMMETRY_ASYMMETRIC / 2 = 対称化アセットが無い / 3 = 例外
sys.exit(_rc)
