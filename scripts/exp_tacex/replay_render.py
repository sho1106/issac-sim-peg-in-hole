"""記録した姿勢を書き戻して録画する（方策を回さない再生レンダラ）。

**なぜ要るか**: カメラ（render product）を作ると GelSight の画像が変わり、触覚アームの
入力が分布外になる。同一 δ・同一 g・同一 checkpoint で成功率が **56.25% → 3.91%**
まで落ちることを実測した（`scripts/plan27/results/repro.json`）。照明を外しても
warmup を 0 にしても同じ 5/128 で、**カメラの存在自体**が効いている。
∴ 方策をカメラつきで回した映像は「その方策の挙動」ではない。

本スクリプトは方策を一切回さず、`--log_pose` で記録した関節角とルート姿勢を毎ステップ
書き戻して描画する。∴ 映像は**記録したエピソードそのもの**であり、隣に出す波形と同一。

  $env:PYTHONNOUSERSITE=1; $env:PYTHONUTF8=1
  D:\\IsaacStack\\env_tacex_isaac\\Scripts\\python.exe scripts/plan27/replay_render.py \\
      --signals scripts/plan27/render/p27_ref_b0_signals.npz --env 82 --label replay_env082
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

# --- Kit 起動より前に済ませること（順序は render_failure_cases.py と同じ。崩すと落ちる）---
# スタックの場所は環境変数 ISAAC_STACK_ROOT で上書きできる（既定は D:\IsaacStack）。
# 新しい PC では置き場所が変わるので、直書きしない。
ISAAC_ROOT = pathlib.Path(os.environ.get("ISAAC_STACK_ROOT", r"D:\IsaacStack"))
for _k, _v in {
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "OMNI_KIT_ALLOW_ROOT": "1",
    "OMNI_CACHE_DIR": str(ISAAC_ROOT / "cache/ov"),
    "OMNI_DATA_DIR": str(ISAAC_ROOT / "cache/ov_data"),
    "OMNI_LOGS_DIR": str(ISAAC_ROOT / "cache/ov_logs"),
    "XDG_CACHE_HOME": str(ISAAC_ROOT / "cache/xdg"),
    "HF_HOME": str(ISAAC_ROOT / "cache/huggingface"),
    "TORCH_HOME": str(ISAAC_ROOT / "cache/torch"),
    "TORCHDYNAMO_DISABLE": "1",
}.items():
    os.environ.setdefault(_k, _v)
    if _k.endswith("_DIR") or _k in ("XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME"):
        pathlib.Path(os.environ[_k]).mkdir(parents=True, exist_ok=True)

import h5py  # noqa: F401,E402  競合 HDF5 を先に載せられると 0xc0000139 で落ちる

try:
    import uipc  # noqa: F401,E402  AppLauncher より前に読む（後回しだと access violation）
except Exception:  # noqa: BLE001
    pass

from isaaclab.app import AppLauncher  # noqa: E402

TASK = "TacEx-Factory-PegInsert-Direct-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--signals", required=True, help="--log_pose つきで撮った npz")
parser.add_argument("--env", type=int, default=None, help="再生する env 番号（1本だけ撮る）")
parser.add_argument("--envs", default=None,
                    help="複数 env を1セッションで撮る。'all' か '0,5,7' か '0-127'。"
                         "Kit 起動が1回で済むので、全 env を撮るならこちら")
parser.add_argument("--stride", type=int, default=1,
                    help="何 step ごとに1コマ撮るか。全 env を撮るときは 3 程度に落とす")
parser.add_argument("--label", default=None, help="1本撮りのときの出力名")
parser.add_argument("--label_fmt", default="p27_rp_env{env:03d}",
                    help="複数撮りのときの出力名（{env} が入る）")
parser.add_argument("--outdir", default="scripts/plan27/render")
parser.add_argument("--res", type=int, nargs=2, default=[960, 540])
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--warmup", type=int, default=24)
parser.add_argument("--eye_offset", type=float, nargs=3, default=[0.547, -0.776, 0.592])
parser.add_argument("--look_offset", type=float, nargs=3, default=[0.0, 0.0, 0.020])
parser.add_argument("--focal_length", type=float, default=80.0)
parser.add_argument("--max_steps", type=int, default=0, help="0 でエピソード終端まで")
parser.add_argument("--skip_existing", action="store_true",
                    help="mp4 が既にある env を飛ばす（チャンク実行の再開用）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.ui")
enable_extension("isaacsim.util.debug_draw")

import tacex_tasks  # noqa: F401,E402  タスク登録
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def parse_envs(spec: str, n: int) -> list[int]:
    """'all' / '0,5,7' / '0-127' を env 番号の並びにする。"""
    if spec == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n]
    assert not bad, f"範囲外の env: {bad}（0..{n-1}）"
    return out


class Recording:
    """記録した npz を「再生に必要な配列」だけに絞って持つ層。"""

    def __init__(self, path: pathlib.Path):
        with np.load(path, allow_pickle=False) as z:
            need = ("joint_pos", "robot_root", "peg_root", "hole_root")
            missing = [k for k in need if k not in z.files]
            assert not missing, f"姿勢が記録されていない: {missing}（--log_pose つきで撮り直す）"
            self.joint_pos = z["joint_pos"]
            self.robot_root = z["robot_root"]
            self.peg_root = z["peg_root"]
            self.hole_root = z["hole_root"]
            # 穴原点 → 観測フレーム（開口面側）の局所オフセット。カメラの狙いはここ。
            # 元スクリプトの tip_off = fixed_pos_obs_frame - fixed_pos と同じ量。
            self._tip_off = (z["hole_obs_frame"][0] - z["hole_pos"][0])   # [N,3]
            self._done = z["done"]
            self.meta = json.loads(str(z["meta_json"]))
        self.num_envs = self.joint_pos.shape[1]

    def end(self, env: int) -> int:
        idx = np.where(self._done[:, env])[0]
        return int(idx[0]) + 1 if len(idx) else self._done.shape[0]

    def tip_off(self, env: int) -> np.ndarray:
        return self._tip_off[env]

    def hole_world(self, env: int) -> np.ndarray:
        return self.hole_root[0, env, :3]


class CameraRecorder:
    """`render_failure_cases.py` と同じ方式（カメラは動かさず焦点距離で寄せる）。

    ここではカメラが触覚を変えても構わない——**方策を回していない**ので影響しない。
    """

    def __init__(self, sim, eye, look_at, res, focal_length: float):
        import omni.replicator.core as rep
        self.sim = sim
        cam = rep.create.camera(position=tuple(eye), look_at=tuple(look_at),
                                focal_length=focal_length)
        self.rp = rep.create.render_product(cam, tuple(res))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annot.attach([self.rp])
        try:
            import isaaclab.sim as sim_utils
            cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
            cfg.func("/World/RecDome", cfg)
        except Exception as exc:  # noqa: BLE001
            print("LIGHT_FAIL", type(exc).__name__, exc, flush=True)
        self.frames: list[np.ndarray] = []
        self.duplicated = 0

    def _grab(self):
        a = np.asarray(self.annot.get_data())
        return a[:, :, :3].copy() if a.size else None

    def warmup(self, n: int) -> None:
        for _ in range(n):
            self.sim.render()
        self._grab()

    def capture(self) -> None:
        for _ in range(3):
            self.sim.render()
            fr = self._grab()
            if fr is not None and fr.max() > 10:
                self.frames.append(fr)
                return
        if self.frames:
            self.frames.append(self.frames[-1])
            self.duplicated += 1

    def is_blank(self) -> bool:
        """撮れていない env を機械で拾う。中身のある画は標準偏差が二桁になる。"""
        if not self.frames:
            return True
        return float(np.asarray(self.frames[len(self.frames) // 2]).std()) < 3.0

    def settle(self, tries: int = 6, per: int = 24) -> bool:
        """画が出るまで描画を回す。

        env ごとにカメラを作り足すと、**新しい render product の1枚目が一様色**のことがある
        （2026-08-17 実測: 固定 warmup だけだと 128env 中 2割が一様色になった）。
        中身のある画が1枚取れるまで待ってから本番の撮影に入る。
        """
        for _ in range(tries):
            for _ in range(per):
                self.sim.render()
            fr = self._grab()
            if fr is not None and float(np.asarray(fr).std()) >= 3.0:
                return True
        return False

    def write(self, path: pathlib.Path, fps: int) -> None:
        import imageio
        if not self.frames:
            print("REPLAY_DONE frames=0", flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(path).replace(".mp4", "_preview.png"),
                        self.frames[len(self.frames) // 2])
        imageio.mimwrite(str(path), self.frames, fps=fps, quality=9)

    def dispose(self) -> None:
        """render product を毎 env 作り直すので、使い終わったら必ず捨てる。"""
        try:
            self.annot.detach([self.rp])
        except Exception:  # noqa: BLE001
            pass
        try:
            self.rp.destroy()
        except Exception:  # noqa: BLE001
            pass
        self.frames = []


class Player:
    """記録の姿勢を毎フレーム書き戻す層。物理は進めない（重力で落ちないように）。"""

    def __init__(self, u, rec: Recording, device):
        self.u, self.rec, self.device = u, rec, device
        self.zeros_v = torch.zeros((rec.num_envs, rec.joint_pos.shape[2]), device=device)

    def _t(self, a) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)

    def apply(self, t: int) -> None:
        r = self.rec
        self.u._robot.write_joint_state_to_sim(self._t(r.joint_pos[t]), self.zeros_v)
        self.u._robot.write_root_pose_to_sim(self._t(r.robot_root[t]))
        self.u._held_asset.write_root_pose_to_sim(self._t(r.peg_root[t]))
        self.u._fixed_asset.write_root_pose_to_sim(self._t(r.hole_root[t]))
        self.u.scene.write_data_to_sim()


def shoot(u, rec: Recording, play: "Player", env_i: int, label: str) -> dict:
    """1 env ぶん撮る。カメラは env ごとに作り直す（穴の位置が env ごとに違うため）。"""
    n = rec.end(env_i) if args.max_steps == 0 else min(rec.end(env_i), args.max_steps)
    hole, tip = rec.hole_world(env_i), rec.tip_off(env_i)
    eye = tuple(float(hole[i] + tip[i] + args.eye_offset[i]) for i in range(3))
    look_at = tuple(float(hole[i] + tip[i] + args.look_offset[i]) for i in range(3))
    play.apply(0)
    r = CameraRecorder(u.sim, eye=eye, look_at=look_at, res=args.res,
                       focal_length=args.focal_length)
    settled = r.settle(per=max(1, args.warmup))
    for t in range(0, n, max(1, args.stride)):
        play.apply(t)
        r.capture()
    blank = r.is_blank()
    out = pathlib.Path(args.outdir) / f"{label}.mp4"
    if not blank:
        r.write(out, args.fps)
    nf, dup = len(r.frames), r.duplicated
    # ⚠ **render product を破棄してはいけない。** 破棄すると次の env 以降の annotator が
    #    空を返し続け、2本目からすべて BLANK になる（2026-08-17 実測: 破棄あり 1/3 本・
    #    破棄なし 3/3 本）。env ごとにカメラを足して溜めたまま進める。
    r.frames = []
    print(f"REPLAY_DONE env={env_i} label={label} frames={nf} duplicated={dup} "
          f"steps={n} stride={args.stride} settled={int(settled)} "
          f"{'BLANK（画が出ない・mp4 は書かない）' if blank else f'mp4={out}'}", flush=True)
    return {"env": env_i, "label": label, "frames": nf, "blank": blank, "steps": n,
            "settled": settled}


def main() -> int:
    rec = Recording(pathlib.Path(args.signals))
    assert (args.env is None) != (args.envs is None), "--env か --envs のどちらか一方を指定する"
    if args.envs is not None:
        targets = parse_envs(args.envs, rec.num_envs)
        labels = [args.label_fmt.format(env=i) for i in targets]
    else:
        targets, labels = [args.env], [args.label or f"p27_rp_env{args.env:03d}"]
    if args.skip_existing:
        keep = [(i, lab) for i, lab in zip(targets, labels)
                if not (pathlib.Path(args.outdir) / f"{lab}.mp4").is_file()]
        print(f"SKIP_EXISTING 既にある {len(targets) - len(keep)} 本を飛ばす", flush=True)
        targets = [i for i, _ in keep]
        labels = [lab for _, lab in keep]
    if not targets:
        print("REPLAY_ALL_DONE 撮るものが無い", flush=True)
        simulation_app.close()
        return 0
    print(f"REPLAY signals={args.signals} envs={len(targets)} num_envs={rec.num_envs} "
          f"stride={args.stride}", flush=True)

    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=rec.num_envs)
    cfg.tactile_in_obs = True
    cfg.tactile_placebo = False
    cfg.seed = int(rec.meta.get("seed", 0))
    env = gym.make(TASK, cfg=cfg, render_mode=None)
    env.reset()
    u = env.unwrapped
    play = Player(u, rec, u.device)

    done = [shoot(u, rec, play, i, lab) for i, lab in zip(targets, labels)]
    blank = [d["env"] for d in done if d["blank"]]
    print(f"REPLAY_ALL_DONE 撮れた {len(done) - len(blank)}/{len(done)} 本"
          f"／カメラが外れた env: {blank}", flush=True)
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
