# 学習済み RL 方策(Factory-PegInsert)を、明示カメラで描画して mp4 化する。
# IsaacLab 既定の RecordVideo は /OmniverseKit_Persp を使うが、本ヘッドレス構成では
# その render product がティックせず全フレーム黒になる。ここでは replicator で自前カメラの
# render product を作り、rgb annotator から各ステップのフレームを取得して動画化する。
#
# 重要: import 順序を play.py に合わせる。omni.replicator / imageio を冒頭で import すると
# omni 側の HDF5 DLL が先にロードされ、isaaclab の h5py 読込が 0xc0000139(entrypoint not found)で
# クラッシュする。よって replicator / imageio は env 構築後に遅延 import する。
import argparse
import sys

# Kit(AppLauncher)起動より前に h5py を import し、正しい HDF5 DLL をプロセスに先読みさせる。
# これをしないと rendering.kit が先に競合 HDF5 を読み、isaaclab の h5py 読込が
# 0xc0000139(entrypoint not found)でハードクラッシュする。
import h5py  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render trained rl_games policy with an explicit camera.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--warmup", type=int, default=24)
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--eye", type=float, nargs=3, default=[0.9, 0.9, 0.7])
parser.add_argument("--look_at", type=float, nargs=3, default=[0.0, 0.0, 0.25])
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--out", type=str, default="rl_policy.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以降の import 順序は play.py と一致させる（h5py DLL を正しい順序でロードするため）。"""

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402, F401
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate)
    vecenv.register("IsaacRlgWrapper",
                    lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # 穴(fixed asset)の世界位置を読み、挿入部にカメラを寄せる（既定look_atは穴を外す）。
    e0 = env.unwrapped
    if getattr(args_cli, "auto_frame", True) and hasattr(e0, "_fixed_asset"):
        H = e0._fixed_asset.data.root_pos_w[0].detach().cpu().tolist()
        peg = e0._held_asset.data.root_pos_w[0].detach().cpu().tolist()
        print(f"PEG_WORLD={tuple(round(v,4) for v in peg)}", flush=True)
        # ペグと穴の中点をやや上に注視し、斜め上前方から見下ろして挿入の沈み込みを写す
        look_at = (H[0], H[1], H[2] + 0.12)             # ロボット作業域
        eye = (H[0] + 0.55, H[1] - 0.78, H[2] + 0.62)   # 広い3/4俯瞰（Factory検証で有効だった画角）
        print(f"AUTO_FRAME hole_world={tuple(round(v,4) for v in H)} eye={tuple(round(v,3) for v in eye)} look_at={tuple(round(v,3) for v in look_at)}", flush=True)
    else:
        eye, look_at = tuple(args_cli.eye), tuple(args_cli.look_at)

    # --- h5py 等が正順でロード済の「後」に、描画系を遅延 import ---
    import numpy as np
    import imageio
    import omni.replicator.core as rep

    sim = env.unwrapped.sim
    cam = rep.create.camera(position=eye, look_at=look_at)
    rp = rep.create.render_product(cam, tuple(args_cli.res))
    annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annot.attach([rp])
    try:  # 全方向ドーム光で挿入部を照らす（暗いテーブルにペグが沈むのを防ぐ・funnelで実証済）
        import isaaclab.sim as sim_utils
        sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)).func(
            "/World/RecDome", sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)))
    except Exception as _e:
        print("LIGHT_FAIL", type(_e).__name__, _e, flush=True)

    def grab():
        a = np.asarray(annot.get_data())
        return a[:, :, :3].copy() if a.size else None

    for _ in range(args_cli.warmup):
        sim.render()
    _ = grab()

    out_path = args_cli.out if os.path.isabs(args_cli.out) else os.path.join(os.path.dirname(__file__), args_cli.out)
    frames = []
    nonblack = 0
    z_track = []  # (z_disp[mm], xy[mm]) env0。z_disp=ペグ底が穴底から何mm上か（0で完全挿入）
    with torch.inference_mode():
        for _t in range(args_cli.frames):
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)  # 決定論(mu)で評価
            obs, _, dones, _ = env.step(actions)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            sim.render()                  # annotatorは1回描画ごとに有効/空を交互に返す
            fr = grab()
            if fr is not None and fr.max() > 10:   # 空(黒)フレームは破棄→残りは全て有効＝ちらつき無し
                frames.append(fr)
                nonblack += 1
            try:  # 数値で挿入を追跡（peg_insertはheld/fixed原点がbase/targetと一致）
                zd = float((e0.held_pos[0, 2] - e0.fixed_pos[0, 2]).item()) * 1e3
                xy = float(torch.linalg.vector_norm(e0.held_pos[0, :2] - e0.fixed_pos[0, :2]).item()) * 1e3
                z_track.append((zd, xy))
            except Exception:
                pass

    succ0 = None
    try:
        succ = e0._get_curr_successes(e0.cfg_task.success_threshold, False)
        succ0 = bool(succ[0].item())
    except Exception as ex:
        print("SUCCESS_READ_FAIL", type(ex).__name__, ex, flush=True)
    if z_track:
        zd0 = z_track[0][0]
        zdmin = min(z for z, _ in z_track)
        zdlast, xylast = z_track[-1]
        print(f"INSERT_NUMERIC start_zdisp={zd0:.2f}mm min_zdisp={zdmin:.2f}mm "
              f"last_zdisp={zdlast:.2f}mm last_xy={xylast:.2f}mm success_env0={succ0} "
              f"(成功閾値 z<1.0mm & xy<2.5mm)", flush=True)
        try:
            import csv
            with open(os.path.join(os.path.dirname(__file__), "rl_insert_ztrack.csv"), "w", newline="") as cf:
                wcsv = csv.writer(cf); wcsv.writerow(["step", "z_disp_mm", "xy_mm"])
                for i, (z, xy) in enumerate(z_track):
                    wcsv.writerow([i, f"{z:.3f}", f"{xy:.3f}"])
            print(f"ZTRACK_CSV rows={len(z_track)}", flush=True)
        except Exception as ex:
            print("CSV_FAIL", ex, flush=True)

    if frames:
        mid = frames[len(frames) // 2]
        imageio.imwrite(out_path.replace(".mp4", "_preview.png"), mid)
        imageio.mimwrite(out_path, frames, fps=args_cli.fps, quality=8)
        print(f"RENDER_DONE frames={len(frames)} nonblack={nonblack} "
              f"mid_mean={mid.mean():.2f} mid_max={mid.max()} out={out_path}")
    else:
        print("RENDER_DONE frames=0 (annotator empty)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
