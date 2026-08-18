"""F3挙動観察: 学習済みFORGE方策を摂動条件で評価（再学習なし・成功率収集）。
--mu       : ペグ/穴の摩擦係数を上書き（学習時0.75）
--pos_noise: 穴位置観測ノイズ[m]を上書き（学習時0.001=±1mm）
計画08 F3。エピソード境界(全env同時タイムアウト)ごとに extras['successes'] を収集。
"""
import argparse, sys
import h5py  # noqa: F401  GUI/rendering kit との DLL 競合回避（headlessでも無害）
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--mu", type=float, default=None)
parser.add_argument("--pos_noise", type=float, default=None)
parser.add_argument("--belief_residual", type=str, default=None, help="residual_table.npz(残差観測モデル)")
parser.add_argument("--oracle_mu", type=float, default=None, help="P-i: b(r,mu)をμ真値で引くオラクル条件")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    # --- 摂動の適用（cfg直接書き換え・spawn前）---
    applied = {}
    # DRタスクを評価するときは friction DR を無効化し固定μで評価する（DRは学習時のみの機構）
    if getattr(env_cfg, "friction_dr_range", None) is not None:
        env_cfg.friction_dr_range = None
        applied["dr_disabled_for_eval"] = True
    if args_cli.mu is not None:
        env_cfg.task.held_asset_cfg.friction = args_cli.mu
        env_cfg.task.fixed_asset_cfg.friction = args_cli.mu
        applied["mu"] = args_cli.mu
    if args_cli.pos_noise is not None:
        env_cfg.obs_rand.fixed_asset_pos = [args_cli.pos_noise] * 3
        applied["pos_noise"] = args_cli.pos_noise
    if args_cli.belief_residual is not None:
        env_cfg.belief_residual_npz = args_cli.belief_residual
        applied["belief_residual"] = True
    if args_cli.oracle_mu is not None:
        env_cfg.belief_oracle_mu = args_cli.oracle_mu
        applied["oracle_mu"] = args_cli.oracle_mu
    print(f"EVAL_APPLIED {applied} (defaults: friction=0.75, pos_noise=0.001)", flush=True)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions,
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda n, a, **kw: RlGamesGpuEnv(n, a, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner(); runner.load(agent_cfg)
    agent = runner.create_player(); agent.restore(resume_path); agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    ep_rates = []
    ep_steps = []  # 成功エピソード平均step(生存者平均・成功率と必ず併記)
    max_steps = 200 * (args_cli.episodes + 1)
    with torch.inference_mode():
        for _t in range(max_steps):
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _r, dones, infos = env.step(actions)
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            if dones.any():
                succ = None
                st = None
                if isinstance(infos, dict):
                    succ = infos.get("successes", None)
                    st = infos.get("success_times", None)
                    if succ is None and "episode" in infos:
                        succ = infos["episode"].get("successes", None)
                        st = infos["episode"].get("success_times", st)
                if succ is None:  # フォールバック: env内部の成功フラグ
                    succ = env.unwrapped.ep_succeeded.float().mean().item()
                else:
                    succ = float(succ.mean().item() if hasattr(succ, "mean") else succ)
                st = float(st.mean().item() if hasattr(st, "mean") else st) if st is not None else float("nan")
                ep_rates.append(succ)
                ep_steps.append(st)
                print(f"EP {len(ep_rates)}: success={succ:.4f} steps_succ_mean={st:.1f}", flush=True)
                if len(ep_rates) >= args_cli.episodes:
                    break
    n = args_cli.num_envs * len(ep_rates)
    mean = sum(ep_rates) / max(1, len(ep_rates))
    import math as _m
    stv = [x for x in ep_steps if not _m.isnan(x)]
    st_mean = sum(stv) / len(stv) if stv else float("nan")
    print(f"EVAL_RESULT {applied} episodes={len(ep_rates)} n_trials={n} success_mean={mean:.4f} steps_succ_mean={st_mean:.1f} per_ep={[round(r,3) for r in ep_rates]} per_ep_steps={[round(x,1) for x in ep_steps]}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
