"""plan24 L0（入力アブレーション）/ L2a（上限測定）の実行ドライバ。

**数値を出すだけ**で判定はしない（plan24 §3 の判定式は凍結済み・本スクリプトは触れない）。

冪等性: 出力 jsonl に同じセル（layer/cond/noise_mm/n_envs/checkpoint）の status=OK 行が既にあれば
Isaac を起動せずに終了する（`--force` で再実行）。env の seed を固定するので、同じ引数なら同じ数値が出る。

上流 TacEx は一切変更しない。アブレーションは**方策へ渡す直前の観測テンソル**を
`scripts/plan24/ablation.py` で書き換えて実現する（obs[:, 19:403] = 触覚 384 次元）。

既知の地雷（`reports/tactile_peg_learning_raw_2026-08-05.md` §5.2）:
  1. `clip_actions` は `agent_cfg["params"]["env"]`（=1.0）から取る。`params["config"]` だと既定 100 になり
     行動が桁違いのまま env に入って成功率が 60 倍ずれる。
  2. 成功は `extras["ep_succeeded_vec"]` を step 直後（env 内の自動リセット前の値）で読む。

使い方（1プロセス1条件。同一プロセスで2つ目の env を作るとハングする実測がある）:

    python scripts/plan24/run_l0_l2a.py --layer L0 --cond cond_A --noise_mm 1.0 \
        --checkpoint <arm_on の .pth> --out scripts/plan24/results/l0.jsonl \
        --dump_mean scripts/plan24/results/tactile_mean_1mm.npy
    python scripts/plan24/run_l0_l2a.py --layer L0 --cond cond_B --noise_mm 1.0 \
        --checkpoint <arm_on の .pth> --mean_npy scripts/plan24/results/tactile_mean_1mm.npy \
        --out scripts/plan24/results/l0.jsonl
    python scripts/plan24/run_l0_l2a.py --layer L2a --noise_mm 0.0 \
        --checkpoint <arm_off の .pth> --out scripts/plan24/results/l2a.jsonl
"""

import argparse
import json
import os
import math
import pathlib
import subprocess
import sys
import time
import traceback

try:
    import uipc  # noqa: F401  # AppLauncher より前に読む（後だと access violation）
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ablation import TACTILE_DIM, TACTILE_START, TactileAblation, ci_half, l2_rows  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TASK = "TacEx-Factory-PegInsert-Direct-v0"
# スタックの場所は環境変数 ISAAC_STACK_ROOT で上書きできる（既定は D:\IsaacStack）。
# 新しい PC では置き場所が変わるので、直書きしない。
ISAAC_ROOT = pathlib.Path(os.environ.get("ISAAC_STACK_ROOT", r"D:\IsaacStack"))
AGENT_CFG = ISAAC_ROOT / "TacEx/source/tacex_tasks/tacex_tasks/factory/agents/rl_games_ppo_cfg.yaml"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--layer", choices=["L0", "L2a"], required=True)
parser.add_argument("--cond", choices=["cond_A", "cond_B", "cond_C"], default="cond_A",
                    help="L0 の条件（L2a では cond_A 固定＝無改変）")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--noise_mm", type=float, required=True, help="obs_rand.fixed_asset_pos [mm]")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=1, help="1 env あたりのエピソード数")
parser.add_argument("--seed", type=int, default=0, help="env の乱数シード（冪等性のため固定）")
parser.add_argument("--perm_seed", type=int, default=0, help="cond_C のシャッフル置換のシード")
parser.add_argument("--mean_npy", default=None, help="cond_B が使う触覚平均ベクトル(.npy)")
parser.add_argument("--dump_mean", default=None, help="このランの触覚平均ベクトルを .npy へ書く")
parser.add_argument("--out", required=True)
parser.add_argument("--force", action="store_true", help="既に同じセルがあっても再実行する")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

TACTILE_ARM = args.layer == "L0"   # L0 は触覚ありアーム(403次元) / L2a は対照アーム(19次元)
if args.layer == "L2a":
    args.cond = "cond_A"


def cell_key_of(rec: dict) -> tuple:
    return (rec.get("layer"), rec.get("cond"), float(rec.get("noise_mm", -1)),
            rec.get("n_envs"), rec.get("checkpoint"))


def already_done(out_path: pathlib.Path, key: tuple) -> bool:
    if not out_path.is_file():
        return False
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "OK" and cell_key_of(rec) == key:
            return True
    return False


def git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --- 冪等性: Isaac を起動する前に既存セルを確認して抜ける -----------------
OUT_PATH = pathlib.Path(args.out)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CELL = {
    "layer": args.layer,
    "cond": args.cond,
    "noise_mm": args.noise_mm,
    "n_envs": args.num_envs,
    "checkpoint": args.checkpoint,
}
if already_done(OUT_PATH, cell_key_of(CELL)) and not args.force:
    print("[skip] 既に同じセルの OK 行がある:", json.dumps(CELL, ensure_ascii=False), flush=True)
    raise SystemExit(0)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> int:
    out_path = OUT_PATH
    rec = dict(CELL)
    rec.update({
        "episodes_per_env": args.episodes,
        "seed": args.seed,
        "perm_seed": args.perm_seed if args.cond == "cond_C" else None,
        "tactile_in_obs": TACTILE_ARM,
        "git_hash": git_hash(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    try:
        import numpy as np
        import torch
        import yaml
        import gymnasium as gym

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.ui")
        enable_extension("isaacsim.util.debug_draw")

        import tacex_tasks  # noqa: F401  # タスク登録
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner

        mean_vec = None
        if args.cond == "cond_B":
            if not args.mean_npy:
                raise ValueError("cond_B には --mean_npy が要る")
            mean_vec = np.load(args.mean_npy)
            rec["mean_npy"] = args.mean_npy
            rec["mean_npy_mean"] = float(mean_vec.mean())
        ablation = TactileAblation(args.cond, mean_vec=mean_vec, perm_seed=args.perm_seed)

        cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
        cfg.tactile_in_obs = TACTILE_ARM
        cfg.tactile_placebo = False
        cfg.seed = args.seed
        m = args.noise_mm / 1000.0
        cfg.obs_rand.fixed_asset_pos = [m, m, m]      # 観測（と行動基準フレーム）の穴位置ノイズ [m]
        rec["obs_rand_fixed_asset_pos"] = list(cfg.obs_rand.fixed_asset_pos)

        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        rec["observation_space"] = int(np.prod(np.array(cfg.observation_space).ravel()))

        agent_cfg = yaml.safe_load(AGENT_CFG.read_text(encoding="utf-8"))
        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["minibatch_size"] = args.num_envs
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args.checkpoint
        # 学習側 train.py と同じ導出（params.env であって params.config ではない）。地雷1。
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        rec["clip_obs"] = clip_obs
        rec["clip_actions"] = clip_actions

        wrapped = RlGamesVecEnvWrapper(env, "cuda:0", clip_obs, clip_actions)
        vecenv.register("IsaacRlgWrapper", lambda cfg_name, nenv, **kw: RlGamesGpuEnv(cfg_name, nenv, **kw))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                              "env_creator": lambda **kw: wrapped})

        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(args.checkpoint)
        player.reset()

        obs = wrapped.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = player.get_batch_size(obs, 1)
        if player.is_rnn:
            player.init_rnn()

        steps = int(u.max_episode_length) * args.episodes
        n_ep_done, n_ep_succ = 0, 0
        tac_sum, tac_sq, tac_cnt = None, None, 0
        pert_sum, orig_sum, pert_cnt = 0.0, 0.0, 0
        t0 = time.time()
        for _ in range(steps):
            with torch.inference_mode():
                obs = player.obs_to_torch(obs)
                if TACTILE_ARM:
                    block = obs[:, TACTILE_START:TACTILE_START + TACTILE_DIM]
                    if tac_sum is None:
                        tac_sum = torch.zeros(TACTILE_DIM, dtype=torch.float64, device=block.device)
                        tac_sq = torch.zeros_like(tac_sum)
                    tac_sum += block.double().sum(dim=0)
                    tac_sq += (block.double() ** 2).sum(dim=0)
                    tac_cnt += block.shape[0]
                    orig = block.double().clone()
                    obs = ablation.apply(obs)          # ★方策へ渡す直前に置換する
                    # [addendum1 F] 置換の実効量。摂動が効いていない可能性を数値で残すため必須。
                    new = obs[:, TACTILE_START:TACTILE_START + TACTILE_DIM].double()
                    pert_sum += float(l2_rows(new - orig).sum().item())
                    orig_sum += float(l2_rows(orig).sum().item())
                    pert_cnt += orig.shape[0]
                act = player.get_action(obs, is_deterministic=True)
                obs, _, dones, infos = wrapped.step(act)
                if player.is_rnn and player.states is not None:
                    for s in player.states:
                        s[:, dones, :] = 0.0
                # 地雷2: 自動リセット前のエピソード成功フラグを読む
                vec = infos.get("ep_succeeded_vec") if isinstance(infos, dict) else None
                if vec is None:
                    vec = u.extras.get("ep_succeeded_vec")
                d = dones.nonzero(as_tuple=False).flatten()
                if len(d) and vec is not None:
                    n_ep_done += len(d)
                    n_ep_succ += int(vec[d].bool().sum().item())
        rec["sec"] = round(time.time() - t0, 1)
        rec["steps"] = steps

        if tac_sum is not None and tac_cnt:
            mean = (tac_sum / tac_cnt).cpu().numpy()
            var = (tac_sq / tac_cnt).cpu().numpy() - mean ** 2
            rec["tactile_obs_mean"] = float(mean.mean())
            rec["tactile_obs_std_over_dims"] = float(mean.std())
            if args.dump_mean:
                dump = pathlib.Path(args.dump_mean)
                dump.parent.mkdir(parents=True, exist_ok=True)
                np.save(dump, mean)
                dump.with_suffix(".json").write_text(json.dumps({
                    "source": "run_l0_l2a.py --dump_mean",
                    "cond": args.cond, "noise_mm": args.noise_mm, "checkpoint": args.checkpoint,
                    "n_samples": int(tac_cnt), "dim": int(mean.shape[0]),
                    "mean_of_means": float(mean.mean()),
                    "min": float(mean.min()), "max": float(mean.max()),
                    "within_dim_std_mean": float(np.sqrt(np.clip(var, 0, None)).mean()),
                }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                rec["dump_mean"] = str(dump)

        p = n_ep_succ / n_ep_done if n_ep_done else float("nan")
        rec.update({
            "n": n_ep_done, "n_success": n_ep_succ, "success_rate": p,
            "ci_half": ci_half(p, n_ep_done) if n_ep_done else None,
            # [addendum1 F] 置換前後の平均 L2 ノルム（cond_A は無改変なので 0.0）
            "perturb_l2": (pert_sum / pert_cnt) if pert_cnt else 0.0,
            "orig_l2": (orig_sum / pert_cnt) if pert_cnt else None,
            "status": "OK",
        })
    except Exception as exc:  # noqa: BLE001
        rec.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)[:400],
                    "tb_tail": traceback.format_exc().splitlines()[-6:]})
    finally:
        # Isaac は stdout をバッファするので、数値は必ずファイルへ書く
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("[plan24]", json.dumps(rec, ensure_ascii=False)[:600], flush=True)
    return 0 if rec.get("status") == "OK" else 1


if __name__ == "__main__":
    import os

    code = main()
    sys.stdout.flush()
    os._exit(code)
