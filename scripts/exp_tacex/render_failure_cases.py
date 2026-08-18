"""触覚あり／なしの2アームを**同じ条件**で走らせ、失敗時の動作を軌跡として記録する。

目的は「触覚を足すと何が変わって性能が下がるのか」を、成功率でなく**動作**で見ること。
`reports/tactile_peg_learning_raw_2026-08-05.md` §5.3 の −16.0pt（±3mm）が何をしている
エピソードで生じているかを、対にした軌跡で突き合わせる。

**両アームを対にするための3点**（どれか欠けると比較が成立しない）
1. `--delta_mode table` … env ごとに固定した δ（穴位置の信念誤差）を**毎リセット後に注入し直す**。
   env が自動リセットすると δ は σ で引き直されるので、1回だけの注入では2本目以降が別条件になる。
2. 同じ `--seed` … 初期の手先位置（±20mm）・把持ずれ・穴姿勢がそろう。
3. 解析は**各 env の1本目のエピソードだけ**を使う（`motion_metrics.classify_episodes`）。
   2本目以降は終了時刻がアームごとに違い、初期条件がそろわない。

⚠ 本ドライバが出す数値は**事前登録なし＝探索的**。証拠格付けしない。

実行例:
  # 対にした本測定（描画なし・全 env の軌跡を保存）
  D:\\IsaacStack\\env_tacex_isaac\\Scripts\\python.exe -u scripts/plan24/render_failure_cases.py \\
      --label pair_on --arm on --delta_mode table --num_envs 128 --steps 150 --no_render
  # 代表例の録画（1 env・δ を指定）
  ... --label dx5 --arm on --delta_mode fixed --delta_mm 5 0 0 --num_envs 1 --steps 149
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

# --- Kit 起動より前に済ませること（順序を崩すとハードクラッシュする） ---------------
# 1) EULA ほかの環境変数。素の python から起動すると対話プロンプトで即死する。
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

# 2) h5py を先に読ませる。rendering.kit が競合 HDF5 を先に載せると isaaclab の h5py 読込が
#    0xc0000139 でハードクラッシュする（render_rl_policy.py の教訓）。
import h5py  # noqa: F401,E402

# 3) uipc を AppLauncher より前に読む。tacex_tasks が uipc 版タスクを条件付き登録するため、
#    後回しにすると起動 1 秒で access violation で落ちる（plan22 raw §1 の改変 6）。
try:
    import uipc  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from isaaclab.app import AppLauncher  # noqa: E402

TASK = "TacEx-Factory-PegInsert-Direct-v0"
AGENT_CFG = ISAAC_ROOT / "TacEx/source/tacex_tasks/tacex_tasks/factory/agents/rl_games_ppo_cfg.yaml"
CKPT = {
    "on": str(ISAAC_ROOT / "TacEx/logs/rl_games/Factory/plan22_arm_on/nn/last_Factory_ep_200_rew_342.65515.pth"),
    "off": str(ISAAC_ROOT / "TacEx/logs/rl_games/Factory/plan22_arm_off/nn/last_Factory_ep_200_rew_331.76602.pth"),
}
HERE = pathlib.Path(__file__).resolve().parent
TACTILE_SLICE = (19, 403)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", required=True, help="この条件の名前（出力ファイル名になる）")
parser.add_argument("--arm", default="on", choices=["on", "off"],
                    help="on=触覚あり(403次元) / off=対照(19次元)")
parser.add_argument("--checkpoint", default=None, help="省略時は --arm から選ぶ")
parser.add_argument("--delta_mode", default="table", choices=["table", "fixed", "none"],
                    help="table=env毎に固定した δ を毎リセット注入 / fixed=--delta_mm を全envへ / none=注入しない")
parser.add_argument("--delta_mm", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="fixed モードで注入する δ=[x,y,z]（mm）")
parser.add_argument("--delta_sigma_mm", type=float, default=3.0, help="table モードの σ")
parser.add_argument("--delta_seed", type=int, default=0, help="table モードの乱数種（両アームで同一にする）")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--seed", type=int, default=0, help="両アームでそろえる（初期手先位置・把持ずれが同一になる）")
parser.add_argument("--no_render", action="store_true", help="描画を省く（本測定用）")
parser.add_argument("--no_rec_light", action="store_true",
                    help="録画用のドームライトを足さない（切り分け用。画は暗くなる）。"
                         "**触覚ありの成功率低下はこの照明が原因ではない**——照明を外しても "
                         "成功率は 3.91%% のまま変わらなかった（2026-08-17 実測・"
                         "scripts/plan27/results/repro.json）。触覚ありの録画は "
                         "scripts/plan27/replay_render.py の姿勢再生で撮ること")
parser.add_argument("--log_tactile", action="store_true", help="触覚 384 次元も保存する（ファイルが大きくなる）")
parser.add_argument("--log_pose", action="store_true",
                    help="関節角とルート姿勢も保存する（後から姿勢再生で録画するため）")
parser.add_argument("--log_raw_steps", type=int, default=0,
                    help="全 env の触覚生像 [32,32,6] を先頭 N step だけ保存する（plan27 の CNN 読み出し用）。"
                         "0 で無効。全ステップ保存はファイルが巨大になるので接触前の窓だけを取る")
parser.add_argument("--render_env", type=int, default=0, help="録画・触覚生像を取る env の番号")
parser.add_argument("--res", type=int, nargs=2, default=[960, 540])
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--warmup", type=int, default=24)
# カメラ位置・注視点は**この組でのみ被写体に向く**（CameraRecorder の注記）。既定から動かさない。
parser.add_argument("--eye_offset", type=float, nargs=3, default=[0.547, -0.776, 0.592])
parser.add_argument("--look_offset", type=float, nargs=3, default=[0.0, 0.0, 0.020])
parser.add_argument("--focal_length", type=float, default=80.0,
                    help="寄りは焦点距離で作る（既定24mm→80mmで約3.3倍）。位置を動かすと画角が破綻する")
parser.add_argument("--outdir", default=str(HERE / "render"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True      # TacEx の触覚センサはカメラを使う＝必須
if args.checkpoint is None:
    args.checkpoint = CKPT[args.arm]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.ui")
enable_extension("isaacsim.util.debug_draw")

import tacex_tasks  # noqa: F401,E402  タスク登録
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from motion_metrics import make_delta_table  # noqa: E402  同ディレクトリの純関数


class DeltaInjector:
    """狙った δ を env に入れ、δ から派生する状態を reset 末尾と同じ式で作り直す層。

    `init_fixed_pos_obs_noise` は観測ノイズであると同時に**行動フレームのずれ**でもある
    （factory_env.py:912-919）。δ だけ差し替えて行動フレームを直さないと、
    方策が観測する基準位置と、行動が適用される基準位置が食い違う。

    さらに env は**リセットのたびに δ を引き直す**（同 :747-753）。1回だけ注入しても
    2本目以降のエピソードは別条件になるので、`reinject_after_reset` を毎ステップ呼ぶ。
    """

    def __init__(self, env_unwrapped, table_m: torch.Tensor):
        self.u = env_unwrapped
        self.table = table_m                       # [N,3] m 単位・env ごとに固定
        self.bounds = torch.tensor(env_unwrapped.cfg.ctrl.pos_action_bounds,
                                   device=env_unwrapped.device)

    def _set_delta_and_frame(self) -> None:
        u = self.u
        u.init_fixed_pos_obs_noise[:] = self.table
        u.fixed_pos_action_frame[:] = u.fixed_pos_obs_frame + u.init_fixed_pos_obs_noise

    def _repair_initial_actions(self, env_ids) -> None:
        """reset 直後の env だけ「動かない初期行動」を作り直す（進行中の env は触らない）。"""
        u = self.u
        pos_actions = u.fingertip_midpoint_pos[env_ids] - u.fixed_pos_action_frame[env_ids]
        pos_actions = pos_actions @ torch.diag(1.0 / self.bounds)
        u.actions[env_ids, 0:3] = pos_actions
        u.prev_actions[env_ids, 0:3] = pos_actions

    def inject_initial(self) -> None:
        self._set_delta_and_frame()
        self._repair_initial_actions(torch.arange(self.u.num_envs, device=self.u.device))

    def reinject_after_reset(self, dones: torch.Tensor) -> bool:
        """リセットされた env があれば δ を入れ直す。入れ直したら True。"""
        ids = torch.nonzero(dones.bool(), as_tuple=False).flatten()
        if ids.numel() == 0:
            return False
        self._set_delta_and_frame()
        self._repair_initial_actions(ids)
        return True


class SignalLog:
    """毎ステップの信号を貯めて npz に落とすだけの層（計算はしない）。

    軌跡は**全 env** を残す（動作の分類に要る）。触覚の生像だけは容量が大きいので
    `--render_env` の1本に限る。
    """

    PER_ENV = ("tcp_pos", "tcp_quat", "tcp_target", "peg_pos", "peg_quat",
               "hole_pos", "hole_obs_frame", "delta", "grasp_offset",
               "wrench", "joint_torque", "success", "done",
               # 姿勢再生で録画するために要る（--log_pose）。カメラを作ると触覚観測が変わり
               # 成功率が低下するので、録画は方策を回さず姿勢を書き戻して撮る。
               "joint_pos", "robot_root", "peg_root", "hole_root")

    def __init__(self, log_tactile: bool, log_raw_steps: int = 0):
        self.rows: dict[str, list] = {k: [] for k in self.PER_ENV}
        self.log_tactile = log_tactile
        self.log_raw_steps = log_raw_steps
        self.rows["tactile_raw_env"] = []
        if log_tactile:
            self.rows["tactile_obs"] = []
        if log_raw_steps > 0:
            self.rows["tactile_raw_all"] = []      # [min(T,N_steps), N, 32, 32, 6]

    def append(self, **kw) -> None:
        # None は「この step では記録しない」の意味（生像は先頭 N step だけ取るため）。
        # 詰めると配列が不揃いになって保存時に落ちる。
        for k in list(self.rows):
            if k in kw and kw[k] is not None:
                self.rows[k].append(kw[k])

    def save(self, path: pathlib.Path, meta: dict) -> dict:
        arrays = {k: np.asarray(v) for k, v in self.rows.items() if v}
        arrays["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return {k: list(a.shape) for k, a in arrays.items() if a.ndim > 0}


class CameraRecorder:
    """replicator の明示カメラ。IsaacLab 既定の RecordVideo は本構成で黒画になるため使わない。

    **画角の寄せ方（2026-08-14 の実測）**: この構成でカメラを被写体に向けられるのは
    `rep.create.camera(position, look_at)` の**特定の1組**だけで、同じ向きのまま距離を縮めると
    被写体を外す（rotation 明示でも `TiledCamera.set_world_poses_from_view` でも直らなかった）。
    ∴ **カメラは動かさず focal_length で寄せる**。詳細は knowledge/runbook-tacex-env.md §3.4。
    """

    def __init__(self, sim, eye, look_at, res, focal_length: float, rec_light: bool = True):
        import omni.replicator.core as rep
        self.sim = sim
        cam = rep.create.camera(position=tuple(eye), look_at=tuple(look_at), focal_length=focal_length)
        self.rp = rep.create.render_product(cam, tuple(res))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annot.attach([self.rp])
        # ⚠ **カメラを作ること自体**が GelSight の画像を変え、触覚アームでは方策の入力が
        #    学習時の分布から外れて成功率が 56.25% → 3.91% に低下する
        #    （2026-08-17 実測・scripts/plan27/results/repro.json）。
        #    照明を外しても warmup を 0 にしても同じ 5/128 だったので、**この照明は原因ではない**。
        #    触覚ありの録画は scripts/plan27/replay_render.py の姿勢再生で撮ること。
        if not rec_light:
            self.frames: list[np.ndarray] = []
            self.duplicated = 0
            return
        try:    # 暗い卓面にペグが沈むのを防ぐ
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
        """1ステップ1フレームを保証する（annotator は描画ごとに有効/空を交互に返す）。"""
        for _ in range(3):
            self.sim.render()
            fr = self._grab()
            if fr is not None and fr.max() > 10:
                self.frames.append(fr)
                return
        if self.frames:                      # 取れなければ直前フレームを複製して同期を守る
            self.frames.append(self.frames[-1])
            self.duplicated += 1

    def write(self, path: pathlib.Path, fps: int) -> None:
        import imageio
        if not self.frames:
            print("RENDER_DONE frames=0", flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(path).replace(".mp4", "_preview.png"), self.frames[len(self.frames) // 2])
        imageio.mimwrite(str(path), self.frames, fps=fps, quality=9)


def build_delta_table_m() -> np.ndarray:
    """注入する δ の表（m 単位・[N,3]）。両アームで同一になることが要件。"""
    if args.delta_mode == "table":
        return make_delta_table(args.num_envs, args.delta_sigma_mm, args.delta_seed) / 1000.0
    if args.delta_mode == "fixed":
        return np.tile(np.asarray(args.delta_mm, dtype=float) / 1000.0, (args.num_envs, 1))
    return np.zeros((args.num_envs, 3))


def build_env_and_player():
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
    cfg.tactile_in_obs = (args.arm == "on")
    cfg.tactile_placebo = False
    cfg.seed = args.seed
    m = args.delta_sigma_mm / 1000.0
    cfg.obs_rand.fixed_asset_pos = [m, m, m]

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    agent_cfg = yaml.safe_load(AGENT_CFG.read_text(encoding="utf-8"))
    agent_cfg["params"]["config"]["num_actors"] = args.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = args.num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    # 地雷: clip_actions は params["env"] 側から取る（params["config"] は既定 100.0 で桁が変わる）
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))

    wrapped = RlGamesVecEnvWrapper(env, "cuda:0", clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda cfg_name, nenv, **kw: RlGamesGpuEnv(cfg_name, nenv, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.reset()
    return env, wrapped, player, clip_obs, clip_actions


def main() -> int:
    outdir = pathlib.Path(args.outdir)
    meta = {
        "label": args.label, "task": TASK, "arm": args.arm, "checkpoint": args.checkpoint,
        "delta_mode": args.delta_mode, "delta_mm": list(args.delta_mm),
        "delta_sigma_mm": args.delta_sigma_mm, "delta_seed": args.delta_seed,
        "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed,
        "render_env": args.render_env, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "grading": "探索的（事前登録なし）＝証拠にしない",
        "pairing": "解析は各 env の1本目のエピソードのみ（2本目以降は初期条件がそろわない）",
    }
    env, wrapped, player, clip_obs, clip_actions = build_env_and_player()
    u = env.unwrapped
    meta["observation_space"] = int(u.cfg.observation_space)
    meta["clip_obs"], meta["clip_actions"] = clip_obs, clip_actions
    if clip_obs != float("inf"):
        print(f"WARN clip_obs={clip_obs} != inf: 観測の作り直しにクランプが要る", flush=True)

    obs = wrapped.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    table_np = build_delta_table_m()
    table = torch.tensor(table_np, dtype=torch.float32, device=u.device)
    injector = DeltaInjector(u, table)
    if args.delta_mode != "none":
        injector.inject_initial()
        obs = u._get_observations()["policy"]     # δ 差し替え後の観測を取り直す（clip_obs=inf 前提）
    meta["delta_table_mm_head"] = np.round(table_np[:4] * 1e3, 4).tolist()
    meta["delta_table_sha_head"] = float(np.abs(table_np).sum())
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    g = u.held_asset_pos_noise.detach().cpu().numpy() * 1000.0
    meta["grasp_offset_mm_env0"] = [float(v) for v in g[args.render_env]]
    print(f"INJECT mode={args.delta_mode} table_head_mm={meta['delta_table_mm_head']} "
          f"grasp_offset_env{args.render_env}_mm={np.round(g[args.render_env], 3).tolist()}", flush=True)

    rec = None
    if not args.no_render:
        tip_off = (u.fixed_pos_obs_frame[0] - u.fixed_pos[0]).detach().cpu().numpy()
        hole = u._fixed_asset.data.root_pos_w[args.render_env].detach().cpu().numpy() + tip_off
        eye = tuple(float(hole[i] + args.eye_offset[i]) for i in range(3))
        look_at = tuple(float(hole[i] + args.look_offset[i]) for i in range(3))
        meta["camera"] = {"eye": list(eye), "look_at": list(look_at),
                          "focal_length": args.focal_length}
        print(f"CAMERA eye={tuple(round(v, 4) for v in eye)} "
              f"look_at={tuple(round(v, 4) for v in look_at)} focal={args.focal_length}", flush=True)
        rec = CameraRecorder(u.sim, eye=eye, look_at=look_at, res=args.res,
                             focal_length=args.focal_length,
                             rec_light=not args.no_rec_light)
        rec.warmup(args.warmup)

    log = SignalLog(args.log_tactile, args.log_raw_steps)
    n_reinject = 0
    t0 = time.time()
    for _t in range(args.steps):
        with torch.inference_mode():
            action = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, infos = wrapped.step(action)
            if isinstance(obs, dict):
                obs = obs["obs"]
            if player.is_rnn and player.states is not None:
                for s in player.states:
                    s[:, dones, :] = 0.0

            vec = infos.get("ep_succeeded_vec") if isinstance(infos, dict) else None
            if vec is None:
                vec = u.extras.get("ep_succeeded_vec")
            succ = (vec.detach().bool() if vec is not None
                    else torch.zeros(u.num_envs, dtype=torch.bool, device=u.device))

            def npa(x):
                return x.detach().float().cpu().numpy()

            tac_all = npa(u._get_tactile_obs()) if args.log_raw_steps > 0 else None
            log.append(
                tactile_raw_all=(tac_all.astype(np.float16)
                                 if (tac_all is not None and _t < args.log_raw_steps) else None),
                tcp_pos=npa(u.fingertip_midpoint_pos), tcp_quat=npa(u.fingertip_midpoint_quat),
                tcp_target=npa(u.ctrl_target_fingertip_midpoint_pos),
                peg_pos=npa(u.held_pos), peg_quat=npa(u.held_quat),
                hole_pos=npa(u.fixed_pos), hole_obs_frame=npa(u.fixed_pos_obs_frame),
                delta=npa(u.init_fixed_pos_obs_noise), grasp_offset=npa(u.held_asset_pos_noise),
                wrench=npa(u.applied_wrench), joint_torque=npa(u.joint_torque),
                joint_pos=(npa(u._robot.data.joint_pos) if args.log_pose else None),
                robot_root=(npa(u._robot.data.root_state_w[:, :7]) if args.log_pose else None),
                peg_root=(npa(u._held_asset.data.root_state_w[:, :7]) if args.log_pose else None),
                hole_root=(npa(u._fixed_asset.data.root_state_w[:, :7]) if args.log_pose else None),
                success=succ.cpu().numpy(), done=dones.detach().bool().cpu().numpy(),
                tactile_raw_env=npa(u._get_tactile_obs()[args.render_env]),
                tactile_obs=(obs[:, TACTILE_SLICE[0]:TACTILE_SLICE[1]].detach().float().cpu().numpy()
                             if (args.log_tactile and args.arm == "on") else None),
            )
            # δ の入れ直しは**記録の後**（記録は「その step で方策が置かれていた条件」を残す）
            if args.delta_mode != "none" and injector.reinject_after_reset(dones):
                n_reinject += 1
                obs = u._get_observations()["policy"]
        if rec is not None:
            rec.capture()

    meta["sec"] = round(time.time() - t0, 1)
    meta["n_reinject_steps"] = n_reinject
    meta["frames"] = len(rec.frames) if rec is not None else 0

    if rec is not None:
        rec.write(outdir / f"{args.label}.mp4", args.fps)
    shapes = log.save(outdir / f"{args.label}_signals.npz", meta)
    (outdir / f"{args.label}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"RUN_DONE label={args.label} arm={args.arm} sec={meta['sec']} "
          f"reinject_steps={n_reinject} npz={outdir / (args.label + '_signals.npz')}", flush=True)
    print("SIGNALS " + json.dumps(shapes, ensure_ascii=False), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
