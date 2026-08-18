"""失敗時の「動作」を軌跡から読むための純関数（Isaac 非依存・テスト対象）。

ここには**計算だけ**を置く。Isaac の起動・env の操作・描画は `render_failure_cases.py` の担当。
分けている理由は、方策を走らせずに分類ロジックを検査できるようにするため。

⚠ しきい値は**運用値**であって凍結した判定式ではない。本モジュールを使う解析は探索的。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 触覚アームと対照アームを同じ δ で比べるための既定 σ（mm）。plan22 S2 本測定の最大条件に合わせる。
DEFAULT_SIGMA_MM = 3.0


def make_delta_table(n_envs: int, sigma_mm: float = DEFAULT_SIGMA_MM, seed: int = 0) -> np.ndarray:
    """env ごとに固定の δ（穴位置の信念誤差・mm）を作る。

    同じ引数なら必ず同じ表を返す＝**両アームに同一の δ を与える**ための土台。
    分布は評価本測定と同じ「3軸独立の正規分布」。
    """
    if n_envs <= 0:
        raise ValueError(f"n_envs は正: {n_envs}")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma_mm, size=(n_envs, 3)).astype(np.float64)


def pos_actions_from_frame(fingertip_pos: np.ndarray, action_frame: np.ndarray,
                           bounds: np.ndarray) -> np.ndarray:
    """reset 末尾と同じ式で「動かない」初期行動を作る（factory_env.py の 912-919 行）。

    δ を差し替えたら行動フレームもこの初期行動も作り直さないと、
    方策が見る世界と行動が効く世界が食い違う。
    """
    return (fingertip_pos - action_frame) / np.asarray(bounds, dtype=float)


def first_done_step(done: np.ndarray) -> np.ndarray:
    """[T,N] の done から、env ごとの**最初の終了ステップ**を返す。終了しなければ T。"""
    done = np.asarray(done, dtype=bool)
    T = done.shape[0]
    any_done = done.any(axis=0)
    idx = np.argmax(done, axis=0)
    return np.where(any_done, idx, T)


def lateral_mm(peg_pos: np.ndarray, hole_pos: np.ndarray) -> np.ndarray:
    """ペグと穴の**横方向**距離 [mm]。入力は m 単位の [...,3]。"""
    d = np.asarray(peg_pos)[..., :2] - np.asarray(hole_pos)[..., :2]
    return np.linalg.norm(d, axis=-1) * 1e3


def depth_mm(peg_pos: np.ndarray, hole_pos: np.ndarray) -> np.ndarray:
    """ペグ原点が穴原点より何 mm 上にあるか。**小さいほど深く入っている**。"""
    return (np.asarray(peg_pos)[..., 2] - np.asarray(hole_pos)[..., 2]) * 1e3


def force_n(wrench: np.ndarray) -> np.ndarray:
    """インピーダンス制御が出しているタスク空間の力の大きさ [N]（接触力の実測ではない）。"""
    return np.linalg.norm(np.asarray(wrench)[..., 0:3], axis=-1)


def tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """剛体の**軸が鉛直から何度傾いているか** [deg]。入力は (w,x,y,z) の [...,4]。

    回転行列の R[2,2] = 1 − 2(x²+y²) が「物体の z 軸と世界の z 軸の内積」なので、
    その arccos が傾き。ペグの受動傾き（β_passive 系の量）を軌跡から読むために使う。
    """
    q = np.asarray(quat_wxyz, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError(f"quat は (w,x,y,z) の4成分: {q.shape}")
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(n == 0.0, 1.0, n)
    cos = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@dataclass(frozen=True)
class MotionThresholds:
    """動作分類の運用しきい値（mm）。凍結判定式ではない。"""

    approach: float = 2.0     #: この横ずれ以内まで寄れたら「穴口まで来た」とみなす
    engage: float = 5.0       #: 初期高さからこれだけ下がれば「入りかけた」とみなす
    backout: float = 3.0      #: 最深からこれだけ戻れば「戻った」とみなす
    stall_span: float = 0.5   #: 終盤でこの範囲しか動かなければ「止まった」とみなす
    stall_window: int = 30    #: 終盤とみなすステップ数


#: 分類ラベル。順序は「失敗の深さ」順（浅い失敗ほど前）。
MOTION_LABELS = ("inserted", "not_approached", "approached_not_entered",
                 "entered_then_backed_out", "stalled_inside")


def classify_motion(lateral: np.ndarray, depth: np.ndarray, success: bool,
                    th: MotionThresholds = MotionThresholds()) -> str:
    """1エピソードの軌跡を、失敗の**しかた**で分類する。

    入力は既に1エピソード分に切り出された [T] 配列（mm 単位）。
    """
    lateral = np.asarray(lateral, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if lateral.ndim != 1 or depth.ndim != 1 or len(lateral) != len(depth):
        raise ValueError(f"lateral と depth は同じ長さの1次元: {lateral.shape} {depth.shape}")
    if len(lateral) == 0:
        raise ValueError("空のエピソードは分類できない")
    if success:
        return "inserted"

    drop = depth[0] - depth              # 正なら下がった
    if drop.max() < th.engage:           # そもそも入りかけていない
        return "not_approached" if lateral.min() > th.approach else "approached_not_entered"

    deepest = int(np.argmin(depth))
    if depth[-1] - depth[deepest] > th.backout:
        return "entered_then_backed_out"

    w = min(th.stall_window, len(depth))
    tail_span = max(depth[-w:].ptp(), lateral[-w:].ptp())
    return "stalled_inside" if tail_span < th.stall_span else "approached_not_entered"


def classify_episodes(peg_pos: np.ndarray, hole_pos: np.ndarray, done: np.ndarray,
                      success: np.ndarray, th: MotionThresholds = MotionThresholds()) -> list[str]:
    """[T,N,3] の軌跡から env ごとに**最初のエピソード**だけを切り出して分類する。

    最初のエピソードに限るのは、2本目以降は終了時刻がアームごとに違って
    初期条件がそろわなくなるため（対にして比べられるのは1本目だけ）。
    """
    lat = lateral_mm(peg_pos, hole_pos)      # [T,N]
    dep = depth_mm(peg_pos, hole_pos)        # [T,N]
    ends = first_done_step(done)             # [N]
    out = []
    for i in range(lat.shape[1]):
        e = max(1, int(ends[i]))
        out.append(classify_motion(lat[:e, i], dep[:e, i], bool(success[i]), th))
    return out


def paired_table(labels_a: list[str], labels_b: list[str]) -> dict[tuple[str, str], int]:
    """同じ初期条件で走らせた2アームのラベルを突き合わせた集計。

    キーは (アームA のラベル, アームB のラベル)。
    「対照は入ったのに触覚ありは入らなかった」エピソードを数えるために使う。
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"対にならない: {len(labels_a)} vs {len(labels_b)}")
    table: dict[tuple[str, str], int] = {}
    for a, b in zip(labels_a, labels_b):
        table[(a, b)] = table.get((a, b), 0) + 1
    return table
