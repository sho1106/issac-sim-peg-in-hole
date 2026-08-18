"""plan24 L0 入力アブレーション: 方策へ渡す直前の観測を書き換える純関数群。

上流 TacEx（`D:\\IsaacStack\\TacEx`）は一切変更しない。書き換えは評価ドライバ側で行う。
本モジュールは配列演算だけで構成し、Isaac ランタイムに依存しない（numpy でも torch でも動く）ので
そのまま pytest で検査できる。

観測の並び（上流 `factory_env._get_observations` を読んで確認・2026-08-10）:

    obs[:,  0:19]  = cfg.obs_order + prev_actions（触覚なしアームと完全に同一）
    obs[:, 19:403] = 両指の触覚 RGB [32,32,6] を 4x4 平均プーリング→平坦化した 384 次元

条件（plan24 §3 L0 の表から転記。意味の解釈はしない）:

    cond_A  無改変
    cond_B  触覚 384 次元を訓練時の平均値（次元ごとの定数）に置換
    cond_C  触覚 384 次元を別エピソードの同ステップ値に置換（env 軸のシャッフル）
"""

from __future__ import annotations

import math

import numpy as np

#: 触覚ブロックの開始次元と幅（上流の実装に一致させる。ここが単一の定義箇所）
TACTILE_START = 19
TACTILE_DIM = 384

CONDS = ("cond_A", "cond_B", "cond_C")


def ci_half(p: float, n: int) -> float:
    """95%CI の半幅 = 1.96*sqrt(p(1-p)/n)（plan24 §3 で凍結した式）。"""
    if n <= 0:
        raise ValueError(f"n は正でなければならない: {n}")
    return 1.96 * math.sqrt(p * (1.0 - p) / n)


def l2_rows(a):
    """[N, D] の各行の L2 ノルムを返す（numpy/torch 共通）。

    plan24 addendum 1 §F の `perturb_l2`（置換前後の平均 L2 ノルム）の計算に使う。
    """
    if a.ndim != 2:
        raise ValueError(f"[N, D] の2次元が要る: ndim={a.ndim}")
    return (a * a).sum(axis=-1) ** 0.5


def fixed_derangement(n: int, seed: int) -> np.ndarray:
    """固定シードから「自分自身を指さない」置換（derangement）を作る。

    cond_C は「別エピソードの同ステップ値」なので、env i が自分の触覚を受け取ってはいけない。
    seed が同じなら常に同じ置換になる（ドライバの冪等性）。
    """
    if n < 2:
        raise ValueError(f"derangement には n>=2 が要る: {n}")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    # 不動点を後続要素と入れ替えて潰す（最後の1つは前の要素と交換する）
    for i in range(n):
        if perm[i] == i:
            j = i + 1 if i + 1 < n else i - 1
            perm[i], perm[j] = perm[j], perm[i]
    if np.any(perm == np.arange(n)):  # 交換で新たな不動点ができた場合の保険
        for i in range(n):
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
    return perm.astype(np.int64)


def _is_torch(a) -> bool:
    return type(a).__module__.split(".")[0] == "torch"


def _copy(a):
    return a.clone() if hasattr(a, "clone") else a.copy()


def _as_like(values: np.ndarray, ref):
    """numpy 配列を ref（numpy/torch・device・dtype）に合わせて変換する。"""
    if _is_torch(ref):
        import torch  # torch テンソルが来たときだけ読む

        return torch.as_tensor(values, dtype=ref.dtype, device=ref.device)
    return np.asarray(values, dtype=ref.dtype)


class TactileAblation:
    """観測テンソル [N, D] の触覚ブロックだけを条件に従って置換する。

    副作用を持たない（入力は変更せず、置換後の新しい配列を返す）。
    """

    def __init__(
        self,
        cond: str,
        mean_vec: np.ndarray | None = None,
        perm_seed: int = 0,
        start: int = TACTILE_START,
        dim: int = TACTILE_DIM,
    ) -> None:
        if cond not in CONDS:
            raise ValueError(f"未知の条件: {cond}（{CONDS} のいずれか）")
        if cond == "cond_B":
            if mean_vec is None:
                raise ValueError("cond_B には触覚の平均ベクトル（mean_vec）が要る")
            mean_vec = np.asarray(mean_vec, dtype=np.float64).reshape(-1)
            if mean_vec.shape[0] != dim:
                raise ValueError(f"mean_vec の長さが {mean_vec.shape[0]}（期待 {dim}）")
        self.cond = cond
        self.mean_vec = mean_vec
        self.perm_seed = perm_seed
        self.start = start
        self.dim = dim
        self._perm_cache: dict[int, np.ndarray] = {}

    @property
    def stop(self) -> int:
        return self.start + self.dim

    def perm(self, n_env: int) -> np.ndarray:
        if n_env not in self._perm_cache:
            self._perm_cache[n_env] = fixed_derangement(n_env, self.perm_seed)
        return self._perm_cache[n_env]

    def _check(self, obs) -> None:
        if obs.ndim != 2:
            raise ValueError(f"観測は [N, D] の2次元でなければならない: ndim={obs.ndim}")
        if obs.shape[1] < self.stop:
            raise ValueError(
                f"観測の次元 {obs.shape[1]} が触覚ブロック [{self.start}:{self.stop}] に足りない"
                "（触覚ありアームの観測を渡しているか確認する）"
            )

    def apply(self, obs):
        """方策へ渡す直前の観測に条件を適用して返す。"""
        if self.cond == "cond_A":
            return obs
        self._check(obs)
        out = _copy(obs)
        if self.cond == "cond_B":
            out[:, self.start : self.stop] = _as_like(self.mean_vec, obs)
            return out
        # cond_C: env 軸の固定 derangement で触覚ブロックだけを入れ替える
        block = _copy(obs[:, self.start : self.stop])
        perm = self.perm(obs.shape[0])
        if _is_torch(obs):
            import torch  # torch テンソルが来たときだけ読む

            idx = torch.as_tensor(perm, dtype=torch.long, device=obs.device)
        else:
            idx = perm
        out[:, self.start : self.stop] = block[idx]
        return out
