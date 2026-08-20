# 上流 TacEx への issue 草案（未投稿）

**状態: 草案。ユーザーの承認が出るまで投稿しない。**

投稿先候補: https://github.com/DH-Ng/TacEx/issues
（2026-08-20 時点で本件に該当する既存 issue は無い。open 9 / closed 3 を確認済み）

以下は英語の本文案。日本語の補足は `<!-- -->` で囲んでいるので、投稿時は残しても消してもよい。

---

## Title

`GelSight left/right sensors respond asymmetrically in the two-finger Factory task (right ~1/2 of left after contact)`

---

## Body

### Summary

In `TacEx-Factory-PegInsert-Direct-v0`, the **right** GelSight sensor responds
systematically weaker than the left one **after contact**. Before contact the two are symmetric.

I could not find the root cause and would appreciate a pointer from someone who knows how the
gripper asset was authored. I have a self-contained reproduction script and have ruled out
four plausible explanations, listed below.

| | left | right | ratio L/R |
|---|---|---|---|
| height map min distance | 27.8828 mm | 28.1561 mm | diff **+0.2733 mm** |
| indentation depth (Taxim formula) | 0.6293 mm | 0.3682 mm | **1.71** |
| tactile RGB temporal std (full episode) | 0.01928 | 0.00676 | **2.85** |
| tactile RGB temporal std (steps 0–7, before contact) | 0.00291 | 0.00303 | **0.96** |
| visible pixels (depth < far clip) | 17.32 % | 11.39 % | — |

Reproduced independently on two machines:

| machine | GPU | driver | height map diff | indentation ratio |
|---|---|---|---|---|
| A | RTX 5080 (sm_120) | 591.86 | +0.2625 mm | 1.71 |
| B | RTX 4060 (sm_89) | 591.86 | +0.2498 mm | 1.698 |

Both with Isaac Sim 5.1.0.0 / Isaac Lab 2.3.2 / torch 2.7.0+cu128 / Windows 11.
Upstream TacEx at `adceed41` plus a few Windows-compat patches that do not touch the sensor path.

### Why this matters

A real pair of GelSight sensors is expected to respond symmetrically. A policy trained here
learns to rely on the left finger more than the right, which will not transfer to hardware.

### Environment

- Isaac Sim 5.1.0.0 (pip), Isaac Lab 2.3.2, Python 3.11.9, torch 2.7.0+cu128, numpy 1.26.0
- Windows 11 Pro 26200
- TacEx `adceed41`
- Task `TacEx-Factory-PegInsert-Direct-v0`, `num_envs=2..128`, headless, `--enable_cameras`

### Reproduction

Create the env, reset, step with zero actions, and read both sensors. No policy or checkpoint
needed — the asymmetry is present from the first frames.

```python
u.gsmini_left.data.output["height_map"]     # (N, 32, 32), camera depth in mm
u.gsmini_right.data.output["height_map"]
```

A ready-made script is at
<!-- 投稿時に自分のリポジトリの URL を入れる -->
`scripts/exp_tacex/check_gelsight_symmetry.py` in
https://github.com/sho1106/issac-sim-peg-in-hole — it prints `SYMMETRY_OK` /
`SYMMETRY_ASYMMETRIC` and writes a json.

### What I ruled out

**1. It is not the gelpad link transform.**
`physx_rigid_gelpads.usd` does have an asymmetry — the gelpads sit 0.2954 mm off mirror symmetry
along the grip axis, while the sensor cases (and their cameras) are mirror-symmetric to 1.3 µm.
But editing the gelpad **link** xform changes nothing: PhysX uses the `FixedJoint`'s
`physics:localPos0`, and the same 0.2954 mm value is written there too.

```
/panda/gelsight_mini_case_left/FixedJointCaseLeft    localPos0.z = -0.024254449
/panda/gelsight_mini_case_right/FixedJointCaseRight  localPos0.z = -0.023959097
```

**2. The gelpads are not what the depth camera sees.**
Moving the right gelpad by **11 mm** (20x the asymmetry) produced **bit-identical** height maps
(same `num_envs`, seed and initial conditions). What the camera images is the **peg**, seen as a
horizontal band; the gelpad does not appear in the depth buffer.

**3. It is not the other joints or the meshes.**
The finger prismatic joints are properly mirror-symmetric (same `localPos`, axis, limits and
drive; `localRot` differs only by the mirror sign). The gelpad meshes are mirror-identical
(same vertex count, thickness, and min/mean/median of the projected point distribution to 1e-4).
`gsmini_*_joint` has a 0.163 mm asymmetry in its own z, but projected on the grip axis that is
−0.0001 mm. Composing the whole joint chain gives +0.2982 mm, essentially the same as the xform
figure — it does not account for the ~0.49 mm residual.

**4. Making the gelpads symmetric makes it worse.**
Setting the joint difference to 0 (geometrically symmetric pads) doubled the asymmetry:

| gelpad joint difference | height map diff | indentation ratio |
|---|---|---|
| 0.0000 mm (**geometrically symmetric**) | **+0.5397 mm** | **8.721** |
| 0.1500 mm | +0.3877 mm | 2.753 |
| **0.2954 mm (as shipped)** | **+0.2498 mm** | **1.698** |
| 0.4000 mm | +0.1547 mm | 1.339 |
| 0.5441 mm | +0.1135 mm | 1.229 |
| 0.8000 mm | −0.0079 mm | 0.988 |

Least squares: `hm_diff = -0.6767 * joint_diff + 0.4865` (R² = 0.9483, zero crossing 0.7190 mm).

So there is a **separate ~0.49 mm asymmetry in the system, and the shipped gelpad offset is
partially cancelling it.**

Two caveats I want to be explicit about:

- The **finger joints saturate around joint_diff ≈ 0.4 mm** — they pin to the same values for
  0.40 / 0.5441 / 0.80. A naive "wider gap → peg moves" model only holds below that.
- The system is **deterministic but multistable**. Re-running the same USD three times gives
  bit-identical results, but two files with the *same* joint difference (0.0002 mm apart) gave
  height map differences 0.114 mm apart, settling at different grasp equilibria. So the link
  xform does matter — it sets the spawn pose, which selects which equilibrium is reached.
  **Extrapolating the fit to pick a "correct" offset is not sound.**

### Where I got stuck

Reading the runtime physics poses (`body_pos_w`, projected on the axis joining the two cameras,
63.4641 mm apart):

| body | projection [mm] | asymmetry |
|---|---|---|
| `gelsight_mini_case_left` / `_right` | −31.7321 / +31.7320 | −0.0001 (symmetric) |
| `gelpad_left` / `_right` | −7.4776 / +7.7728 | +0.2952 |
| `panda_leftfinger` / `panda_rightfinger` | −0.1407 / −0.0178 | −0.1229 |
| held peg (`root_pos_w`) | **+0.1295** | — |

The peg sits **0.1295 mm toward the right camera** (0.1333 mm on the other machine), so
geometrically the right camera should be **~0.26 mm closer** to it. The measured height map says
the right is **~0.27 mm farther**. The sign is inverted.

With a geometrically symmetric gelpad placement (joint_diff = 0) the fingers still end up
asymmetric: left −0.2871 mm, right +0.1286 mm. The gripper command is symmetric
(`ctrl_target_joint_pos[:, 7:9] = 0.0` for both) and the joints are symmetric, so this points at
the reset / grasp construction rather than the asset geometry. The peg is placed analytically
relative to `panda_fingertip_centered`, which sits −0.0822 mm off the midpoint of the two
sensor cameras — the Franka frame and the added sensors have no reason to coincide.

One possibility I could not confirm: the right camera may not be seeing the nearest point of the
peg. Its visible area is 11.39 % vs 17.32 %, and the centroid of the visible region is ~5 px
(of 32) off-centre, while the left one lands where the pinhole model predicts. But a 0.13 mm
lateral peg offset only accounts for ~0.8 % of the field of view, so that does not explain a 5 px
shift on its own.

### Questions

1. Is the 0.2954 mm gelpad offset in `physx_rigid_gelpads.usd` intentional (e.g. compensating
   for something), or an authoring artefact?
2. Is there a known reason the two sensor cameras would frame the grasped object differently
   despite mirror-symmetric placement?
3. Related: `gelsight_sensor.py:238` notes that `clipping range doesn't matter for existing
   camera prim -> only applied when camera is spawned # TODO fix?`. The cameras in this asset are
   pre-authored, so the cfg's clipping range is ignored. Is that expected?

Happy to run further experiments — I have both machines set up and a scripted A/B harness.

<!--
投稿時のチェック:
- リポジトリ URL を実際のものに置き換える（scripts/exp_tacex/check_gelsight_symmetry.py へのリンク）
- 「machine A / B」の書き方でホスト名・内部IP・ユーザー名を出さないこと（現状は出していない）
- 数値は docs/gelsight-right-sensor-asymmetry.md と一致していること
- 断定していないこと（原因は未特定と明記している）を確認
-->
