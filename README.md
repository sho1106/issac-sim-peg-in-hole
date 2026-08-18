# issac-sim-peg-in-hole

**Blackwell 世代の GPU（RTX 50 系 / sm_120）＋ Windows で、Isaac Sim の peg-in-hole 実験を再現するための最小キット。**

3つの実験を、学習・評価・サンプル実行まで通せる状態で置いてある。

| # | 実験 | 中身 | タスクID |
|---|---|---|---|
| 1 | **FORGE** | 力覚で挿入する RL。視覚を遮った盲目挿入・傾き制御・探索マップなど7種 | `Isaac-Forge-*` / `Isaac-Factory-CylInsert-*` |
| 2 | **TacEx（触覚なし）** | GelSight 触覚センサを載せた Factory。センサは動かすが観測には入れない対照 | `TacEx-Factory-PegInsert-Direct-v0` |
| 3 | **TacEx（触覚あり）** | 上と同じタスクで、触覚 RGB を方策の観測に加えたもの | 同上 ＋ `env.tactile_in_obs=True` |

> **注意**: 触覚実験のベースは **Factory** であって Forge ではない。TacEx は Isaac Lab の Factory を独自にコピーしたコード（944行）を持っており、そこに GelSight を足している。FORGE 側（力覚チャネルを持つ）とは別系統で、互いに独立している。

---

## 1. 動作要件

### 検証済みの構成（この組み合わせで実際に動かした）

| 項目 | バージョン | 備考 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5080 16GB（cc 12.0 / sm_120） | Blackwell 世代 |
| GPU ドライバ | 591.86 | GUI・録画・計算のすべてが動く。**610.74 は GUI と RTX レンダラが起動クラッシュする** |
| OS | Windows 11 Pro 26200 | |
| Python | 3.11.9 | Isaac Sim 5.1 は 3.11 系が前提 |
| Isaac Sim | 5.1.0.0 | pip 版 |
| Isaac Lab | v2.3.2 ＋ カスタムタスク7種 | フォーク（`ext/IsaacLab`） |
| TacEx | 上流 `adceed41` ＋ Blackwell 対応 | フォーク（`ext/TacEx`） |
| PyTorch | 2.7.0+cu128 | **CPU 版に巻き上がると全部壊れる** |
| CUDA | 12.8（torch 同梱） | `sm_120` を含んでビルドされている |
| numpy | 1.26.0 | **2.x に上げると Isaac が動かない** |

### VRAM

16.3GB で足りるが余裕はない。実測値:

- Isaac の学習（128 env）: 約 9.6GB
- 最小起動（2 env）: 約 6.1GB
- デスクトップ常駐（ブラウザ等）だけで 11GB 使っていた実測がある

同じ GPU で他の学習や LLM 推論を同時に走らせない。詳細は §6.7。

---

## 2. 構成

```
issac-sim-peg-in-hole/
├─ ext/
│   ├─ IsaacLab/          submodule: Isaac Lab フォーク（カスタムタスク7種）
│   └─ TacEx/             submodule: TacEx フォーク（Blackwell 対応＋触覚観測）
├─ scripts/
│   ├─ doctor.py          Blackwell 固有の地雷を数秒で判定（Isaac を起動しない）
│   ├─ verify_setup.py    Isaac を起動してタスク登録を確認（約10秒）
│   ├─ exp_forge/         FORGE の評価・録画
│   ├─ exp_tacex/         TacEx の評価
│   └─ lib/               共通モジュール
├─ env/
│   ├─ constraints.txt        壊してはいけない依存ピン（pip に必ず -c で渡す）
│   ├─ requirements.lock.txt  完全なスナップショット
│   └─ activate_isaac.ps1     venv 有効化＋EULA 受諾＋キャッシュを D: へ退避
├─ docs/
└─ results/                   実行時の出力（git 管理外）
```

### ディレクトリ名の制約（重要）

**`scripts/` の直下に、import 可能なパッケージと同じ名前のディレクトリを置いてはいけない。**

Python はスクリプトのあるディレクトリを `sys.path[0]` に置く。そこに `tacex/` があると、Python はそれを名前空間パッケージとして解決し、**pip で入れた本物の `tacex` を隠す**。しかも `PathFinder` は editable install の finder より先に走るので、正規のインストールが負ける。

実際に踏んだエラー:

```
ImportError: cannot import name 'GelSightSensor' from 'tacex' (unknown location)
```

このため実験ディレクトリには `exp_` を付けている（`exp_tacex` / `exp_forge`）。

---

## 3. セットアップ

### 3.1 リポジトリを置く

**短いパスに置くこと。** Isaac Lab はパスの深いファイル（最長で相対 144 文字）を含み、深い階層に clone すると `Filename too long` で checkout が途中で止まる。

```powershell
git config --global core.longpaths true
git clone --recurse-submodules https://github.com/sho1106/issac-sim-peg-in-hole.git D:\Common\github\issac-sim-peg-in-hole
```

> TacEx が入れ子で持つ submodule `libuipc` は**初期化しない**。1段だけで足りる。初期化するとソースビルドが走る。

### 3.2 venv と Isaac Sim

**pip を打つときは必ず `-c env\constraints.txt` を付ける。** 過去に stable-baselines3 が torch を CPU 版へ、`--force-reinstall` が numpy を 2.x へ巻き上げて環境を壊した実績がある。

```powershell
py -3.11 -m venv D:\IsaacStack\env_isaaclab
D:\IsaacStack\env_isaaclab\Scripts\Activate.ps1
python -m pip install --upgrade pip

# torch を先に入れる（順序が重要・下の注を読むこと）
pip install -c env\constraints.txt torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -c env\constraints.txt "isaacsim[all,extscache]==5.1.0.0" --extra-index-url https://pypi.nvidia.com
```

> **torch を先に入れる理由**: 逆順だと `isaacsim-core` が `torch==2.7.0` を要求し、constraints が `torch==2.7.0+cu128`（ローカル版）を強制するが、その wheel は pytorch の cu128 index にしか無いため ResolutionImpossible になる。先に入れておけば `==2.7.0` は `2.7.0+cu128` で満たされ、再取得されない。

### 3.3 Isaac Lab（フォーク）を editable で入れる

`isaaclab.bat --install` は**非対話 cmd で壊れる**（python 検出が空を返し、最後に EULA プロンプトが EOF で落ちる）。バッチを迂回して直接入れる。

```powershell
pip install setuptools==75.8.0 wheel toml
cd ext\IsaacLab
pip install -c ..\..\env\constraints.txt --no-build-isolation -e source\isaaclab
pip install -c ..\..\env\constraints.txt --no-build-isolation -e source\isaaclab_assets
pip install -c ..\..\env\constraints.txt --no-build-isolation -e source\isaaclab_mimic
pip install -c ..\..\env\constraints.txt --no-build-isolation -e source\isaaclab_rl
pip install -c ..\..\env\constraints.txt --no-build-isolation -e source\isaaclab_tasks
pip install -c ..\..\env\constraints.txt rl-games==1.6.5 gym==0.26.2 numba==0.59.1
```

> **`setuptools==75.8.0` の固定が要る理由**: setuptools 81+ が `pkg_resources` を撤去したのに、依存の `flatdict==4.0.1` の setup.py が `import pkg_resources` する。ビルド分離環境にも最新 setuptools が入るため、venv に古いものを入れるだけでは回避できない。`--no-build-isolation` と併用して venv 側を使わせる。

### 3.4 TacEx 用の venv（触覚実験を回す場合）

**FORGE 用の venv に TacEx を入れない。** 実験の再現性のため凍結する。robocopy で複製してから入れる。

```powershell
robocopy D:\IsaacStack\env_isaaclab D:\IsaacStack\env_tacex_isaac /E /MT:16   # exit code 1 = 正常
```

> **罠**: 複製した venv の `Scripts\pip.exe` は**元の venv の python を指したまま**。複製側へ入れるつもりが元を汚す。**必ず `<venv>\Scripts\python.exe -m pip` を使う。**

```powershell
$PY="D:\IsaacStack\env_tacex_isaac\Scripts\python.exe"
& $PY -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
& $PY -m pip install -e ext\TacEx\source\tacex_assets
& $PY -m pip install -e ext\TacEx\source\tacex
& $PY -m pip install -e ext\TacEx\source\tacex_tasks
& $PY -m pip install pyuipc
& $PY -m pip install h5py==3.11.0      # GUI を使うなら必須（§6.3）
```

> **`tacex_uipc` は `pip install -e` してはいけない。** setup.py が CMakeExtension を持ち、editable でも古いフォークの libuipc ソースビルドが走る。使うなら PYTHONPATH に置き、solver は上流 pyuipc を使う。
>
> **Isaac の venv に `nvidia-*-cu12` wheel を入れない。** torch が `site-packages/nvidia/*/bin` を DLL 探索に足すため、torch 同梱（cu12.8）と wheel（cu12.9）が混ざって Kit が起動時にクラッシュする。`doctor.py` がこれを検査する。

### 3.5 起動スクリプト

`env\activate_isaac.ps1` を `D:\IsaacStack\` へコピーする。venv 有効化に加えて、Omniverse の EULA を非対話で受諾し（無いと import 時に対話プロンプトで止まる）、Kit のキャッシュを D: へ逃がして C: の枯渇を防ぐ。

毎回の起動:

```powershell
. D:\IsaacStack\activate_isaac.ps1
$env:TORCHDYNAMO_DISABLE = "1"     # Windows に Triton が無いので torch.compile を切る（必須）
```

---

## 4. 動作確認

### 4.1 doctor（数秒・Isaac を起動しない）

```powershell
D:\IsaacStack\env_isaaclab\Scripts\python.exe    scripts\doctor.py
D:\IsaacStack\env_tacex_isaac\Scripts\python.exe scripts\doctor.py
```

`DOCTOR_OK` が出れば次へ。`NG` は「このまま進むと落ちる」、`WARN` は「用途によっては困る」。

### 4.2 verify_setup（約10秒・Isaac を起動する）

```powershell
cmd /c "D:\IsaacStack\env_isaaclab\Scripts\python.exe    -u scripts\verify_setup.py --profile forge > results\verify_forge.log 2>&1"
cmd /c "D:\IsaacStack\env_tacex_isaac\Scripts\python.exe -u scripts\verify_setup.py --profile tacex > results\verify_tacex.log 2>&1"
```

`SETUP_OK` が出れば構築成功。`MISS` が出たらフォークのブランチになっていない。

> **出力の取り方に2つ罠がある。** `python -u` を付けないと stdout がバッファされて結果が出ない。PowerShell の `2>&1 | Out-File` は native の stderr を壊すので `cmd /c "... > log 2>&1"` で受ける。

---

## 5. 3つの実験を回す

以下は共通の前提。`$FORGE_PY` = `D:\IsaacStack\env_isaaclab\Scripts\python.exe`、`$TACEX_PY` = `D:\IsaacStack\env_tacex_isaac\Scripts\python.exe`。

環境変数は §3.5 の3行を先に通しておく。**`full_experiment_name` は必ず指定する**（§6.6）。

### 5.1 FORGE

```powershell
# 学習
cmd /c "$FORGE_PY -u ext\IsaacLab\scripts\reinforcement_learning\rl_games\train.py `
    --task Isaac-Forge-PegInsertBlind-Direct-v0 --num_envs 128 --headless --seed 0 `
    --max_iterations 200 agent.params.config.full_experiment_name=my_run > train.log 2>&1"

# 評価（学習済み .pth の成功率を測る）
cmd /c "$FORGE_PY -u scripts\exp_forge\forge_eval.py `
    --task Isaac-Forge-PegInsertBlind-Direct-v0 --checkpoint <.pth> `
    --num_envs 128 --episodes 4 --headless > eval.log 2>&1"

# サンプル実行（動画）。--out は絶対パスで渡す（§6.4）
cmd /c "$FORGE_PY -u scripts\exp_forge\render_rl_policy.py `
    --task Isaac-Forge-PegInsertBlind-Direct-v0 --checkpoint <.pth> `
    --num_envs 1 --seed 0 --frames 300 --res 960 540 --fps 30 `
    --out D:\...\results\sample.mp4 --headless > render.log 2>&1"
```

評価は摂動を掛けられる。`--mu <摩擦係数>`（学習時 0.75）・`--pos_noise <m>`（学習時 0.001 ＝ ±1mm）。

### 5.2 TacEx（触覚なし・対照アーム）

**`--enable_cameras` が要る**（GelSight センサがカメラ経路を使うため）。触覚なしアームもセンサ自体は動かす。こうすると2アームの計算負荷が揃い、速度差が結果に混入しない。

```powershell
# 学習
cmd /c "$TACEX_PY -u ext\TacEx\scripts\reinforcement_learning\rl_games\train.py `
    --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 128 --headless --enable_cameras `
    --seed 0 --max_iterations 200 `
    agent.params.config.full_experiment_name=my_arm_off > train.log 2>&1"

# 評価（L2a = 対照アーム 19次元）
cmd /c "$TACEX_PY -u scripts\exp_tacex\run_l0_l2a.py --layer L2a --noise_mm 1.0 `
    --checkpoint <.pth> --num_envs 256 --episodes 1 --out results\eval.jsonl > eval.log 2>&1"
```

### 5.3 TacEx（触覚あり）

学習コマンドに `env.tactile_in_obs=True` を足すだけ。観測は既存19次元の**後ろに**触覚384次元（8×8×6・平均プーリング）が連結され、403次元になる。既存の並びは変えていない。

```powershell
# 学習
cmd /c "$TACEX_PY -u ext\TacEx\scripts\reinforcement_learning\rl_games\train.py `
    --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 128 --headless --enable_cameras `
    --seed 0 --max_iterations 200 env.tactile_in_obs=True `
    agent.params.config.full_experiment_name=my_arm_on > train.log 2>&1"

# 評価（L0 = 触覚ありアーム 403次元）
cmd /c "$TACEX_PY -u scripts\exp_tacex\run_l0_l2a.py --layer L0 --cond cond_A --noise_mm 1.0 `
    --checkpoint <.pth> --num_envs 256 --episodes 1 --out results\eval.jsonl > eval.log 2>&1"
```

関連する cfg フラグ（すべて既定 False）:

| フラグ | 意味 |
|---|---|
| `env.tactile_in_obs` | 両指の触覚 RGB を観測に加える |
| `env.tactile_pool` | 平均プーリングの縮小率（既定 4 → 32/4 = 8 → 8·8·6 = 384 次元） |
| `env.tactile_placebo` | 同じ次元・同じスケールの無情報ノイズに差し替える対照 |
| `env.tactile_raw_key` | 生の `[N,32,32,6]` も `tactile` キーで返す（CNN 版用） |

> **触覚アームの動画を撮るときは §6.9 を読むこと。** 方策を回しながら録画すると成功率が 56.25% → 3.91% に落ちる。

### 5.4 学習時間の目安（RTX 5080・128 env・実測）

| 実験 | 1 iter | 200 iter | 出典 |
|---|---|---|---|
| FORGE | 42.4 s | **約 2.4 h** | 800 iteration のフルラン（9.41 h）から割った値 |
| TacEx（触覚なし） | 76.1 s | **約 4.2 h** | 200 iteration のフルラン（253.6 分） |
| TacEx（触覚あり） | 77.1 s | **約 4.3 h** | 200 iteration のフルラン（257 分） |

**この表はフルランの総時間から割った値で統一してある。** 起動・warmup・カーネルのコンパイル・チェックポイント保存を含むので、予算取りにはこちらを使う。短い smoke から外挿した値（§7.2）はこれらより 20% ほど速く出るので、見積りには使わない。

触覚を入れても遅くならない（差 1.3%）。対照アームもセンサを動かしているため。

**GPU は1枚しかない前提で、これらは同時に回せない。** 3本を最初から学習すると合計約11時間かかる。動作確認だけなら §7 のように 10 iteration の smoke ＋ 既存チェックポイントでの評価で足りる。

---

## 6. Blackwell / Windows の地雷

実際に踏んだものだけを挙げる。

### 6.1 `import uipc` は `AppLauncher` より前に置く

Kit 起動後に `uipc` を import すると `PyInit_pyuipc` で access violation を起こし、**0.7 秒でプロセスごと落ちる**（終了コード 0xC0000005）。TacEx 側のスクリプトは全てこの前置きを持っている。

```python
try:
    import uipc  # noqa: F401
except Exception:
    pass
from isaaclab.app import AppLauncher   # これより前に置く
```

### 6.2 初回の `Engine("cuda")` は約100秒無反応

pyuipc の wheel は sm_89 ビルドなので、Blackwell では PTX の JIT コンパイルが走る。2回目以降は約2.4秒（`%APPDATA%\NVIDIA\ComputeCache`）。**固まったと誤診して殺さないこと。**

### 6.3 GUI モードだけ h5py で即死する

`ImportError: DLL load failed while importing _errors` ＋ `Windows fatal exception: 0xc0000139`。Kit 同梱 HDF5 と pip の h5py のミスマッチ。**`h5py==3.11.0`（HDF5 1.14.2）を入れる。** headless では起きない。

### 6.4 出力先は絶対パスで渡す

hydra が cwd を変えるため、`--out results\foo.mp4` のような相対パスはスクリプトのディレクトリ基準に解決され、`FileNotFoundError: The directory ...\scripts\exp_forge\results does not exist` になる。

### 6.5 `.ps1` に日本語を書くなら UTF-8 BOM 付きで保存する

PowerShell 5.1 は BOM の無いファイルを ANSI（Shift-JIS）として読む。日本語コメントが化けてパーサを壊し、**無関係な行の変数代入まで巻き込む**（`$repo='D:\...'` が `D:` になった実例がある）。

### 6.6 smoke 実行がチェックポイントを上書きする

実験名が固定なので、短い smoke でも `logs\rl_games\<Task>\test\nn\<Task>.pth` を上書きする。**必ず `agent.params.config.full_experiment_name=<別名>` を付ける。**

### 6.7 GPU の空きは `ollama ps` で確認する

Ollama は**サービスが常駐したままモデルだけロードされている**状態を取る。`Get-Process ollama` では判定できない。`ollama ps` の出力が空であることが唯一確実な確認。

走行中の学習の横で別プロセスが LLM を再ロードし、Isaac の 9.6GB と合わせて 16GB を超過。fps が 240→6 に落ちた末に `gpu.foundation.plugin.dll` でアクセス違反（exit 3221225477）、**epoch 76/200 で 8.4 時間分が消えた**実例がある。`doctor.py` が起動前にこれを検査する。

### 6.8 ロールアウトは再現しない

同一 seed・同一 δ・同一 checkpoint で 128 env を撮り直しても、**成否の一致は 116/128（90.6%）**。step 0 の時点で既にペグ位置が 0.08mm ずれる。初期条件（δ・穴位置）だけは完全一致する。

∴ **「同じエピソードを撮り直す」ことはできない。** 評価値の再現は厳密一致ではなく、既知範囲に収まるかで判定する。

### 6.9 触覚アームは録画すると成功率が落ちる

カメラ（replicator の render product）を作るだけで GelSight の画像が変わり、触覚アーム（403次元）の入力が学習時の分布から外れる。実測: **成功率 56.25% → 3.91%**。照明を切っても warmup を 0 にしても同じ 5/128 で、決定論的な差だった。step 0 の物理は完全一致なのに触覚だけ最大 0.263 ずれる。

∴ **カメラつきで方策を回した映像は、その方策の挙動を表さない。** 触覚アームの動画は「方策を回さず、記録した姿勢を書き戻して描画する」方式で撮る。触覚なしアーム（19次元）は触覚を見ないので影響しない。

---

## 7. 動作確認の実測（2026-08-19・RTX 5080 / driver 591.86）

このリポジトリの内容で、3実験すべての **学習・評価・サンプル実行**を実際に通した記録。

### 7.1 環境検査

| 対象 | doctor.py | verify_setup.py |
|---|---|---|
| `env_isaaclab`（FORGE） | `DOCTOR_OK`（WARN 1: h5py 3.16.0） | `SETUP_OK` 7/7・4.3秒 |
| `env_tacex_isaac`（TacEx） | `DOCTOR_OK`（WARN 0） | `SETUP_OK` 3/3・10.2秒 |

GPU の認識:

```
torch 2.7.0+cu128 (cuda 12.8)
arch list: sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
gpu[0]: NVIDIA GeForce RTX 5080 (cc 12.0)
```

### 7.2 学習（smoke・10 iteration・128 env）

| 実験 | 1 iter | 観測次元 | 既存記録との比較 |
|---|---|---|---|
| FORGE | **40.77 s**（`Training time: 407.69秒`） | — | 記録 42.36 s/iter・**−3.8%** |
| TacEx（触覚なし） | **60.0 s** | `RunningMeanStd: (19,)` | 記録 76.1 s/iter・−21%（下の注） |
| TacEx（触覚あり） | **59.0 s** | `RunningMeanStd: (403,)` = 19 + 触覚384 | 記録 77.1 s/iter・−23%（同上） |

**触覚を入れても遅くならない**（59.0 vs 60.0 s/iter・差 1.7%）。既存記録の差 1.3% と同じ傾向。対照アームも GelSight センサ自体は動かしているため、計算負荷が揃う。

> **注: この2つは測り方が違うので直接比較できない。**
>
> - 既存記録 = **200 iteration の総所要時間**（253.6 分 / 257 分）を 200 で割った値。起動・warmup・CUDA カーネルのコンパイル・チェックポイント保存が全部含まれる
> - こちらの実測 = **10 iteration の smoke** の `fps total` から算出した値。起動と warmup はこの計算に入っていない
>
> 償却の対象が違うぶん smoke 側が速く出る。**200 iter の見積りには既存記録の側を使うほうが安全**（このリポジトリの §5.4 の表も既存記録を採っている）。1 iter が既存記録より 21〜23% 速い理由は、この条件差で説明できる範囲を超えているかどうかまでは切り分けていない。

観測次元が `(19,)` / `(403,)` と分かれることで、**触覚の配線が cfg フラグで正しく切り替わっている**ことが直接確認できる。

### 7.3 評価（既存の学習済みチェックポイント）

| 実験 | 条件 | 実測 | 既存記録 | 差 |
|---|---|---|---|---|
| FORGE（盲目挿入） | `forge_blind_f6`・**μ=0.75**・±1mm・n=512 | **0.9414** | 0.9453 | **−0.39pt** |
| TacEx（触覚なし） | `plan22_arm_off`・±1mm・n=256 | **0.9492**（243/256） | 0.949 | **±0.0pt** |
| TacEx（触覚あり） | `plan22_arm_on`・±1mm・n=256 | **0.9063**（232/256） | **0.9062（232/256）** | **成功本数まで一致** |

FORGE のエピソード別: `0.961 / 0.938 / 0.938 / 0.930`、成功時の平均ステップ 29.3。

**既存記録の出典**（値だけで引くと取り違えるので、行とチェックポイントまで特定する）:

| 実測値 | 出典 | 行の中身 |
|---|---|---|
| 0.9453 | `e0_reeval_2026-07-13.csv:12` | `f6_mu075,0.75,,logs\rl_games\Forge\forge_blind_f6\nn\Forge.pth,0.9453,512,OK` |
| 0.949 | `0029-plan22-tactile-naive-wiring-rejected.md:18` | ±1mm・触覚なし **94.9% ±2.7** |
| 0.9062 | `plan24_raw_2026-08-10.md:292` | ±1mm・`cond_A` **90.62%（232/256）**・CI [87.05, 94.20] |

> **⚠ `0.9453` は3か所に別の意味で存在する。** ①F6盲目 μ0.75 の成功率（上表・これを引いている）②E5a Oracle μ0.75 の成功率（`e5a_oracle_actor_raw_2026-07-13.md:26`・真値穴位置＋真μを与えた別の方策）③`budget_convergence_raw_2026-07-23.md:333` の `engaged_mean`（着座ではなく**係合**の別指標。同じ行の `success_mean` は 0.8770）。**値で照合せず、行とチェックポイントで照合すること。**

> **⚠ FORGE 盲目の成功率は μ で 21pt 動く。** 同じ `forge_blind_f6` で μ0.05→0.75 が **73.4% → 94.5%**（`e0_reeval_raw_2026-07-13.md:24-27`）。μ を書かない「盲目挿入 94.x%」は記録側が一意に定まらない。

触覚ありは **232/256 と成功本数まで一致**した。plan24 が同条件で再測定した値と同一で、丸め表記だけが違う（90.62% vs 90.63%）。

**定性的な結論も再現した**: ±1mm では触覚ありのほうが成功率が低い（−4.30pt。既存記録は −3.9pt）。触覚を観測に足すことが、この条件では利得にならないという既存の所見と一致する。

> **判定基準は厳密一致ではない。** §6.8 のとおりロールアウトは実行ごとに違う（成否一致 90.6%）。既知の値の周辺に収まるかで判定している。

### 7.4 サンプル実行

| 実験 | 方法 | 結果 |
|---|---|---|
| FORGE | `render_rl_policy.py`（方策を回しながら録画） | **300 フレーム・黒フレーム 0**・960×540・3.95MB |
| TacEx（触覚なし） | `render_failure_cases.py --arm off`（同上） | 960×540・6.3MB・`observation_space: 19` |
| TacEx（触覚あり） | `render_failure_cases.py --arm on --no_render --log_pose` → `replay_render.py`（**姿勢再生**） | 960×540・5.4MB・`observation_space: 403` |

触覚ありだけ2段階なのは §6.9 のため。方策を回しながら録画すると挙動そのものが変わるので、姿勢を記録してから書き戻して描画する。

生成物（`results/sample_tacex/`）:

```
verify_off.mp4 / verify_off_preview.png / verify_off_signals.npz / verify_off_meta.json
verify_on_signals.npz / verify_on_meta.json
verify_on_replay.mp4 / verify_on_replay_preview.png
```

`*_meta.json` に checkpoint・δ テーブル・`observation_space`・`clip_actions` が残るので、後から条件を突き合わせられる。

### 7.5 既存資産の保全

smoke 学習は `full_experiment_name` を別名にしたので、既存のチェックポイントは1つも壊れていない。

```
学習前 112 本 → 学習後 114 本（増分は verify_smoke_* 配下のみ）
forge_blind_f6/nn/Forge.pth の更新日時: Jul 6 07:35 のまま
```

---

## 8. ライセンス / 出自

- **Isaac Lab**: BSD-3-Clause（NVIDIA）。フォークは公式 v2.3.2 にカスタムタスク7種を追加したもの
- **TacEx**: 上流 [DH-Ng/TacEx](https://github.com/DH-Ng/TacEx)（arXiv 2411.04776）。フォークは Blackwell / Isaac Sim 5.1 対応と触覚観測の配線を追加したもの
