"""長時間 GPU ジョブの門番と記録。

**なぜ要るか**: RTX 5080 は 16.3GB しかなく、Isaac の学習が約 9.6GB を要求する。
走行中に別プロセス（ローカル LLM など）がモデルを載せると 16GB を超え、fps が
240→6 に落ちた末に `gpu.foundation.plugin.dll` でアクセス違反（exit 3221225477）を
起こす。実際に epoch 76/200 で 8.4 時間分の学習が消えた。

**なぜここにあるか**: この2つは元々 5 つのドライバに**コピペで配られていた**
（`run_e0_grid.py` / `run_plan17_stage1_grid.py` / `run_l2b_training.py` ほか）。
コピー元を直しても写した先が直らないので、1 箇所にまとめた。

使い方:

    from lib.gpu_guard import preflight_gpu, start_gpu_sampler

    blocked = preflight_gpu()
    if blocked:
        record({"exit_code": -2, "blocked_by": blocked})
        sys.exit(2)

    stop = [False]
    start_gpu_sampler(Path("results/gpu_samples.jsonl"), stop)
    try:
        ...  # 長時間ジョブ
    finally:
        stop[0] = True
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

#: これを超えて他プロセスが VRAM を掴んでいたら起動を見送る [MiB]
PREFLIGHT_MAX_OTHER_MIB = 2000

#: VRAM を記録する間隔 [秒]
GPU_SAMPLE_SEC = 60

_OLLAMA_HINT = (
    r'& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ps'
    r' → 行があれば stop <model>。プロセスの有無では判定できない'
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def gpu_used_mib() -> int | None:
    """GPU の使用 VRAM [MiB]。取得できなければ None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def preflight_gpu(max_other_mib: int = PREFLIGHT_MAX_OTHER_MIB) -> str | None:
    """他プロセスが VRAM を掴んでいたら理由の文字列を返す。問題なければ None。

    計測できないときは None を返す（門番をかけない）。測れないことを理由に
    ジョブを止めると、nvidia-smi が無い環境で何も動かせなくなるため。
    """
    used = gpu_used_mib()
    if used is None:
        return None
    if used > max_other_mib:
        return (
            f"他プロセスが VRAM を {used} MiB 使っている（上限 {max_other_mib}）。"
            f" ローカル LLM が載っていないか確認すること: {_OLLAMA_HINT}"
        )
    return None


def start_gpu_sampler(
    path: Path, stop_flag: list[bool], interval_sec: int = GPU_SAMPLE_SEC
) -> threading.Thread:
    """VRAM を定期記録するスレッドを起動する。

    失敗したときに「誰が食っていたか」を後から言えるようにするための記録。
    起動前の空き確認だけでは、走行中に載ってくる相手を捕まえられない。

    `stop_flag` は1要素のリスト。呼び出し側が `stop_flag[0] = True` で止める。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def loop() -> None:
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            while not stop_flag[0]:
                fh.write(json.dumps({"at": _now(), "gpu_used_mib": gpu_used_mib()}) + "\n")
                fh.flush()
                for _ in range(interval_sec):
                    if stop_flag[0]:
                        return
                    time.sleep(1)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th
