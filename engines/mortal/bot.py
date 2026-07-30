"""Mortal 子程序端:一行 MJAI 事件進,一行動作出。

**這個檔案跑在 Python 3.12 的 ``engines/mortal/.venv`` 裡,不是主程式的 3.14。**
它 import torch 與 libriichi,主環境兩者都沒有,也裝不起來。父程序那一端是
``src/mia/engine/mortal.py``,那邊只組命令列,完全不碰這些套件。

執行方式(父程序會自動帶好參數,手動測試時可以直接跑)::

    engines/mortal/.venv/bin/python engines/mortal/bot.py \
        --seat 0 --weights models/mortal_298k.pth

不依賴 cwd。上游 ``mortal/`` 目錄裡的模組彼此用同層 import(``model``、
``engine``),``libriichi.so`` 也放在那裡,所以那個目錄要進 ``sys.path``
—— 這裡自己插,而不是要求呼叫端先 ``cd`` 過去。

**不 import 上游的 ``prelude``。** 它會 ``import torch.utils.tensorboard``,
連帶要求裝 tensorboard —— 那是訓練才需要的東西,推論用不到。它做的另外兩件事
(把 stdin 設成 utf-8、關掉警告)在這裡沒有必要:父程序開子程序時就指定了
utf-8,警告本來就走 stderr,不會污染 stdout 上的協定。

與上游 ``mortal.py`` 的三個差異
-------------------------------
1. **握手**:載入權重要好幾秒,先送一行 ``hello`` 父程序才分得出「還在載入」
   與「已經死了」。
2. **每一行都回一行**:上游只在有動作時輸出,父程序無法區分「不需要動作」與
   「還在算」,只能等滿逾時。這裡不動作也回 ``{"type":"none"}``。
3. **例外轉成 JSON**:上游讓例外往上拋,父程序只會看到 stderr 上的 traceback
   與一個 returncode。這裡把它變成 ``{"type":"error","message":...}``,
   父程序可以指名道姓地報出是哪個引擎、在哪一手壞掉。

推論本身完全沒有改 —— 用的是上游的 ``MortalEngine`` 與 ``libriichi.mjai.Bot``。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

BOT_NAME = "mortal"

#: 上游 clone 的位置,相對於本檔案。libriichi.so 也在這裡面。
UPSTREAM = Path(__file__).resolve().parent / "Mortal" / "mortal"


def _emit(payload: dict) -> None:
    """往 stdout 送一行 JSON 並立刻 flush。

    flush 不能省。父程序是同步等回覆的,緩衝住就是雙方互等,表現出來會是
    一個沒有任何線索的逾時。
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _load(weights: str, seat: int, upstream: Path):
    """載入權重、組出 libriichi 的 Bot。回傳 ``(bot, 模型標籤)``。

    import 寫在函式內而非檔案開頭:torch 載入要好幾秒,失敗訊息也長,放在這裡
    才能被 :func:`main` 的 try 包住、轉成一行 JSON 送回父程序。開頭 import 的話
    父程序只會拿到 stderr 上的 traceback 與一個非零的 returncode。
    """
    sys.path.insert(0, str(upstream))

    import torch
    from engine import MortalEngine
    from libriichi.mjai import Bot
    from model import DQN, Brain

    state = torch.load(weights, weights_only=True, map_location=torch.device("cpu"))
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    num_blocks = cfg["resnet"]["num_blocks"]
    conv_channels = cfg["resnet"]["conv_channels"]

    if "tag" in state:
        tag = state["tag"]
    else:
        stamp = datetime.fromtimestamp(state["timestamp"], tz=UTC).strftime("%y%m%d%H")
        tag = f"mortal{version}-b{num_blocks}c{conv_channels}-t{stamp}"

    brain = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])

    engine = MortalEngine(
        brain,
        dqn,
        version=version,
        is_oracle=False,
        device=torch.device("cpu"),
        enable_amp=False,
        # 快速評估會在「明顯只有一個選擇」時跳過網路推論。跳過的那些手就沒有
        # Q 值可以回報,而 UI 要靠 meta 解釋「為什麼這樣打」,所以關掉。
        # 代價是每手都跑一次網路,實測約 14 ms,可以接受。
        enable_quick_eval=False,
        enable_rule_based_agari_guard=True,
        name=BOT_NAME,
    )
    return Bot(engine, seat), tag


def main() -> int:
    parser = argparse.ArgumentParser(description="Mortal MJAI bot(子程序端)")
    parser.add_argument("--seat", type=int, required=True, choices=range(4), help="自己的座位 0~3")
    parser.add_argument("--weights", required=True, help="權重 .pth 的路徑")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=UPSTREAM,
        help=f"上游 Mortal 的 mortal/ 目錄,預設 {UPSTREAM}",
    )
    args = parser.parse_args()

    try:
        bot, tag = _load(args.weights, args.seat, args.upstream)
    except Exception as exc:  # noqa: BLE001 - 任何載入失敗都要讓父程序看得懂
        _emit({"type": "error", "message": f"載入失敗: {exc}", "traceback": traceback.format_exc()})
        return 1

    _emit({"type": "hello", "name": BOT_NAME, "seat": args.seat, "detail": tag})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            reaction = bot.react(line)
        except Exception as exc:  # noqa: BLE001 - 同上,一手算壞不該讓整場沒有訊息地結束
            _emit(
                {
                    "type": "error",
                    "message": f"react 失敗: {exc}",
                    "event": line[:500],
                    "traceback": traceback.format_exc(),
                }
            )
            continue
        # react 回 None 代表這一手不需要動作 —— 補一行 none,維持一進一出
        _emit(json.loads(reaction) if reaction else {"type": "none"})

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
