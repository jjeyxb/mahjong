#!/usr/bin/env python3
"""開一個牌譜,一步一步按、一步截一張 —— 副露素材用,**不是 app 的功能**。

為什麼要有這個
--------------
``tools/capture.py --replay`` 錄得到牌譜,但兩件事很慢:要人坐在旁邊看,
而且牌譜沒有時間基準(整場紀錄在 ``fetchGameRecord`` 一開始一次送完),
只能事後用單調對齊去猜哪一幀是哪個狀態。

這裡改成**由我們決定畫面什麼時候前進**:按一次「下一步」、等動畫演完、截一張。
於是每張圖都知道自己是「按了幾次之後」,答案從紀錄直接算
(:func:`mia.groundtruth.paipu.replay_steps`),不必對齊、也不會掉幀。

與 ``tools/autoplay.py`` 同一個性質:替人按畫面,只為了錄素材。

用法
----
第一次先看介面長什麼樣(下一步按鈕在哪)::

    .venv/bin/python tools/paipu.py "<牌譜連結>" --probe

登入、等牌譜開起來,它每幾秒存一張圖到輸出目錄。拿圖找出按鈕之後::

    .venv/bin/python tools/paipu.py "<牌譜連結>" --next 0.95,0.60

輸出(``data/recordings/paipu-<時間>/``):

* ``ws.jsonl`` —— 封包,含 ``fetchGameRecord`` 的回應(答案從這裡算)
* ``record.bin`` —— 回應裡的整場紀錄,單獨存一份省得每次重解封包
* ``frames/NNNN.png`` 與 ``steps.jsonl`` —— 每張圖是按了幾次之後
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from autoplay import _press, viewport
from mia.calibration.canvas import Canvas
from mia.groundtruth.dump import DumpWriter
from mia.groundtruth.liqi import LiqiParseError, LiqiParser, MessageKind
from mia.groundtruth.paipu import decode_records
from mia.groundtruth.schema import LiqiSchema
from mia.utils.logging import logger, setup_logging
from mia.utils.paths import DATA_DIR

if TYPE_CHECKING:  # pragma: no cover
    from playwright.sync_api import Page

_FETCH_RECORD = ".lq.Lobby.fetchGameRecord"

#: 與 ``tools/gt.py`` 同一個設定檔目錄,登入過一次兩邊都不必再登。
DEFAULT_PROFILE = DATA_DIR / "live" / "chrome-profile"

#: 畫布。16:9 是必要的:按鈕座標是畫布的比例,非 16:9 雀魂會加黑邊
#: (而且 ``20260801-112437`` 那份 1.83 的素材整份讀不出來,牌桌排版會變)。
#: 取 1920x1080 而不是 autoplay 的 1280x720:副露牌很小,解析度直接決定認不認得出來。
DEFAULT_CANVAS = "1920x1080"

#: 連續幾張一模一樣就當作演完了。牌譜最後一步之後再按下一步,畫面不會再變。
#: 但結算面板、流局畫面也可能停好幾步不動,所以不能太小。
_SAME_LIMIT = 8

#: ``--probe`` 最多存幾張。
_PROBE_SHOTS = 30


@dataclass
class Stepper:
    """一步一步按牌譜。滿足 :class:`~mia.groundtruth.cdp.AutoPlayer`。"""

    out: Path
    schema: LiqiSchema
    next_at: tuple[float, float] | None
    settle: float
    probe: bool = False

    done: bool = False
    clicks: int = 0
    records: int | None = None

    _parsers: dict[str, LiqiParser] = field(default_factory=dict, repr=False)
    _ready_at: float | None = field(default=None, repr=False)
    _shots: int = field(default=0, repr=False)
    _last_hash: str = field(default="", repr=False)
    _same: int = field(default=0, repr=False)

    # ------------------------------------------------------------ 封包

    def observe(self, payload: bytes, *, from_client: bool, flow: str) -> None:
        parser = self._parsers.setdefault(flow, LiqiParser(self.schema))
        try:
            message = parser.parse(payload, from_client=from_client)
        except LiqiParseError:
            return
        if message.method != _FETCH_RECORD or message.kind is not MessageKind.RESPONSE:
            return
        res: Any = message.payload  # 類別是執行期從 liqi.json 產生的
        data = bytes(res.data)
        if not data and res.data_url:
            # 紀錄大的時候伺服器只給網址。客戶端自己也是去這裡下載。
            logger.info("紀錄改走 data_url,下載中:{}", res.data_url)
            with urllib.request.urlopen(res.data_url, timeout=30) as response:
                data = response.read()
        (self.out / "record.bin").write_bytes(data)
        head = res.head
        accounts = [
            {"seat": a.seat, "account_id": a.account_id, "nickname": a.nickname}
            for a in head.accounts
        ]
        (self.out / "head.json").write_text(
            json.dumps({"uuid": head.uuid, "accounts": accounts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.records = len(decode_records(self.schema, data))
        logger.info("收到牌譜 {}:{} 步", head.uuid, self.records)
        # 牌譜開起來還要載入、演開場,先等一下再開始截。
        self._ready_at = time.monotonic() + max(self.settle, 3.0)

    # ------------------------------------------------------------ 畫面

    def act(self, page: Page) -> None:
        now = time.monotonic()
        if self.done:
            return
        if self.probe:
            if self._ready_at is None:
                self._ready_at = now
            if now >= self._ready_at:
                self._shoot(page, f"probe-{self._shots:02d}.png")
                self._ready_at = now + 3.0
                if self._shots >= _PROBE_SHOTS:
                    self.done = True
            return
        if self.records is None or self._ready_at is None or now < self._ready_at:
            return

        # 先截目前這張(按了 clicks 次之後),再按下一步。
        image = self._shoot(page, f"frames/{self.clicks:04d}.png")
        if image is None:
            self._ready_at = now + 1.0
            return
        digest = hashlib.sha1(image).hexdigest()
        self._same = self._same + 1 if digest == self._last_hash else 0
        self._last_hash = digest
        with (self.out / "steps.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"clicks": self.clicks, "frame": f"frames/{self.clicks:04d}.png",
                                 "same": self._same}) + "\n")
        if self._same >= _SAME_LIMIT:
            logger.info("連續 {} 張沒變,當作演完了。共按 {} 次", self._same, self.clicks)
            self.done = True
            return

        size = viewport(page)
        if size is None or self.next_at is None:
            self._ready_at = now + 1.0
            return
        _press(page, int(size[0] * self.next_at[0]), int(size[1] * self.next_at[1]))
        self.clicks += 1
        if self.clicks % 20 == 0:
            logger.info("已按 {} 次(紀錄 {} 步)", self.clicks, self.records)
        self._ready_at = time.monotonic() + self.settle

    def _shoot(self, page: Page, name: str) -> bytes | None:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image: bytes = page.screenshot(path=str(path))
        except Exception as exc:  # noqa: BLE001 - 存不下來不值得中斷
            logger.warning("存不下畫面:{}", exc)
            return None
        self._shots += 1
        return image


def _point(text: str) -> tuple[float, float]:
    x, y = (float(v) for v in text.split(","))
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError("座標是畫布的比例,0 到 1")
    return x, y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url", help="牌譜連結")
    parser.add_argument("--out", type=Path, help="輸出目錄,預設 data/recordings/paipu-<時間>")
    parser.add_argument("--next", type=_point, metavar="X,Y", help="「下一步」按鈕,畫布比例")
    parser.add_argument("--settle", type=float, default=1.5, help="按完等幾秒再截(預設 1.5)")
    parser.add_argument("--probe", action="store_true", help="只每 3 秒截一張,用來找按鈕")
    parser.add_argument("--canvas", default=DEFAULT_CANVAS)
    parser.add_argument("--user-data-dir", default=str(DEFAULT_PROFILE))
    args = parser.parse_args(argv)

    if not args.probe and args.next is None:
        parser.error("沒有 --next 就不知道要按哪裡;先用 --probe 看介面")
    setup_logging()

    from mia.groundtruth.cdp import CdpCapture, PlaywrightMissingError

    out: Path = args.out or DATA_DIR / "recordings" / time.strftime("paipu-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(
        json.dumps({"url": args.url, "next": args.next, "settle": args.settle,
                    "canvas": args.canvas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stepper = Stepper(out=out, schema=LiqiSchema.load(), next_at=args.next,
                      settle=args.settle, probe=args.probe)
    print(f"輸出目錄:{out}")
    print("請在開起來的瀏覽器裡登入(登入過就不用),牌譜會自己開。Ctrl-C 結束。")
    try:
        with DumpWriter(out / "ws.jsonl") as writer:
            CdpCapture(writer).run_launched(
                url=args.url,
                user_data_dir=args.user_data_dir,
                canvas=Canvas.parse(args.canvas),
                autoplay=stepper,
                stop_check=lambda: stepper.done,
            )
    except PlaywrightMissingError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    if stepper.records is None and not args.probe:
        print("沒收到 fetchGameRecord —— 牌譜沒開起來?", file=sys.stderr)
        return 1
    print(f"按了 {stepper.clicks} 次,紀錄 {stepper.records} 步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


