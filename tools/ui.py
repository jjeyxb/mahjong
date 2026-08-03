#!/usr/bin/env python
"""開 MIA 的側邊視窗。

四種餵資料的方式:

``--live``
    **真的接上遊戲。** 按下視窗左下角的「開始遊戲」會開一個受控的瀏覽器並開始
    抓封包,同時可以擷取畫面跑 CV,兩條路的結果一起顯示。這是實際使用的模式,
    其餘三種都是為了驗它的某一段。

``--replay <ws.jsonl>``
    把錄下的對局照時間重播 —— 引擎真的在跑,建議是真的算出來的。**不需要開遊戲**,
    所以 UI 隨時可以驗、可以截圖給論文用。

``--session <錄影目錄>``
    把錄下的畫面跑一次 CV,顯示辨識出來的手牌與向聽。驗的是功能 1 那條路。

``--demo``
    塞一組寫死的假資料就停住。純粹用來看版面,不需要任何素材。這是預設。

四種模式都有 **Overlay**(疊在遊戲上的精簡 HUD),開關在側邊視窗的設定頁 ——
它與側邊視窗訂閱同一份狀態,所以重播與示範模式也看得到,截圖不必開遊戲。

用法::

    # 實際使用:按「開始遊戲」開瀏覽器,登入後撥開關就開始給建議
    python tools/ui.py --live --mortal models/mortal_298k.pth

    # 已經有另一個 gt.py 在錄了,只要跟著那個檔案走
    python tools/ui.py --live --tail data/recordings/now/ws.jsonl

    python tools/ui.py --demo
    python tools/ui.py --replay tests/fixtures/real_game_full.jsonl --speed 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from mia.calibration.canvas import Canvas, CanvasChoice
from mia.engine import AIEngine, DummyEngine, EngineGroup
from mia.live.runtime import LiveRuntime
from mia.mjai import MjaiEvent
from mia.mjai.handstate import HandTracker
from mia.ui.overlay.window import OverlayWindow
from mia.ui.panel.window import PanelWindow, present
from mia.ui.state import UiState
from mia.ui.viewmodel import ViewModel
from mia.utils.logging import setup_logging

DEMO_HAND = ("1m", "1m", "2m", "3s", "4m", "5pr", "6m", "8p", "8s", "8s", "9m", "E", "P")

#: 即時模式下多久把郵箱套進 UI 一次(毫秒)。
#:
#: 100 ms 是刻意的:擷取是 15 fps(66 ms),再快只是重畫同樣的東西;而人看
#: 手牌變化,10 Hz 已經完全跟得上。郵箱會把這段時間內的更新合併掉,所以
#: 「調慢」不會漏資訊,只會讓畫面晚 0.1 秒 —— 遠小於遊戲自己的動畫時間。
PUMP_INTERVAL_MS = 100


def build_engines(args: argparse.Namespace) -> list[AIEngine]:
    """建引擎。**模型排在規則式 baseline 之前。**

    順序就是 ``ViewState.primary`` 的優先序。baseline 排前面的話,headline 會
    顯示 baseline 的建議、真正的模型被擠到下面那排,而 Q 值長條會整段消失
    (baseline 沒有 meta)—— 那正是實際跑起來看到的症狀。
    """
    engines: list[AIEngine] = []
    for weights in args.mortal or []:
        from mia.engine.mortal import mortal_engine

        engines.append(mortal_engine(weights, seat=args.seat, name=Path(weights).stem))
    engines.append(DummyEngine())
    return engines


def load_events(path: Path) -> list[MjaiEvent]:
    """讀事件流。與 ``tools/advise.py`` 同一套,兩種格式都吃。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from advise import load_events as _load

    events, _ = _load(path)
    return events


def run_replay(args: argparse.Namespace, model: ViewModel) -> None:
    """照事件順序重播,每 ``interval`` 毫秒一步。

    用 QTimer 而不是背景執行緒:引擎的 react 是同步的、單次 10~20 ms,直接在
    主執行緒跑不會讓 UI 卡住,而且完全避開「Qt widget 只能在主執行緒動」
    這個限制。真正接上遊戲時才需要執行緒,那是另一件事。
    """
    events = load_events(args.replay)
    if not events:
        raise SystemExit(f"{args.replay} 裡沒有事件")

    seat = next((int(getattr(e, "id", 0)) for e in events if e.TYPE == "start_game"), 0)
    args.seat = seat
    tracker = HandTracker(_name="ui")
    group = EngineGroup(build_engines(args))
    group.start()
    model.set_notices([f"重播 {args.replay.name} — 自己坐 {seat}"])

    step = iter(range(len(events)))

    def advance() -> None:
        try:
            index = next(step)
        except StopIteration:
            model.set_notices([f"重播結束({len(events)} 個事件)"])
            group.close()
            timer.stop()
            return

        event = events[index]
        tracker.feed(event)
        model.update_packet_hand(tracker.tiles, drawn=tracker.drawn)
        result = group.react(event)
        # 只在有人真的要動作時換掉建議 —— 每一步都覆蓋會讓畫面在「切 3s」與
        # 「不需要動作」之間閃,而後者佔了九成的事件
        if result.actions:
            model.update_advices(result.advices)

    timer = QTimer()
    timer.timeout.connect(advance)
    timer.start(max(1, round(1000 / args.speed)))
    # 掛在 model 上避免被 GC —— QTimer 沒有 parent 時會被當成暫時物件回收
    model._timer = timer  # type: ignore[attr-defined]  # noqa: SLF001


class _RememberedCanvas:
    """把畫布尺寸的選擇轉給 runtime,順手記到磁碟。

    這一層存在的理由只有「記住」::class:`LiveRuntime` 不該知道
    ``data/ui_state.json`` 這種東西存不存在,而 :class:`PanelWindow` 也不該。
    持久化的決定屬於接線的這一層。
    """

    def __init__(self, runtime: LiveRuntime, state: UiState) -> None:
        self._runtime = runtime
        self._state = state

    def can_pick_canvas(self) -> bool:
        return self._runtime.can_pick_canvas()

    def canvas(self) -> str | None:
        return self._runtime.canvas()

    def set_canvas(self, key: str | None) -> None:
        self._runtime.set_canvas(key)
        self._state.canvas = key
        self._state.save()


def build_live_runtime(
    args: argparse.Namespace, model: ViewModel, *, canvas: CanvasChoice | None = None
) -> LiveRuntime:
    """組出 runtime。**什麼都還沒開始** —— 瀏覽器等使用者按「開始遊戲」,
    兩個功能等使用者撥開關。

    工廠是延遲呼叫的:``PacketWorker`` 一建立就會去組引擎,而 Mortal 要載
    130MB 權重。使用者沒打開 AI 建議的話,那 10 秒完全不該花。

    兩條路各自獨立:沒有 ``--mortal`` 也可以只用畫面辨識,擷取子程序起不來
    也不影響 CV。這正是三個功能刻意解耦的地方。
    """
    from mia import features
    from mia.config.loader import load_config
    from mia.live import PacketWorker, UpdateBus, VisionWorker, capture_command
    from mia.live.runtime import Feature, Worker
    from mia.live.source import CaptureLauncher
    from mia.utils.paths import DATA_DIR
    from mia.vision.tiles.classify import DEFAULT_SKIN

    bus = UpdateBus()
    feature_list: list[Feature] = []
    capture: CaptureLauncher | None = None

    if not args.no_packets:
        if args.tail:
            # 別人已經在錄了,只跟著走 —— 不要再開一個擷取子程序去搶同一個瀏覽器
            tail: Path = args.tail

            def dump_path() -> Path:
                return tail
        else:
            capture = CaptureLauncher(
                lambda dump: capture_command(
                    dump,
                    mode=args.capture_mode,
                    url=args.url,
                    user_data_dir=args.user_data_dir,
                    connect=args.connect,
                    # 每次開瀏覽器才讀:使用者可能在上一場結束後才改選單
                    canvas=canvas.key if canvas else None,
                ),
                root=DATA_DIR / "live",
            )
            # 路徑是 launcher 懶決定的:每按一次「開始遊戲」換一個新檔案,所以
            # 這裡不能先取出來存著 —— 那樣第二場會繼續讀第一場的檔案。
            launcher = capture

            def dump_path() -> Path:
                return launcher.dump

        def make_packets() -> Worker:
            # from_start=True:座位、寶牌、誰立直了全在先前的事件裡,所以
            # 中途才打開開關也要從頭讀一遍把局面追上來
            return PacketWorker(bus, dump=dump_path(), engines=build_engines(args), from_start=True)

        feature_list.append(Feature(features.ADVICE, make_packets))

    if not args.no_vision:
        config = load_config()
        skin = args.skin or DEFAULT_SKIN

        def make_vision() -> Worker:
            return VisionWorker(bus, config=config, skin=skin, canvas=canvas)

        feature_list.append(Feature(features.VISION, make_vision))

    if not feature_list:
        raise SystemExit("--no-vision 與 --no-packets 同時給了,那就沒有東西可以顯示")

    return LiveRuntime(model, bus, features=feature_list, capture=capture, canvas=canvas)


def start_live(runtime: LiveRuntime, model: ViewModel) -> None:
    """讓 pump 開始跑。瀏覽器與兩個功能都還沒啟動。"""
    runtime.start()
    timer = QTimer()
    timer.timeout.connect(runtime.pump)
    timer.start(PUMP_INTERVAL_MS)
    # 掛在 model 上避免被 GC —— QTimer 沒有 parent 時會被當成暫時物件回收
    model._timer = timer  # type: ignore[attr-defined]  # noqa: SLF001


def run_session(args: argparse.Namespace, model: ViewModel) -> None:
    """把錄下的畫面跑一次 CV。驗的是功能 1。"""
    from mia.calibration.table import Calibration
    from mia.config.loader import load_config
    from mia.recorder import SessionReader
    from mia.utils.geometry import Size
    from mia.vision.roi import RoiSet
    from mia.vision.tiles.classify import TemplateSet, classify_hand
    from mia.vision.tiles.hand import read_hand

    session = SessionReader(args.session)
    if session.table_rect is None:
        raise SystemExit("session 沒有 table_rect —— 錄製時校正沒鎖定")

    records = [r for r in session.manifest.frames if r.changed]
    if not records:
        raise SystemExit("session 裡沒有有變化的幀")

    first = session.load_image(records[0])
    height, width = first.shape[:2]
    config = load_config()
    rois = RoiSet(
        config.roi,
        Calibration(session.table_rect, Size(width, height), source="manual"),
    )
    box = rois["own_hand"].rect
    templates = TemplateSet.load()
    model.set_notices([f"重播畫面 {args.session.name} — {len(records)} 幀"])

    step = iter(range(len(records)))

    def advance() -> None:
        try:
            index = next(step)
        except StopIteration:
            model.set_notices([f"重播結束({len(records)} 幀)"])
            timer.stop()
            return

        roi = session.load_image(records[index])[box.as_slice()]
        hand = read_hand(roi)
        concealed, drawn = classify_hand(roi, hand, templates)
        model.update_cv_hand(
            [m.label for m in concealed],
            drawn=drawn.label if drawn else None,
            confident=all(m.is_confident for m in (*concealed, *(x for x in [drawn] if x))),
        )

    timer = QTimer()
    timer.timeout.connect(advance)
    timer.start(max(1, round(1000 / args.speed)))
    model._timer = timer  # type: ignore[attr-defined]  # noqa: SLF001


def run_demo(model: ViewModel) -> None:
    """一組寫死的資料。只為了看版面,不代表任何真實局面。"""
    from mia.engine.base import Advice
    from mia.mjai import Dahai

    model.update_packet_hand((*DEMO_HAND, "F"), drawn="F")
    # mask_bits 的位元位置要真的對應那幾張牌,q_values 也要照索引升冪 ——
    # 隨便填的話畫面上會出現「建議切 E」但候選清單裡根本沒有 E,那種示範
    # 比沒有示範更糟。9m=8、8p=16、3s=20、E=27,最高分落在 E。
    model.update_advices(
        [
            Advice(
                "mortal_298k",
                Dahai(actor=0, pai="E", tsumogiri=False),
                {
                    "mask_bits": (1 << 8) | (1 << 16) | (1 << 20) | (1 << 27),
                    "q_values": [0.94, 0.31, -0.55, 1.28],
                },
                14.0,
            ),
            Advice("baseline", Dahai(actor=0, pai="9m", tsumogiri=False), None, 0.4),
        ]
    )
    model.set_notices(["示範資料 —— 不是真實局面"])


def _quit_on_signals(app: QApplication) -> None:
    """讓 Ctrl-C 與 SIGTERM 走正常的關閉流程。

    預設行為下 SIGTERM 直接終止行程,``finally`` 不會跑 —— 引擎子程序會因為
    stdin 收到 EOF 自己結束,但**擷取子程序不會**:那是一個獨立的 Chromium,
    父程序死掉它照樣開著。留一個孤兒瀏覽器在螢幕上是使用者看得到的問題。

    Python 只在位元組碼之間處理訊號,而 ``app.exec()`` 卡在 C 裡面。即時模式
    本來就有一個 100 ms 的 QTimer 在跑,解譯器每一次 tick 都會拿回控制權,
    所以處理函式最慢在 100 ms 內生效。
    """
    import signal

    def quit_app(signum: int, _frame: object) -> None:
        print(f"\n收到訊號 {signum},正在關閉…")
        app.quit()

    signal.signal(signal.SIGINT, quit_app)
    signal.signal(signal.SIGTERM, quit_app)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--live", action="store_true", help="真的接上遊戲(畫面 + 封包)")
    source.add_argument("--replay", type=Path, help="重播封包錄影(引擎真的在跑)")
    source.add_argument("--session", type=Path, help="重播畫面錄影(跑 CV)")
    source.add_argument("--demo", action="store_true", help="塞假資料看版面")
    parser.add_argument("--mortal", action="append", metavar="WEIGHTS", help="加一個 Mortal 引擎")
    parser.add_argument("--speed", type=float, default=4.0, help="重播速度(步/秒),預設 4")
    parser.add_argument("--skin", default=None, help="牌面素材皮膚")
    parser.add_argument("--log-level", default="WARNING")

    live = parser.add_argument_group("即時模式(--live)")
    live.add_argument("--no-vision", action="store_true", help="不跑畫面辨識,只要 AI 建議")
    live.add_argument("--no-packets", action="store_true", help="不接封包,只要畫面辨識")
    live.add_argument(
        "--tail",
        type=Path,
        metavar="WS.JSONL",
        help="跟著別的程序正在寫的錄影檔走,不自己開擷取子程序",
    )
    live.add_argument(
        "--capture-mode",
        default="cdp",
        choices=("cdp", "proxy", "local"),
        help="封包擷取方式,預設 cdp(零前置設定)",
    )
    live.add_argument("--url", help="要開啟的網址,預設網頁版雀魂")
    live.add_argument("--user-data-dir", help="持久化的瀏覽器設定檔目錄,可保留登入狀態")
    live.add_argument("--connect", metavar="ENDPOINT", help="連到已在跑的瀏覽器")
    args = parser.parse_args(argv)
    args.seat = 0

    setup_logging(level=args.log_level)
    app = QApplication(sys.argv[:1])

    model = ViewModel()

    # Overlay 與側邊視窗訂閱**同一個** ViewModel —— 兩種呈現方式,一份狀態。
    # 它自己記得上次是開著還是收著,所以這裡不需要命令列參數。
    #
    # 狀態要在 runtime **之前**讀:上次選的畫布尺寸得跟著進去,不然第一次按
    # 「開始遊戲」會用自動偵測開,使用者要再改一次選單才生效。
    ui_state = UiState.load()
    canvas = CanvasChoice(Canvas.parse(ui_state.canvas))

    # runtime 必須在視窗**之前**建好:視窗上的開關要接到它。反過來的話開關
    # 只能先畫成停用,之後再想辦法補接 —— 而那正是最容易忘記做的一步。
    runtime: LiveRuntime | None = (
        build_live_runtime(args, model, canvas=canvas) if args.live else None
    )

    # options 是**兩個視窗共用**的,所以只放兩邊都收的參數。側邊視窗獨有的
    # 東西一律寫在下面的呼叫裡 —— 塞進 options 的話 Overlay 會收到一個它不
    # 認得的關鍵字,而 `**options` 加上 type: ignore 讓 mypy 看不出來,
    # 要到真的執行才炸(canvas 就這樣炸過一次)。
    options: dict[str, object] = {}
    if args.skin:
        options["skin"] = args.skin
    if runtime is not None:
        options["switchboard"] = runtime
    overlay = OverlayWindow(model, ui_state=ui_state, **options)  # type: ignore[arg-type]
    model.subscribe(overlay.apply)

    # launcher 與 canvas 只給側邊視窗 —— 「開始遊戲」是一個動作而不是要顯示的
    # 狀態,畫布尺寸則是設定;Overlay 上兩者都沒有位置(鎖定之後也按不到)。
    window = PanelWindow(
        model,
        overlay=overlay,
        launcher=runtime,
        canvas=_RememberedCanvas(runtime, ui_state) if runtime is not None else None,
        **options,  # type: ignore[arg-type]
    )
    model.subscribe(window.apply)

    if runtime is not None:
        start_live(runtime, model)
        _quit_on_signals(app)
        print("按視窗左下角的「開始遊戲」開啟瀏覽器;兩個功能的開關預設關著。")
    elif args.replay:
        run_replay(args, model)
    elif args.session:
        run_session(args, model)
    else:
        run_demo(model)

    present(window)
    # Overlay 在側邊視窗**之後**才顯示,不然置頂的側邊視窗剛 raise 上來會蓋在
    # 它前面 —— 使用者勾了卻看不到,只會以為壞了。
    if overlay.shown:
        overlay.set_shown(True)
    try:
        return app.exec()
    finally:
        # 一定要收:引擎子程序與擷取子程序都不是 daemon 的孩子,漏掉會留下
        # 一個載著 130MB 權重的 Python 與一個孤兒 Chromium。
        if runtime is not None:
            runtime.stop()


if __name__ == "__main__":
    sys.exit(main())
