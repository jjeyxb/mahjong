#!/usr/bin/env python3
"""遮擋驗證工具 —— 擷取層最後一個沒驗過的風險。

`mia.capture.windows` 的 docstring 原本列了三個風險,前兩個在 2026-09-18 收掉,
第三個「遊戲視窗被遮擋時能不能抓到完整內容」一直掛著,理由寫的是「自動化 session
搶不到前景視窗」。**那個理由是錯的** —— 要製造遮擋不需要把別人搶到前景,
只要自己開一個 ``WS_EX_TOPMOST`` 的視窗蓋上去就好,這條路不受 Windows
foreground lock 管。這支工具就是那件事。

為什麼一定要驗
--------------
`capture/base.py` 開頭那段設計說明的整個前提是「以視窗為擷取單位,上層蓋了什麼
都與擷取結果無關」。HUD 能疊在遊戲上、使用者能切到別的視窗去看攻略,全都靠
這個前提。但 Chrome 在 Windows 上有 **native window occlusion tracking**:
視窗被完全蓋住時它可以把該視窗當成隱藏,停掉合成與 ``requestAnimationFrame``。
若這件事發生,PrintWindow 不會報錯、也不會給全黑 —— 它會給一張**看起來完全
正常但是舊的**畫面。那是最難查的失敗:辨識結果合法、只是慢了幾秒,
而牌局狀態機會照著過期的手牌算下去。

所以這支工具量的不是「抓不抓得到」,是「抓到的是不是活的」。

怎麼判定
--------
三個階段各抓 N 幀,比相鄰幀的差異:

* **A 沒有遮擋** —— 建立基準線。基準線本身不會動的話(遊戲停在選單),
  這次測試對「凍結」就沒有鑑別力,工具會直說,不會給一個假的通過。
* **B 被蓋住** —— 同時用 mss 抓**螢幕**同一塊區域,確認上面真的是遮擋視窗
  的顏色。這一步是測試有效性的證明:螢幕上是純色、PrintWindow 卻拿到動態
  畫面,才叫做「遮擋時仍抓得到」。少了它,「通過」有可能只是遮擋根本沒成立。
* **C 移除遮擋** —— 確認 B 的結果不是視窗剛好在那段時間自己停了。

用法::

    # 全自動:自己開一個放動畫的 Chromium 當靶,不需要人在旁邊
    python tools/occlusion_probe.py --self-test

    # 對真正的雀魂視窗測(先用 capture_probe.py --list 拿 handle)
    python tools/occlusion_probe.py --window "#1380412"
    python tools/occlusion_probe.py --window 雀魂 --partial
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# 讓本檔可直接執行,不需要先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mia.capture import CaptureBackend, CaptureError, WindowInfo, create_backend
from mia.config.loader import load_config
from mia.utils.geometry import Rect
from mia.utils.logging import setup_logging

#: 遮擋視窗的顏色。刻意挑一個遊戲畫面不可能大面積出現的洋紅,
#: 螢幕上驗色時才不會把遊戲自己的像素誤認成遮擋。
OCCLUDER_RGB = (255, 0, 255)

#: 螢幕上要有多少比例是遮擋色,才算「遮擋真的成立」。
#: 留 10% 餘裕給滑鼠游標、輸入法候選字窗這類會跑到上面的小東西。
COVERAGE_REQUIRED = 0.9

#: Playwright 預設會替 Chromium 加上這三個旗標。
#:
#: **自我測試刻意把它們拿掉。** 理由是 MIA 真正依賴遮擋安全的場合是**實戰模式**,
#: 而那個模式抓的是使用者自己開的瀏覽器 —— 沒有人會替它加旗標。留著 Playwright
#: 的保護去測,測到的是比實際情況寬鬆的環境,通過了也不能說明實戰沒事。
#: (只有 groundtruth/cdp.py 那條錄製用的路徑是 Playwright 自己開的瀏覽器,
#: 那條路徑反而是這三個旗標保護得到的。)
PROTECTIVE_FLAGS = (
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
)

#: 遮擋視窗的子行程原始碼。用另一個行程而不是同行程開視窗,是因為 Win32 視窗
#: 必須在建立它的執行緒上跑訊息迴圈,擠在擷取迴圈旁邊只會自找麻煩;而且
#: 「另一個程式蓋上來」也更貼近真實情境。
_OCCLUDER_SOURCE = '''
import ctypes, sys, tkinter as tk

# 一定要先宣告 DPI aware,否則 geometry 給的數字會被系統虛擬化縮放,
# 在 125% 螢幕上視窗會小一圈、蓋不滿目標。
ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))

x, y, w, h, colour, ttl = sys.argv[1:7]
root = tk.Tk()
root.overrideredirect(True)           # 沒有標題列,整塊都是純色
root.geometry(f"{w}x{h}+{x}+{y}")
root.configure(bg=colour)
root.attributes("-topmost", True)
root.after(int(ttl) * 1000, root.destroy)   # 保險:父行程掛掉也不會留下孤兒視窗
root.mainloop()
'''


# --------------------------------------------------------------------- 量測


@dataclass(frozen=True, slots=True)
class Sample:
    """一個階段的量測結果。"""

    brightness: float
    """內容區的平均亮度。接近 0 就是抓到全黑。"""

    motion: float
    """相鄰幀的平均絕對差。0 = 每一幀都一樣,也就是畫面凍結。"""

    distinct: int
    """不重複的幀數。``1`` 代表 N 幀完全相同。"""

    frames: int

    def describe(self) -> str:
        return (
            f"亮度={self.brightness:6.1f}  相鄰幀差={self.motion:6.3f}  "
            f"不同幀={self.distinct}/{self.frames}"
        )

    @property
    def is_black(self) -> bool:
        return self.brightness < 5.0

    @property
    def is_frozen(self) -> bool:
        return self.distinct <= 1


def _content(image: np.ndarray) -> np.ndarray:
    """裁掉瀏覽器視窗裝飾,只留頁面內容。

    不裁的話會有個很陰險的假陽性:分頁標題、網址列的游標、下載列都會自己動,
    於是「相鄰幀有差」明明來自視窗裝飾,卻被當成「頁面還活著」。
    這裡只要一個保守的下界,不需要精準 —— 校正層才負責精準。
    """
    height = image.shape[0]
    return image[int(height * 0.2) :, :]


def _sample(backend: CaptureBackend, window: WindowInfo, count: int, gap: float) -> Sample:
    images = []
    for _ in range(count):
        images.append(_content(backend.capture(window).image))
        time.sleep(gap)

    diffs = [
        float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        for a, b in itertools.pairwise(images)
    ]
    hashes = {hashlib.blake2b(i.tobytes(), digest_size=8).digest() for i in images}
    return Sample(
        brightness=float(np.mean(images[0])),
        motion=sum(diffs) / len(diffs) if diffs else 0.0,
        distinct=len(hashes),
        frames=len(images),
    )


# --------------------------------------------------------------------- 判定


@dataclass(frozen=True, slots=True)
class Verdict:
    """判定結果。

    與量測分開是刻意的:判定規則有分支、有「沒有結論」這種第三種答案,
    那是唯一值得寫測試的部分,而它不該需要一台 Windows 機器與一個瀏覽器
    才跑得起來。
    """

    code: int
    """行程結束碼:0 通過、1 失敗、2 這次測試不算數。"""

    headline: str
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.code == 0


def judge(
    *, baseline: Sample, occluded: Sample, restored: Sample, coverage: float
) -> Verdict:
    """把三個階段的量測換成一句話。

    順序有講究:**先問這次測試算不算數,再問結果。** 遮擋沒成立的時候,
    「沒有變黑、沒有凍結」這兩個觀察都是真的,但它們什麼也沒證明 ——
    先報通過再補一句「不過遮擋可能沒成立」,讀的人只會記得通過。
    """
    if coverage < COVERAGE_REQUIRED:
        return Verdict(
            2,
            f"⚠ 這次測試無效:螢幕上只有 {coverage:.1%} 是遮擋視窗的顏色。",
            (
                "遮擋沒有真的成立(可能有別的 topmost 視窗搶在上面,",
                "或目標視窗在測試中被移動了)。",
                "B 階段的結果不能當成「遮擋時沒問題」的證據。",
            ),
        )

    if occluded.is_black:
        return Verdict(
            1,
            "✗ 失敗:遮擋時抓到全黑。",
            (
                "PrintWindow 這條路對這個視窗行不通,要改用 dxcam 或",
                "Windows Graphics Capture,見 capture/windows.py 的「備案」。",
            ),
        )

    if not baseline.is_frozen and occluded.is_frozen:
        return Verdict(
            1,
            "✗ 失敗:遮擋時畫面凍結 —— 抓到的是一張過期的畫面。",
            (
                "這是 Chrome 的 native window occlusion tracking:視窗被完全蓋住時",
                "把它當成隱藏,停掉合成。啟動瀏覽器時加這兩個旗標可以擋下來:",
                "  --disable-backgrounding-occluded-windows",
                "  --disable-features=CalculateNativeWindowOcclusion",
                "但實戰模式抓的是使用者自己開的瀏覽器,加不了旗標 —— 那種情況",
                "只能改用 Windows Graphics Capture,或接受 HUD 不能完全蓋住畫布。",
            ),
        )

    if baseline.is_frozen:
        return Verdict(
            2,
            "⚠ 沒有結論:基準線(A 階段)本身就是靜止的畫面。",
            (
                "這次只能說「遮擋時沒有變全黑」,沒辦法說「沒有凍結」——",
                "靜止的畫面凍不凍結看起來一模一樣。",
                "請讓遊戲停在有動畫的畫面(對局中、或牌桌待機的立繪)再測一次。",
            ),
        )

    if restored.is_frozen:
        return Verdict(
            2,
            "⚠ 沒有結論:遮擋移除之後(C 階段)畫面反而靜止了。",
            (
                "目標在測試途中自己停了動畫(換場景、進了結算畫面、或是關掉了),",
                "B 階段測到「還活著」有可能只是時間點剛好。請重測一次。",
            ),
        )

    return Verdict(
        0,
        "✓ 通過:視窗被遮擋時,PrintWindow 仍抓到活著的完整內容。",
        (
            f"遮擋時的動態量 {occluded.motion:.3f} 對比基準線 {baseline.motion:.3f}"
            f"(移除遮擋後 {restored.motion:.3f})",
            f"螢幕實測覆蓋率 {coverage:.1%} —— 使用者看到的是純色,擷取拿到的是遊戲畫面。",
        ),
    )


# --------------------------------------------------------------------- 遮擋


def _occluder_coverage(rect: Rect) -> float:
    """螢幕上 ``rect`` 這塊區域有多少比例是遮擋視窗的顏色。

    這是判斷「遮擋到底有沒有成立」的唯一可靠依據。問 Z-order 或
    ``GetForegroundWindow`` 都只說得出意圖,說不出使用者實際看到什麼。
    """
    import mss

    red, green, blue = OCCLUDER_RGB
    with mss.MSS() as sct:  # mss.mss() 是已棄用的舊別名,與 mss_fallback.py 對齊
        shot = sct.grab(
            {"left": rect.x, "top": rect.y, "width": rect.width, "height": rect.height}
        )
        pixels = np.asarray(shot)[:, :, :3]  # mss 給 BGRA
    match = (pixels[:, :, 0] == blue) & (pixels[:, :, 1] == green) & (pixels[:, :, 2] == red)
    return float(np.mean(match))


class Occluder:
    """在螢幕上蓋一塊純色的 topmost 視窗。"""

    def __init__(self, rect: Rect, *, ttl: int = 120) -> None:
        self._rect = rect
        self._ttl = ttl
        self._process: subprocess.Popen[bytes] | None = None
        self._script: Path | None = None

    def __enter__(self) -> Occluder:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix="_occluder.py", delete=False, encoding="utf-8"
        )
        with handle:
            handle.write(_OCCLUDER_SOURCE)
        self._script = Path(handle.name)
        colour = "#{:02x}{:02x}{:02x}".format(*OCCLUDER_RGB)
        self._process = subprocess.Popen(
            [
                sys.executable,
                str(self._script),
                str(self._rect.x),
                str(self._rect.y),
                str(self._rect.width),
                str(self._rect.height),
                colour,
                str(self._ttl),
            ]
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - 保險
                self._process.kill()
        if self._script is not None:
            self._script.unlink(missing_ok=True)

    def wait_until_visible(self, *, timeout: float = 10.0) -> float:
        """等到螢幕上真的看得到它,回傳覆蓋率。"""
        deadline = time.monotonic() + timeout
        coverage = 0.0
        while time.monotonic() < deadline:
            coverage = _occluder_coverage(self._rect)
            if coverage > COVERAGE_REQUIRED:
                return coverage
            time.sleep(0.3)
        return coverage


def _physical_bounds(window: WindowInfo, backend: CaptureBackend) -> Rect:
    """視窗的實體像素邊界 —— 遮擋視窗與 mss 都吃這個單位。

    ``WindowInfo.bounds`` 對外一律是邏輯座標(見 capture/windows.py 的長註解),
    拿它去擺遮擋視窗,在 125% 螢幕上會少蓋掉 20% 的面積。
    """
    frame = backend.capture(window)
    return Rect(
        round(window.bounds.x * frame.scale),
        round(window.bounds.y * frame.scale),
        frame.image.shape[1],
        frame.image.shape[0],
    )


def probe(
    backend: CaptureBackend,
    window: WindowInfo,
    *,
    count: int,
    gap: float,
    partial: bool,
    witness: Callable[[], str] | None = None,
) -> int:
    """跑完三個階段並印出判定。

    Args:
        witness: 可選的旁證來源 —— 回傳「目標自己認為的可見狀態」。
            自我測試會傳 ``document.visibilityState``,用來分辨兩種很不一樣的
            通過:Chrome 根本沒發現被蓋住(那只證明沒踩到),還是 Chrome 知道
            被蓋住了但擷取照樣拿到活的內容(那才是真的證明了擷取獨立於畫面)。
    """
    physical = _physical_bounds(window, backend)
    cover = physical
    if partial:
        # 只蓋右半邊 —— 部分遮擋是比較常見的情境(旁邊開個瀏覽器查番種),
        # 而且 Chrome 的 occlusion tracking 只在**完全**被蓋住時才觸發,
        # 兩種都測才問得出「哪一種會壞」。
        cover = Rect(
            physical.x + physical.width // 2,
            physical.y,
            physical.width - physical.width // 2,
            physical.height,
        )

    def observe(label: str) -> None:
        if witness is not None:
            print(f"  目標自述: {label} → {witness()}")

    print(f"目標視窗: {window}")
    print(f"實體邊界: {physical}")
    print(f"遮擋範圍: {cover}  ({'右半邊' if partial else '全部'})")
    print(f"每階段  : {count} 幀,間隔 {gap:.2f}s\n")

    print("A 沒有遮擋")
    baseline = _sample(backend, window, count, gap)
    print(f"  {baseline.describe()}")
    observe("A")

    with Occluder(cover) as occluder:
        coverage = occluder.wait_until_visible()
        # 給 Chrome 的 occlusion tracking 反應時間。它不是同步的 ——
        # 太快取樣會在它還沒把視窗標成隱藏之前就量完,量到一個假的通過。
        time.sleep(3.0)
        print(f"\nB 被蓋住(螢幕實測覆蓋率 {coverage:.1%})")
        occluded = _sample(backend, window, count, gap)
        print(f"  {occluded.describe()}")
        observe("B")

    time.sleep(2.0)
    print("\nC 遮擋移除")
    restored = _sample(backend, window, count, gap)
    print(f"  {restored.describe()}")
    observe("C")

    verdict = judge(
        baseline=baseline, occluded=occluded, restored=restored, coverage=coverage
    )
    print("\n" + "=" * 64)
    print(verdict.headline)
    for note in verdict.notes:
        print(f"  {note}")
    return verdict.code


# --------------------------------------------------------------------- 自我測試

#: 自我測試用的靶。每一幀畫一整張隨機雜訊 —— 只要合成停了,
#: 抓回來的畫面就會一模一樣,這是最靈敏的「凍結」指示器。
#:
#: 刻意**不用** setInterval 去改 document.title:標題一變,分頁列就會重繪,
#: 於是「相鄰幀有差」會來自視窗裝飾而不是頁面內容。第一版就是這樣量的,
#: 量到的通過是假的。
_TARGET_PAGE = """<body style='margin:0;background:#101014'>
<canvas id='c' width='640' height='360' style='width:100%'></canvas>
<script>
const ctx = document.getElementById('c').getContext('2d');
const buf = ctx.createImageData(640, 360);
function tick() {
  for (let i = 0; i < buf.data.length; i += 4) {
    const v = Math.random() * 255 | 0;
    buf.data[i] = buf.data[i + 1] = buf.data[i + 2] = v;
    buf.data[i + 3] = 255;
  }
  ctx.putImageData(buf, 0, 0);
  requestAnimationFrame(tick);
}
tick();
</script></body>"""


def self_test(backend: CaptureBackend, *, count: int, gap: float, partial: bool) -> int:
    """自己開一個放動畫的 Chromium 當靶,全程不需要人操作。"""
    from playwright.sync_api import sync_playwright

    before = {w.handle for w in backend.list_windows()}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False, ignore_default_args=list(PROTECTIVE_FLAGS)
        )
        try:
            # no_viewport=True 的理由與 cdp.py 相同:預設會把 viewport 鎖在
            # 1280x720 而不管視窗多大,量到的東西就不是真實情境。
            page = browser.new_context(no_viewport=True).new_page()
            page.set_content(_TARGET_PAGE)
            time.sleep(2.0)

            new = [w for w in backend.list_windows() if w.handle not in before]
            if not new:
                print("找不到剛開的瀏覽器視窗", file=sys.stderr)
                return 1
            # 用 handle 差集認視窗,不用標題比對 —— 標題比對會抓到使用者自己
            # 開著的 Chrome(之前就踩過,量了半天量的是另一台螢幕上的視窗)。
            target = max(new, key=lambda w: w.bounds.area)
            return probe(
                backend,
                target,
                count=count,
                gap=gap,
                partial=partial,
                witness=lambda: str(page.evaluate("document.visibilityState")),
            )
        finally:
            browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="視窗被遮擋時的擷取驗證",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--self-test",
        action="store_true",
        help="自己開一個放動畫的 Chromium 當靶,全自動",
    )
    source.add_argument(
        "--window",
        metavar="SPEC",
        help="目標視窗:標題/程式名子字串,或 #視窗ID",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="只蓋右半邊(Chrome 的 occlusion tracking 只在完全被蓋住時觸發)",
    )
    parser.add_argument("--frames", type=int, default=10, help="每階段抓幾幀(預設 10)")
    parser.add_argument("--gap", type=float, default=0.12, help="幀間隔秒數(預設 0.12)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    if sys.platform != "win32":
        print("這支工具只在 Windows 上有意義", file=sys.stderr)
        return 1

    config = load_config()
    try:
        with create_backend(config.capture) as backend:
            if args.self_test:
                return self_test(
                    backend, count=args.frames, gap=args.gap, partial=args.partial
                )

            from capture_probe import _resolve_window

            window = _resolve_window(backend, args.window, config)
            return probe(
                backend, window, count=args.frames, gap=args.gap, partial=args.partial
            )
    except CaptureError as exc:
        print(f"\n擷取失敗:\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
