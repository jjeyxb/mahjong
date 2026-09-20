#!/usr/bin/env python3
"""把一整局的錄影幀疊成一張「所有牌出現過的位置」聯集圖,供 ROI 量測用。

為什麼需要這個
--------------
單張截圖只能反映**那一瞬間**的狀態:牌河才打到第幾張、誰還沒副露。拿它量
ROI 一定會低估 —— 外框要留的是**最極端**的情況(牌河疊到第三列、四組副露
含槓)。把整局每一幀疊起來取聯集,就能一次看到所有區域各自能長到多大。

怎麼疊
------
單純逐像素取最大值不行:雀魂會渲染 3D 的手伸進來摸打,那些淺色的手掃過整
張桌子,取 max 會把畫面洗白。所以多一道「持續性」條件 —— 動畫的手在某個位
置只停留幾幀,真的放下去的牌會留到本局結束。只有在夠多幀裡都跟本局開局畫面
不一樣的像素,才採用它的最大值。

這也是為什麼**一個 session 請只錄一局**:參考基準是該 session 的第一張正常
幀(此時牌河還空著)。要合併多局就給多個 session,各自算完再取聯集。

用法::

    .venv/bin/python tools/roi_union.py data/recordings/g2/frames --out data/roi_ref/union.png
    .venv/bin/python tools/roi_union.py data/recordings/*/frames --out data/roi_ref/union.png

輸出的圖可以直接餵給 roi_annotate.py::

    .venv/bin/python tools/roi_annotate.py \
        --image data/roi_ref/union.png --out data/roi_ref/roi.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 讓本檔可直接執行,不需要先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: 疊圖一律先正規化到這個尺寸,不同 session 的視窗大小才能合併。
DEFAULT_SIZE = (2560, 1440)

#: 亮度落在這個區間才算「正常牌桌畫面」。和了/立直動畫/轉場會明顯偏亮或偏暗,
#: 它們蓋住整個畫面,疊進去只會把聯集圖洗掉。實測正常幀的平均亮度極集中
#: (中位數 84,四分位距不到 6),用固定區間就夠,不需要自適應。
DEFAULT_BAND = (76.0, 92.0)

#: 與開局畫面不同的幀數要佔多少比例,才認定「這裡真的放了牌」。
#: 太低會混進摸打動畫的手,太高會漏掉本局後期才打出去的牌。
DEFAULT_PERSIST = 0.15

#: 判定像素「跟開局不一樣」的灰階差異門檻。
DIFF_THRESHOLD = 28


def _load_session(session: Path, size: tuple[int, int]) -> list[np.ndarray]:
    """讀出一個 session 裁切並縮放到統一尺寸後的所有幀。"""
    manifest_path = session / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{session} 裡沒有 manifest.json,這不是錄影 session 目錄")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_rect = manifest.get("table_rect")
    if not table_rect:
        raise SystemExit(
            f"{session} 的 manifest 沒有 table_rect —— 錄製當下校正失敗了,"
            "無法確定遊戲畫布在哪裡"
        )

    x, y, w, h = table_rect
    frames_dir = session / "frames"
    paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"{frames_dir} 裡沒有任何影像")

    out = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        out.append(cv2.resize(image[y : y + h, x : x + w], size, interpolation=cv2.INTER_AREA))
    return out


def _union_one(frames: list[np.ndarray], band: tuple[float, float], persist: float) -> np.ndarray:
    """單一 session(單一局)的聯集圖。"""
    normal = [f for f in frames if band[0] <= cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() <= band[1]]
    if not normal:
        raise SystemExit(f"沒有任何幀的亮度落在 {band} 之內,這個 session 可能整段都是動畫")

    # 基準 = 本局第一張正常幀。此時牌河還空著,之後長出來的東西都會被算成差異。
    ref = cv2.cvtColor(normal[0], cv2.COLOR_BGR2GRAY).astype(np.int16)
    height, width = ref.shape
    count = np.zeros((height, width), np.int32)
    brightest = np.zeros_like(normal[0])

    for frame in normal:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        count += np.abs(gray - ref) > DIFF_THRESHOLD
        brightest = np.maximum(brightest, frame)

    persistent = count >= persist * len(normal)
    result = normal[0].copy()
    result[persistent] = brightest[persistent]
    print(f"  正常幀 {len(normal)}/{len(frames)},持續性像素 {persistent.mean() * 100:.1f}%")
    return result


#: 對位相關係數低於此值就當作失敗。不同局的牌面差很多,整圖比對拿不到高分,
#: 靠的是牌桌框線、牌牆、HUD 外框這些靜態結構,實測正常落在 0.55~0.70。
MIN_ALIGN_SCORE = 0.45


def _align(image: np.ndarray, canvas: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    """把一張單獨的畫布圖擺到正確位置,補滿成 ``size``。

    校正時的邊框剝除有時會多剝掉幾十個像素(replay 播放器右側那些 OFF 標籤
    剛好壓在判斷邊界上,實測會少 10~43 px),這種圖直接縮放會讓所有座標歪掉,
    必須平移貼回原位。位置用整張圖去已有的畫布裡比對求得,不寫死任何 UI 地標。
    """
    width, height = size
    if image.shape[:2] == (height, width):
        return image

    # 比例正確 = 這是一張裁乾淨的畫布,只是視窗大小不同。ROI 是正規化座標,
    # 直接縮放到目標尺寸即可 —— 不能走底下的平移路徑,那是給「比例被剝壞、
    # 少了幾十列像素」的圖用的,對這種圖平移只會貼歪。
    target_aspect = width / height
    aspect = image.shape[1] / image.shape[0]
    if abs(aspect - target_aspect) / target_aspect <= 0.01:
        print(f"    比例 {aspect:.4f} 正確,縮放 "
              f"{image.shape[1]}x{image.shape[0]} -> {width}x{height}")
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if image.shape[0] > height or image.shape[1] > width:
        raise SystemExit(
            f"圖比目標畫布 {width}x{height} 還大"
            f"({image.shape[1]}x{image.shape[0]}),無法對位"
        )

    if canvas is None:
        offset = (0, 0)
        print("    (還沒有可對位的基準,假設左上角對齊)")
    else:
        result = cv2.matchTemplate(canvas, image, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if score < MIN_ALIGN_SCORE:
            raise SystemExit(
                f"對位失敗(相關係數 {score:.3f} < {MIN_ALIGN_SCORE})。"
                "這張圖可能不是同一個解析度的雀魂畫布,或根本不是畫布裁切圖。"
            )
        offset = (int(loc[0]), int(loc[1]))
        print(f"    對位 offset={offset} score={score:.3f}")

    padded = np.zeros((height, width, 3), np.uint8)
    padded[offset[1] : offset[1] + image.shape[0], offset[0] : offset[0] + image.shape[1]] = image
    return padded


#: 疊圖比基準亮多少才算「這裡在別的局面出現過牌」。
OVERLAY_THRESHOLD = 25
OVERLAY_ALPHA = 0.45
OVERLAY_TINT = (80, 255, 80)  # BGR,綠色


def _overlay(merged: np.ndarray, base_path: Path, size: tuple[int, int]) -> np.ndarray:
    """把聯集畫成半透明色塊疊在一張乾淨的圖上。

    直接看 max 疊圖很難判讀 —— 六個局面的牌全部半透明重疊在一起,牌面糊成一片。
    改成「一張看得清楚的底圖 + 綠色標出其他局面多出來的部分」,就能同時看到
    牌桌長什麼樣、以及每個區域最遠能長到哪裡。
    """
    base = cv2.imread(str(base_path))
    if base is None:
        raise SystemExit(f"讀不到基準圖: {base_path}")
    base = _align(base, merged, size)

    extra = (
        cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY).astype(np.int16)
        - cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ) > OVERLAY_THRESHOLD

    out = base.copy()
    tint = np.full_like(base, OVERLAY_TINT, dtype=np.uint8)
    out[extra] = cv2.addWeighted(base, 1 - OVERLAY_ALPHA, tint, OVERLAY_ALPHA, 0)[extra]
    print(f"基準圖 {base_path.name},聯集多出來的區域佔 {extra.mean() * 100:.1f}%(標綠色)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把錄影幀疊成 ROI 量測用的聯集圖",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sessions", nargs="*", type=Path,
                        help="錄影 session 目錄(內含 manifest.json 與 frames/)")
    parser.add_argument("--image", type=Path, action="append", default=[], metavar="PATH",
                        help="額外併入的單張畫布圖(可重複)。"
                             "沒有錄影、只有截圖的極端局面就用這個補")
    parser.add_argument("--out", type=Path, required=True, help="輸出 PNG 路徑")
    parser.add_argument("--size", default="2560x1440",
                        help=f"統一縮放尺寸,預設 {DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}")
    parser.add_argument("--persist", type=float, default=DEFAULT_PERSIST,
                        help=f"持續性門檻,預設 {DEFAULT_PERSIST}。"
                             "調低會混進摸打動畫的手,調高會漏掉本局後期打出的牌")
    parser.add_argument("--band", default="76,92",
                        help="正常牌桌畫面的平均亮度區間,用來濾掉和了/立直動畫")
    parser.add_argument("--base", type=Path, metavar="PATH",
                        help="改成把聯集畫成半透明色塊疊在這張乾淨的圖上。"
                             "純疊圖會糊成一片很難判讀,要人眼量測時建議用這個")
    parser.add_argument("--dump-aligned", type=Path, metavar="DIR",
                        help="順便把每個來源對位後的單張圖分開存出來。"
                             "roi_annotate.py 要求所有參考圖同尺寸,用這個產生")
    args = parser.parse_args(argv)

    width, _, height = args.size.partition("x")
    size = (int(width), int(height))
    low, _, high = args.band.partition(",")
    band = (float(low), float(high))

    if not args.sessions and not args.image:
        parser.error("至少要給一個 session 目錄或一張 --image")

    dump = args.dump_aligned
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    merged: np.ndarray | None = None
    for session in args.sessions:
        print(f"{session}")
        one = _union_one(_load_session(session, size), band, args.persist)
        if dump:
            cv2.imwrite(str(dump / f"{session.parent.name}.png"), one)
        merged = one if merged is None else np.maximum(merged, one)

    # 單張圖放後面處理:先有 session 疊出來的完整畫布,才有東西可以拿來對位
    for path in args.image:
        print(f"{path}")
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"讀不到影像: {path}")
        aligned = _align(image, merged, size)
        if dump:
            cv2.imwrite(str(dump / path.name), aligned)
        merged = aligned if merged is None else np.maximum(merged, aligned)

    assert merged is not None
    if dump:
        print(f"對位後的單張圖已存到 {dump}/")

    if args.base:
        merged = _overlay(merged, args.base, size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), merged)
    print(f"\n已輸出 {args.out}  ({size[0]}x{size[1]})")
    print(f"  接著:python tools/roi_annotate.py --image {args.out} --out data/roi_ref/roi.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
