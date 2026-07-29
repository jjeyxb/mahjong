"""邊框剝除與比例修正。

擷取到的視窗影像,遊戲畫布外圍可能包著三種東西:

1. **作業系統的視窗裝飾** —— macOS 的標題列(實測 Retina 下為 56 px)、
   Windows 的邊框。這是以「整個視窗」為單位擷取的必然代價。
2. **遊戲自己的黑邊** —— 視窗比例與畫布比例不符時,雀魂會補上
   letterbox(上下)或 pillarbox(左右)。
3. **瀏覽器 chrome** —— 分頁列、網址列、書籤列。

三者的共同特徵是「一整條線的顏色高度一致」,所以這裡用同一套邊緣剝除演算法
處理:從四邊最外側往內走,只要該行/列仍有足夠比例的像素接近該邊起始線的顏色,
就繼續剝除。

這個做法比「假設邊框是黑色」穩健得多 —— 實測 macOS 亮色主題的標題列是
BGR(235, 234, 232),用黑色當基準完全抓不到。
"""

from __future__ import annotations

import numpy as np

from majsoul_copilot.utils.geometry import Rect

__all__ = ["estimate_background", "find_content_rect", "fit_aspect", "peel_uniform_edges"]


def estimate_background(image: np.ndarray, *, thickness: int = 8) -> np.ndarray:
    """推測影像最外圈的代表色,回傳 BGR uint8 三元素陣列。

    取四條邊的邊緣帶(**刻意排除四個角落**)的中位數。

    排除角落是必要的:macOS 的視窗有圓角,轉角處是完全透明的像素,
    在 premultiplied alpha 下會變成純黑。若把角落算進去,亮色標題列的視窗
    會被誤判成「背景是黑色」,後續所有邊框偵測都跟著錯。

    本函式目前僅供診斷與視覺化使用(見 :mod:`majsoul_copilot.calibration.debug`);
    :func:`find_content_rect` 用的是更穩健的逐線比對,不依賴單一背景色。
    """
    height, width = image.shape[:2]
    t = max(1, min(thickness, height // 4, width // 4))
    samples = [
        image[:t, t:-t].reshape(-1, 3),  # 上緣(去角)
        image[-t:, t:-t].reshape(-1, 3),  # 下緣(去角)
        image[t:-t, :t].reshape(-1, 3),  # 左緣(去角)
        image[t:-t, -t:].reshape(-1, 3),  # 右緣(去角)
    ]
    stacked = np.concatenate([s for s in samples if s.size])
    if stacked.size == 0:  # 影像太小,退回整張的中位數
        stacked = image.reshape(-1, 3)
    return np.median(stacked, axis=0).astype(np.uint8)


def _match_fraction(
    image: np.ndarray, reference: np.ndarray, axis: int, tolerance: int
) -> np.ndarray:
    """每條線上「顏色接近 reference」的像素比例。

    axis=1 → 逐列(row);axis=0 → 逐行(column)。
    """
    diff = np.abs(image.astype(np.int16) - reference.astype(np.int16)).max(axis=2)
    return (diff <= tolerance).mean(axis=axis)


def _peel_forward(fractions: np.ndarray, uniformity: float, limit: int) -> int:
    """從索引 0 往前走,回傳第一個「不夠一致」的位置(即要剝掉幾條線)。"""
    keep = fractions >= uniformity
    keep[limit:] = False  # 不允許剝超過 limit,避免整張同色時把影像吃光
    return int(np.argmin(keep))


def peel_uniform_edges(
    image: np.ndarray,
    *,
    tolerance: int = 20,
    uniformity: float = 0.7,
    max_peel_ratio: float = 0.4,
    max_passes: int = 4,
) -> Rect:
    """從四邊剝除顏色一致的帶狀區域,回傳剩下的內容矩形。

    採**多輪**剝除。同一邊可能疊了好幾層不同顏色的帶狀區域 —— 典型情境是
    「亮色的 macOS 標題列」外加「遊戲自己補的黑色 letterbox」。單輪只用最外側
    那條線的顏色當參考,剝完第一層就會停住;每輪重新取參考色才能繼續往內剝。

    Args:
        image: BGR uint8 影像。
        tolerance: 像素與該邊參考色的通道差異在此值內就算「同色」。
        uniformity: 一條線上要有這個比例以上的像素同色,才會被剝掉。
            用比例而非「全部相同」,是為了容忍標題列上的文字與按鈕、
            黑邊上的滑鼠殘影、圓角的反鋸齒像素。
        max_peel_ratio: 每個方向**累計**最多只能剝掉這個比例的尺寸。
            整張影像同色時這個上限是唯一的煞車。
        max_passes: 最多剝幾輪。沒有任何一邊還能繼續剝時會提早結束。

    Returns:
        內容矩形(相對於輸入影像的像素座標)。完全沒東西可剝時即為整張影像。
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return Rect(0, 0, width, height)

    v_limit = max(1, int(height * max_peel_ratio))
    h_limit = max(1, int(width * max_peel_ratio))
    top = bottom = left = right = 0  # 各方向的累計剝除量

    for _ in range(max_passes):
        view = image[top : height - bottom, left : width - right]
        if view.shape[0] <= 1 or view.shape[1] <= 1:
            break

        # 每輪、每邊都重新以當前最外側那條線的中位數色為參考。
        d_top = _peel_forward(
            _match_fraction(view, np.median(view[0], axis=0), 1, tolerance),
            uniformity,
            v_limit - top,
        )
        d_bottom = _peel_forward(
            _match_fraction(view, np.median(view[-1], axis=0), 1, tolerance)[::-1],
            uniformity,
            v_limit - bottom,
        )
        d_left = _peel_forward(
            _match_fraction(view, np.median(view[:, 0], axis=0), 0, tolerance),
            uniformity,
            h_limit - left,
        )
        d_right = _peel_forward(
            _match_fraction(view, np.median(view[:, -1], axis=0), 0, tolerance)[::-1],
            uniformity,
            h_limit - right,
        )

        if not (d_top or d_bottom or d_left or d_right):
            break
        top += d_top
        bottom += d_bottom
        left += d_left
        right += d_right

    return Rect.from_bounds(left, top, width - right, height - bottom)


def find_content_rect(
    image: np.ndarray,
    *,
    tolerance: int = 20,
    uniformity: float = 0.7,
    max_peel_ratio: float = 0.4,
    min_content_ratio: float = 0.3,
) -> Rect | None:
    """找出遊戲畫布區域,剝掉外圍的視窗裝飾與黑邊。

    Args:
        min_content_ratio: 結果面積至少要佔原影像的這個比例,否則視為偵測失敗
            並回傳 None。這道檢查擋掉兩種退化情形:整張影像近乎同色,
            以及畫面上根本不是遊戲。

    Returns:
        內容矩形,偵測失敗時為 None。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"find_content_rect 需要 BGR 影像,收到 shape={image.shape}")

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return None

    rect = peel_uniform_edges(
        image, tolerance=tolerance, uniformity=uniformity, max_peel_ratio=max_peel_ratio
    )
    if rect.width == 0 or rect.height == 0:
        return None
    if rect.area < min_content_ratio * height * width:
        return None
    return rect


def fit_aspect(rect: Rect, target_aspect: float) -> Rect:
    """把矩形置中裁切成指定寬高比。

    只縮不放 —— 過寬就裁左右,過高就裁上下,結果一定被原矩形包含。
    """
    if target_aspect <= 0:
        raise ValueError(f"target_aspect 須為正數,收到 {target_aspect}")
    if rect.width == 0 or rect.height == 0:
        return rect

    if rect.aspect > target_aspect:  # 太寬 → 裁左右
        new_width = max(1, round(rect.height * target_aspect))
        return Rect(rect.x + (rect.width - new_width) // 2, rect.y, new_width, rect.height)

    new_height = max(1, round(rect.width / target_aspect))
    return Rect(rect.x, rect.y + (rect.height - new_height) // 2, rect.width, new_height)
