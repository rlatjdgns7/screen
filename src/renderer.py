"""패럴랙스 합성 렌더러 — 2레이어 어파인 변환 + 알파 블렌딩"""

import numpy as np
import cv2


def compute_safe_resolution(
    canvas_w: int,
    canvas_h: int,
    max_zoom: float,
    target_w: int = 1920,
    target_h: int = 1080,
) -> tuple[int, int]:
    """업스케일링 없이 가능한 최대 출력 해상도를 계산한다.

    canvas_w, canvas_h: 인페인팅 확장 후 캔버스 크기
    max_zoom: 파이프라인에서 사용될 최대 줌 레벨
    target_w, target_h: 목표 출력 해상도

    Returns:
        (out_w, out_h) — 항상 다운스케일만 발생하는 최대 해상도
    """
    avail_w = canvas_w / max_zoom
    avail_h = canvas_h / max_zoom

    if avail_w >= target_w and avail_h >= target_h:
        return target_w, target_h

    # 목표 비율을 유지하면서 축소
    shrink = max(target_w / avail_w, target_h / avail_h)
    out_w = int(target_w / shrink) & ~1
    out_h = int(target_h / shrink) & ~1
    return max(out_w, 2), max(out_h, 2)


def crop_and_resize(
    image: np.ndarray,
    cx: float,
    cy: float,
    zoom: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """이미지에서 (cx, cy)를 중심으로 zoom 레벨에 맞게 크롭/리사이즈한다."""
    h, w = image.shape[:2]

    view_w = w / zoom
    view_h = h / zoom

    center_x = cx * w
    center_y = cy * h

    x1 = center_x - view_w / 2
    y1 = center_y - view_h / 2
    x2 = x1 + view_w
    y2 = y1 + view_h

    src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2]])
    dst_pts = np.float32([[0, 0], [out_w, 0], [out_w, out_h]])

    mat = cv2.getAffineTransform(src_pts, dst_pts)
    scale_x = out_w / view_w if view_w > 0 else 1.0
    interp = cv2.INTER_AREA if scale_x < 1.0 else cv2.INTER_LANCZOS4

    return cv2.warpAffine(
        image, mat, (out_w, out_h),
        flags=interp,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def render_frame(
    bg_image: np.ndarray,
    fg_bgra: np.ndarray,
    bg_cx: float, bg_cy: float, bg_zoom: float,
    fg_cx: float, fg_cy: float, fg_zoom: float,
    out_w: int = 1920,
    out_h: int = 1080,
) -> np.ndarray:
    """단일 프레임을 렌더링한다 — 배경+대상 합성."""
    bg_cropped = crop_and_resize(bg_image, bg_cx, bg_cy, bg_zoom, out_w, out_h)
    fg_cropped = crop_and_resize(fg_bgra, fg_cx, fg_cy, fg_zoom, out_w, out_h)

    alpha = fg_cropped[:, :, 3:4].astype(np.float32) / 255.0
    fg_rgb = fg_cropped[:, :, :3].astype(np.float32)
    bg_rgb = bg_cropped[:, :, :3].astype(np.float32)

    blended = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
    return blended.astype(np.uint8)
