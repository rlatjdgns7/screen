"""배경 복원 모듈 — LaMa AI 인페인팅 + OpenCV 폴백"""

import cv2
import numpy as np
from PIL import Image


def _try_lama_inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """LaMa AI 인페인팅을 시도한다. 실패 시 None 반환."""
    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()

        pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        result = lama(pil_img, pil_mask)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _cv2_inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV 인페인팅 폴백."""
    return cv2.inpaint(image_bgr, mask, 7, cv2.INPAINT_TELEA)


def inpaint_background(
    original_bgr: np.ndarray,
    alpha: np.ndarray,
    dilate_px: int = 20,
    extend_ratio: float = 0.20,
) -> np.ndarray:
    """대상이 제거된 배경을 인페인팅으로 복원한다.

    LaMa AI 인페인팅을 우선 사용하고, 실패 시 OpenCV 폴백.

    Args:
        original_bgr: BGR 원본 이미지
        alpha: 단채널 알파 마스크 (0~255)
        dilate_px: 마스크 팽창 픽셀 (경계 아티팩트 제거용)
        extend_ratio: 배경 확장 비율

    Returns:
        인페인팅된 BGR 배경 이미지 (확장 포함)
    """
    h, w = original_bgr.shape[:2]

    # 1. 배경 확장 (reflect padding)
    pad_x = int(w * extend_ratio)
    pad_y = int(h * extend_ratio)
    extended = cv2.copyMakeBorder(
        original_bgr, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_REFLECT_101,
    )
    alpha_extended = cv2.copyMakeBorder(
        alpha, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )

    # 2. 인페인트 마스크 (대상 영역 = 255)
    inpaint_mask = (alpha_extended > 10).astype(np.uint8) * 255

    # 3. 마스크 팽창
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    # 4. LaMa 인페인팅 시도 → 실패 시 OpenCV 폴백
    print("      인페인팅 엔진: ", end="", flush=True)
    result = _try_lama_inpaint(extended, inpaint_mask)
    if result is not None:
        print("LaMa (AI)")
        return result

    print("OpenCV Telea (폴백)")
    return _cv2_inpaint(extended, inpaint_mask)
