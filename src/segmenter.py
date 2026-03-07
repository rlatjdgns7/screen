"""대상/배경 분리 모듈 — rembg + BiRefNet/ISNet"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session


@dataclass
class SegmentResult:
    """세그멘테이션 결과"""
    foreground: np.ndarray   # BGRA 대상 이미지
    alpha: np.ndarray        # 단채널 알파 마스크 (0~255)
    bbox: tuple[int, int, int, int]  # (x, y, w, h) 바운딩 박스
    original: np.ndarray     # BGR 원본 이미지


# 모델 우선순위: 정확도 높은 순서대로 시도
_MODEL_PRIORITY = [
    "birefnet-general",
    "isnet-general-use",
    "u2net",
]


def _create_session(model_name: str | None = None):
    """세그멘테이션 세션을 생성한다. 모델 로드 실패 시 폴백."""
    if model_name:
        try:
            return new_session(model_name), model_name
        except Exception:
            pass

    for name in _MODEL_PRIORITY:
        try:
            return new_session(name), name
        except Exception:
            continue

    raise RuntimeError("사용 가능한 세그멘테이션 모델이 없습니다.")


def segment_image(
    image_input: str | np.ndarray,
    model_name: str | None = None,
    feather_radius: int = 3,
) -> SegmentResult:
    """이미지에서 대상과 배경을 분리한다.

    Args:
        image_input: 입력 이미지 경로(str) 또는 BGR numpy 배열
        model_name: rembg 모델 이름 (None이면 자동 선택)
        feather_radius: 마스크 페더링 가우시안 블러 반경 (0이면 비활성)

    Returns:
        SegmentResult 객체
    """
    if isinstance(image_input, np.ndarray):
        original_bgr = image_input.copy()
    else:
        original_bgr = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_input}")

    pil_img = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    session, used_model = _create_session(model_name)
    print(f"      세그멘테이션 모델: {used_model}")
    pil_result = remove(pil_img, session=session)

    rgba = np.array(pil_result)
    foreground_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    alpha = foreground_bgra[:, :, 3].copy()

    # 마스크 페더링 (가우시안 블러) — 값을 작게 해서 에지 선명도 유지
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

    foreground_bgra[:, :, 3] = alpha
    bbox = _compute_bbox(alpha)

    return SegmentResult(
        foreground=foreground_bgra,
        alpha=alpha,
        bbox=bbox,
        original=original_bgr,
    )


def _compute_bbox(alpha: np.ndarray, threshold: int = 10) -> tuple[int, int, int, int]:
    """알파 마스크에서 바운딩 박스를 계산한다."""
    mask = (alpha > threshold).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = alpha.shape
        return (0, 0, w, h)
    return cv2.boundingRect(coords)
