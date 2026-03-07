"""컬러 그레이딩 프리셋 모듈

NumPy + OpenCV 기반으로 LUT 없이 실시간 컬러 그레이딩 적용.
"""

import cv2
import numpy as np


# ─── 프리셋 정의 ─────────────────────────────────────────

GRADE_PRESETS: dict[str, dict] = {
    "cinematic": {
        "shadows_tint": (15, 40, 0),      # BGR: 그림자에 틸 추가
        "highlights_tint": (0, -10, 30),   # BGR: 하이라이트에 오렌지
        "contrast": 1.15,
        "saturation": 0.9,
        "brightness": -5,
    },
    "vintage": {
        "shadows_tint": (10, 15, 25),
        "highlights_tint": (5, 10, -5),
        "contrast": 0.95,
        "saturation": 0.7,
        "brightness": 10,
        "gamma": 1.1,
    },
    "cyberpunk": {
        "shadows_tint": (50, 0, 30),
        "highlights_tint": (-15, 15, 0),
        "contrast": 1.3,
        "saturation": 1.2,
        "brightness": -10,
    },
    "pastel": {
        "shadows_tint": (0, 0, 0),
        "highlights_tint": (10, 10, 10),
        "contrast": 0.85,
        "saturation": 0.55,
        "brightness": 20,
    },
    "bw": {
        "shadows_tint": (0, 0, 0),
        "highlights_tint": (0, 0, 0),
        "contrast": 1.25,
        "saturation": 0.0,
        "brightness": 0,
    },
    "warm": {
        "shadows_tint": (0, 10, 20),
        "highlights_tint": (0, 5, 15),
        "contrast": 1.05,
        "saturation": 1.05,
        "brightness": 5,
    },
    "cool": {
        "shadows_tint": (20, 10, 0),
        "highlights_tint": (15, 5, -10),
        "contrast": 1.05,
        "saturation": 0.95,
        "brightness": 0,
    },
}

GRADE_LABELS: dict[str, str] = {
    "none": "없음",
    "cinematic": "시네마틱 (틸/오렌지)",
    "vintage": "빈티지 (따뜻한 레트로)",
    "cyberpunk": "사이버펑크 (보라/네온)",
    "pastel": "파스텔 (부드러운 톤)",
    "bw": "흑백 (하이 콘트라스트)",
    "warm": "웜톤 (따뜻한)",
    "cool": "쿨톤 (차가운)",
}


def apply_color_grade(frame: np.ndarray, grade_name: str) -> np.ndarray:
    """프레임에 컬러 그레이딩을 적용한다.

    Args:
        frame: BGR uint8 이미지
        grade_name: GRADE_PRESETS 키 또는 "none"

    Returns:
        그레이딩된 BGR uint8 이미지
    """
    if grade_name == "none" or grade_name not in GRADE_PRESETS:
        return frame

    g = GRADE_PRESETS[grade_name]
    f = frame.astype(np.float32)

    # 1. 밝기
    f += g.get("brightness", 0)

    # 2. 콘트라스트 (중간값 128 기준)
    contrast = g.get("contrast", 1.0)
    f = (f - 128.0) * contrast + 128.0

    # 3. 감마
    gamma = g.get("gamma", 1.0)
    if gamma != 1.0:
        f = np.clip(f, 0, 255)
        f = 255.0 * np.power(f / 255.0, 1.0 / gamma)

    # 4. 채도
    saturation = g.get("saturation", 1.0)
    if saturation != 1.0:
        f = np.clip(f, 0, 255)
        gray = np.mean(f, axis=2, keepdims=True)
        f = gray + (f - gray) * saturation

    # 5. 그림자/하이라이트 틴트
    shadows_tint = np.array(g.get("shadows_tint", (0, 0, 0)), dtype=np.float32)
    highlights_tint = np.array(g.get("highlights_tint", (0, 0, 0)), dtype=np.float32)

    luminance = np.mean(f, axis=2, keepdims=True) / 255.0
    shadow_mask = 1.0 - luminance       # 어두울수록 1
    highlight_mask = luminance           # 밝을수록 1

    f += shadows_tint * shadow_mask * 0.5
    f += highlights_tint * highlight_mask * 0.5

    return np.clip(f, 0, 255).astype(np.uint8)
