"""텍스트 오버레이 모듈 — PIL 기반 한글 지원"""

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# 한글 지원 폰트 후보 (Windows 기준)
_FONT_CANDIDATES = [
    "malgun.ttf",        # 맑은 고딕
    "NanumGothicBold.ttf",
    "meiryo.ttc",
    "arial.ttf",
]


def _find_font(size: int):
    """사용 가능한 폰트를 찾아 반환한다."""
    if not _PIL_AVAILABLE:
        return None
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_text_on_frame(
    frame: np.ndarray,
    text: str,
    position: str = "bottom",
    font_size: int = 48,
    color: tuple[int, int, int] = (255, 255, 255),
    shadow: bool = True,
    margin: int = 40,
) -> np.ndarray:
    """프레임 위에 텍스트를 렌더링한다.

    Args:
        frame: BGR uint8 이미지
        text: 렌더링할 텍스트 (빈 문자열이면 원본 반환)
        position: "top" | "center" | "bottom"
        font_size: 폰트 크기 (px)
        color: RGB 색상
        shadow: 그림자 효과 사용 여부
        margin: 화면 가장자리 여백

    Returns:
        텍스트가 합성된 BGR uint8 이미지
    """
    if not text or not text.strip():
        return frame

    if not _PIL_AVAILABLE:
        # PIL 없으면 OpenCV 기본 텍스트 (한글 미지원)
        h, w = frame.shape[:2]
        result = frame.copy()
        scale = font_size / 30.0
        thickness = max(1, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if position == "top":
            y = margin + th
        elif position == "center":
            y = h // 2 + th // 2
        else:
            y = h - margin
        x = (w - tw) // 2
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2)
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color[::-1], thickness)
        return result

    # PIL 경로: 한글 지원
    h, w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _find_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (w - tw) // 2
    if position == "top":
        y = margin
    elif position == "center":
        y = (h - th) // 2
    else:
        y = h - th - margin

    if shadow:
        offset = max(2, font_size // 20)
        draw.text((x + offset, y + offset), text, font=font, fill=(0, 0, 0))

    draw.text((x, y), text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
