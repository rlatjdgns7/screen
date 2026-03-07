"""트랜지션 효과 모듈

이미지 시퀀스 간 전환 효과를 제공한다.
"""

import cv2
import numpy as np


def transition_crossfade(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    progress: float,
) -> np.ndarray:
    """크로스 디졸브: A가 서서히 사라지고 B가 나타남"""
    t = np.clip(progress, 0.0, 1.0)
    return cv2.addWeighted(frame_a, 1.0 - t, frame_b, t, 0)


def transition_zoom_through(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    progress: float,
) -> np.ndarray:
    """줌 스루: A 안으로 빨려 들어가면 B가 줌아웃하며 등장"""
    t = np.clip(progress, 0.0, 1.0)
    h, w = frame_a.shape[:2]

    if t < 0.5:
        # 전반: A가 줌인 + 밝아짐
        zt = t / 0.5  # 0~1
        zoom = 1.0 + zt * 1.5
        bright = zt * 180
        result = _zoom_center(frame_a, zoom)
        result = cv2.add(result, np.full_like(result, int(bright), dtype=np.uint8))
        return result
    else:
        # 후반: B가 줌아웃하며 등장
        zt = (t - 0.5) / 0.5  # 0~1
        zoom = 2.5 - zt * 1.5  # 2.5 → 1.0
        bright = (1.0 - zt) * 180
        result = _zoom_center(frame_b, zoom)
        result = cv2.add(result, np.full_like(result, int(bright), dtype=np.uint8))
        return result


def transition_glitch(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    progress: float,
) -> np.ndarray:
    """글리치 전환: A에 글리치 강화 → 순간 전환 → B 글리치 감소"""
    t = np.clip(progress, 0.0, 1.0)

    if t < 0.5:
        intensity = t / 0.5
        return _apply_glitch_fx(frame_a, intensity)
    else:
        intensity = 1.0 - (t - 0.5) / 0.5
        return _apply_glitch_fx(frame_b, intensity)


def transition_slide_left(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    progress: float,
) -> np.ndarray:
    """슬라이드: A가 왼쪽으로 밀려나고 B가 오른쪽에서 등장"""
    t = np.clip(progress, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep
    h, w = frame_a.shape[:2]
    offset = int(t * w)

    result = np.zeros_like(frame_a)
    # A: 왼쪽으로 이동
    if offset < w:
        result[:, :w - offset] = frame_a[:, offset:]
    # B: 오른쪽에서 등장
    if offset > 0:
        result[:, w - offset:] = frame_b[:, :offset]
    return result


def transition_wipe_down(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    progress: float,
) -> np.ndarray:
    """와이프: 위에서 아래로 B가 덮어씀"""
    t = np.clip(progress, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    h, w = frame_a.shape[:2]
    split = int(t * h)

    result = frame_a.copy()
    if split > 0:
        result[:split] = frame_b[:split]
    return result


# ─── 내부 유틸 ─────────────────────────────────────────


def _zoom_center(frame: np.ndarray, zoom: float) -> np.ndarray:
    """프레임 중심 줌"""
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    half_w = w / (2.0 * zoom)
    half_h = h / (2.0 * zoom)

    x1 = max(0, int(cx - half_w))
    y1 = max(0, int(cy - half_h))
    x2 = min(w, int(cx + half_w))
    y2 = min(h, int(cy + half_h))

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _apply_glitch_fx(frame: np.ndarray, intensity: float) -> np.ndarray:
    """강도 조절 가능한 글리치 효과"""
    if intensity < 0.05:
        return frame

    result = frame.copy()
    h, w = frame.shape[:2]

    n_strips = int(3 + 12 * intensity)
    max_shift = int(60 * intensity)

    for _ in range(n_strips):
        y = np.random.randint(0, max(1, h - 20))
        strip_h = np.random.randint(2, max(3, int(30 * intensity)))
        y2 = min(y + strip_h, h)
        shift = np.random.randint(-max_shift, max_shift + 1)
        result[y:y2] = np.roll(result[y:y2], shift, axis=1)

    # 색수차
    chroma = int(15 * intensity)
    if chroma > 0:
        b, g, r = cv2.split(result)
        M_r = np.float32([[1, 0, chroma], [0, 1, 0]])
        M_b = np.float32([[1, 0, -chroma], [0, 1, 0]])
        r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        result = cv2.merge([b, g, r])

    return result


# ─── 트랜지션 디스패치 ────────────────────────────────────

TRANSITIONS: dict[str, callable] = {
    "crossfade": transition_crossfade,
    "zoom_through": transition_zoom_through,
    "glitch": transition_glitch,
    "slide_left": transition_slide_left,
    "wipe_down": transition_wipe_down,
    "cut": None,  # 컷은 트랜지션 없음
}

TRANSITION_LABELS: dict[str, str] = {
    "cut": "컷 (즉시 전환)",
    "crossfade": "크로스 디졸브",
    "zoom_through": "줌 스루",
    "glitch": "글리치 전환",
    "slide_left": "슬라이드 (좌로)",
    "wipe_down": "와이프 (위→아래)",
}


def apply_transition(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    transition_name: str,
    progress: float,
) -> np.ndarray:
    """트랜지션을 적용한다.

    Args:
        frame_a: 이전 클립의 프레임
        frame_b: 다음 클립의 프레임
        transition_name: TRANSITIONS 키
        progress: 0.0 (A 100%) ~ 1.0 (B 100%)

    Returns:
        트랜지션된 프레임
    """
    fn = TRANSITIONS.get(transition_name)
    if fn is None:
        return frame_b if progress >= 0.5 else frame_a
    return fn(frame_a, frame_b, progress)
