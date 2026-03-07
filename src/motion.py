"""모션 경로 + 이징 함수 모듈"""

import math
from dataclasses import dataclass, field
from typing import Callable


# ─── 이징 함수 ──────────────────────────────────────────────

def ease_linear(t: float) -> float:
    return t


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


def ease_in_out_sine(t: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * t))


def ease_out_expo(t: float) -> float:
    if t >= 1.0:
        return 1.0
    return 1.0 - math.pow(2.0, -10.0 * t)


EASING_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "linear": ease_linear,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_out_sine": ease_in_out_sine,
    "ease_out_expo": ease_out_expo,
}


# ─── 모션 설정 ──────────────────────────────────────────────

@dataclass
class KenBurnsConfig:
    """켄번스 모션 설정"""
    # 시작/종료 뷰 (정규화 좌표 0~1)
    start_cx: float = 0.5   # 시작 중심 X
    start_cy: float = 0.5   # 시작 중심 Y
    start_zoom: float = 1.0  # 시작 줌 레벨

    end_cx: float = 0.5     # 종료 중심 X
    end_cy: float = 0.5     # 종료 중심 Y
    end_zoom: float = 1.3   # 종료 줌 레벨

    easing: str = "ease_in_out_sine"  # 이징 함수 이름

    # 레이어별 속도 배율
    bg_speed: float = 0.18   # 배경 속도 (느리게)
    fg_speed: float = 0.5    # 대상 속도

    duration: float = 5.0    # 초
    fps: int = 30


def get_frame_params(config: KenBurnsConfig, frame_idx: int, total_frames: int):
    """프레임별 (cx, cy, zoom) 파라미터를 배경/대상 각각 반환한다.

    Returns:
        (bg_cx, bg_cy, bg_zoom, fg_cx, fg_cy, fg_zoom)
    """
    t = frame_idx / max(total_frames - 1, 1)
    ease_fn = EASING_FUNCTIONS.get(config.easing, ease_in_out_sine)
    et = ease_fn(t)

    # 기본 이동량
    dcx = config.end_cx - config.start_cx
    dcy = config.end_cy - config.start_cy
    dzoom = config.end_zoom - config.start_zoom

    # 배경 레이어 (느리게)
    bg_cx = config.start_cx + dcx * et * config.bg_speed
    bg_cy = config.start_cy + dcy * et * config.bg_speed
    bg_zoom = config.start_zoom + dzoom * et * config.bg_speed

    # 대상 레이어 (빠르게)
    fg_cx = config.start_cx + dcx * et * config.fg_speed
    fg_cy = config.start_cy + dcy * et * config.fg_speed
    fg_zoom = config.start_zoom + dzoom * et * config.fg_speed

    return bg_cx, bg_cy, bg_zoom, fg_cx, fg_cy, fg_zoom


# ─── 프리셋 ────────────────────────────────────────────────

def _preset_zoom_in(bbox_cx: float = 0.5, bbox_cy: float = 0.5) -> KenBurnsConfig:
    """줌인 — 전체에서 대상으로 접근 (시각적 6% 줌)"""
    return KenBurnsConfig(
        start_cx=0.5, start_cy=0.5, start_zoom=1.0,
        end_cx=bbox_cx, end_cy=bbox_cy, end_zoom=1.06,
        easing="ease_in_out_sine",
    )


def _preset_zoom_out(bbox_cx: float = 0.5, bbox_cy: float = 0.5) -> KenBurnsConfig:
    """줌아웃 — 대상에서 전체로 후퇴 (시각적 6% 줌)"""
    return KenBurnsConfig(
        start_cx=bbox_cx, start_cy=bbox_cy, start_zoom=1.06,
        end_cx=0.5, end_cy=0.5, end_zoom=1.0,
        easing="ease_in_out_sine",
    )


def _preset_pan_left_to_right(**_) -> KenBurnsConfig:
    """패닝 좌→우 (약간의 줌 + 수평 이동)"""
    return KenBurnsConfig(
        start_cx=0.45, start_cy=0.5, start_zoom=1.03,
        end_cx=0.55, end_cy=0.5, end_zoom=1.03,
        easing="ease_in_out_cubic",
    )


def _preset_dramatic_push(bbox_cx: float = 0.5, bbox_cy: float = 0.5) -> KenBurnsConfig:
    """극적 전진 — 강한 줌인 + 느린 시작 (시각적 10% 줌)"""
    return KenBurnsConfig(
        start_cx=0.5, start_cy=0.5, start_zoom=1.0,
        end_cx=bbox_cx, end_cy=bbox_cy, end_zoom=1.10,
        easing="ease_out_expo",
    )


def _preset_slow_drift(**_) -> KenBurnsConfig:
    """느린 표류 — 미세하게 이동 (시각적 2.5% 줌)"""
    return KenBurnsConfig(
        start_cx=0.49, start_cy=0.49, start_zoom=1.0,
        end_cx=0.51, end_cy=0.51, end_zoom=1.025,
        easing="ease_in_out_sine",
    )


PRESETS: dict[str, Callable] = {
    "zoom_in": _preset_zoom_in,
    "zoom_out": _preset_zoom_out,
    "pan_left_right": _preset_pan_left_to_right,
    "dramatic_push": _preset_dramatic_push,
    "slow_drift": _preset_slow_drift,
}


def get_preset(
    name: str,
    bbox: tuple[int, int, int, int] | None = None,
    img_w: int = 1920,
    img_h: int = 1080,
    duration: float = 5.0,
    fps: int = 30,
    extend_ratio: float = 0.20,
) -> KenBurnsConfig:
    """이름으로 프리셋을 가져온다. bbox가 주어지면 대상 중심으로 설정.

    프리셋의 줌 값은 '원본 이미지 기준 시각적 줌'으로 정의되며,
    인페인팅 확장 비율(extend_ratio)을 자동 보정하여
    zoom=1.0일 때 원본 이미지가 프레임을 꽉 채우도록 한다.
    """
    if name not in PRESETS:
        raise ValueError(f"알 수 없는 프리셋: {name}. 사용 가능: {list(PRESETS.keys())}")

    # 바운딩 박스 → 정규화 중심 좌표
    bbox_cx, bbox_cy = 0.5, 0.5
    if bbox is not None:
        x, y, w, h = bbox
        bbox_cx = (x + w / 2) / img_w
        bbox_cy = (y + h / 2) / img_h

    config = PRESETS[name](bbox_cx=bbox_cx, bbox_cy=bbox_cy)
    config.duration = duration
    config.fps = fps

    # 인페인팅 확장 보정: 캔버스가 양쪽으로 extend_ratio씩 확장되므로
    # base_zoom = 1 + 2*extend_ratio 에서 원본 이미지 경계와 일치
    # 프리셋의 줌 값에 base_zoom을 곱해 원본 기준으로 변환
    base_zoom = 1.0 + 2.0 * extend_ratio
    config.start_zoom *= base_zoom
    config.end_zoom *= base_zoom

    # cx, cy도 확장 캔버스 좌표계로 변환
    # 원본 이미지 영역은 캔버스의 [r, 1-r] 구간 (r = extend_ratio / (1 + 2*extend_ratio))
    r = extend_ratio / (1.0 + 2.0 * extend_ratio)
    content_range = 1.0 - 2.0 * r  # 원본이 차지하는 비율
    config.start_cx = r + config.start_cx * content_range
    config.start_cy = r + config.start_cy * content_range
    config.end_cx = r + config.end_cx * content_range
    config.end_cy = r + config.end_cy * content_range

    return config
