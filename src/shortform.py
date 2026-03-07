"""숏폼 임팩트 효과 모듈

이미지 1장 → 줌 펄스 + 셰이크 + 글리치 → 역동적 숏폼 MP4
AI 처리 없음, 순수 이미지 변환으로 초고속 처리.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class ShortformConfig:
    """숏폼 효과 설정"""
    duration: float = 5.0
    fps: int = 30
    bpm: float = 120.0

    # 효과 강도
    zoom_amplitude: float = 0.02
    shake_intensity: float = 3.0
    chroma_base: float = 0.75
    chroma_pulse: float = 1.5
    vignette_strength: float = 0.10
    grain_intensity: float = 6.0
    glitch_probability: float = 0.02
    drift_range: float = 7.5

    # 인트로/아웃트로
    intro_duration: float = 0.4
    outro_duration: float = 0.3

    # 등장 효과 / 컬러 그레이딩
    entry_style: str = "glitch"     # glitch | fade | slide_left | slide_right | slide_up | spin | none
    color_grade: str = "none"       # none | cinematic | vintage | cyberpunk | pastel | bw | warm | cool
    text_overlay: str = ""
    text_position: str = "bottom"   # top | center | bottom
    text_font_size: int = 48

    # 출력
    output_w: int = 1080
    output_h: int = 1920
    crf: int = 18
    use_nvenc: bool = False


# ─── 내부 유틸 ─────────────────────────────────────────


def _clamp_output_to_native(
    img_w: int, img_h: int,
    target_w: int, target_h: int,
    headroom: float = 1.15,
) -> tuple[int, int]:
    """업스케일링 없이 가능한 최대 출력 해상도를 반환한다.

    입력 이미지가 target 해상도를 충족하면 그대로,
    부족하면 원본 화질을 유지하는 최대 크기로 축소한다.
    NVENC 최소 해상도(146x146)를 보장한다.
    """
    # NVENC 하드웨어 인코더 최소 해상도
    MIN_DIM = 146

    scale = max(target_w / img_w, target_h / img_h) * headroom
    if scale <= 1.0:
        return target_w, target_h

    out_w = int(target_w / scale) & ~1  # 짝수 보장 (코덱 호환)
    out_h = int(target_h / scale) & ~1
    out_w = max(out_w, MIN_DIM) & ~1
    out_h = max(out_h, MIN_DIM) & ~1
    return out_w, out_h


def _prepare_canvas(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """이미지를 출력 해상도 + 여유분으로 cover crop 준비"""
    h, w = image.shape[:2]
    headroom = 1.15
    scale = max(out_w / w, out_h / h) * headroom
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _generate_shake_keys(n_frames: int, keyframe_interval: int = 6) -> np.ndarray:
    """유기적 카메라 셰이크 값 (Hermite smoothstep 보간)"""
    n_keys = n_frames // keyframe_interval + 2
    keys = np.random.randn(n_keys, 3).astype(np.float32)
    keys[:, 2] *= 0.4  # 회전은 약하게

    result = np.zeros((n_frames, 3), dtype=np.float32)
    for i in range(n_frames):
        pos = i / keyframe_interval
        k0 = int(pos)
        k1 = min(k0 + 1, n_keys - 1)
        frac = pos - k0
        frac = frac * frac * (3.0 - 2.0 * frac)
        result[i] = keys[k0] * (1.0 - frac) + keys[k1] * frac

    return result


def _make_vignette_mask(h: int, w: int, strength: float) -> np.ndarray:
    """비네팅 마스크 사전 계산"""
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist_sq = ((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2
    mask = 1.0 - strength * dist_sq
    return np.clip(mask, 0.0, 1.0).astype(np.float32)[:, :, np.newaxis]


# ─── 효과 함수 ─────────────────────────────────────────


def _apply_chromatic_aberration(frame: np.ndarray, intensity: float) -> np.ndarray:
    """RGB 채널 색수차 — R은 오른쪽, B는 왼쪽으로 어긋남"""
    shift = int(round(intensity))
    if shift <= 0:
        return frame

    b, g, r = cv2.split(frame)
    h, w = frame.shape[:2]
    M_r = np.float32([[1, 0, shift], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
    r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
    b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b, g, r])


def _apply_glitch(frame: np.ndarray) -> np.ndarray:
    """글리치 효과 — 수평 스트립 변위 + 강한 색수차"""
    result = frame.copy()
    h, w = frame.shape[:2]

    n_strips = np.random.randint(3, 10)
    for _ in range(n_strips):
        y = np.random.randint(0, max(1, h - 30))
        strip_h = np.random.randint(2, 25)
        y2 = min(y + strip_h, h)
        shift = np.random.randint(-40, 40)
        result[y:y2] = np.roll(result[y:y2], shift, axis=1)

    return _apply_chromatic_aberration(result, 10)


def _apply_intro_glitch(frame: np.ndarray, progress: float) -> np.ndarray:
    """인트로: 글리치 스트립이 제자리로 수렴하며 등장"""
    if progress >= 1.0:
        return frame

    result = frame.copy()
    h, w = frame.shape[:2]
    n_strips = 12
    strip_h = h // n_strips + 1
    displacement = int((1.0 - progress) ** 2 * w * 0.5)

    for s in range(n_strips):
        y1 = s * strip_h
        y2 = min((s + 1) * strip_h, h)
        if y1 >= h:
            break
        offset = displacement if s % 2 == 0 else -displacement
        offset += int(np.random.randn() * displacement * 0.2)
        result[y1:y2] = np.roll(result[y1:y2], offset, axis=1)

    chroma = int((1.0 - progress) * 15)
    if chroma > 0:
        result = _apply_chromatic_aberration(result, chroma)
    return result


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _apply_entry(frame: np.ndarray, progress: float, style: str,
                 out_w: int, out_h: int) -> np.ndarray:
    """등장 애니메이션 디스패치"""
    if style == "none" or progress >= 1.0:
        return frame

    if style == "glitch":
        return _apply_intro_glitch(frame, progress)

    t = _ease_out_cubic(np.clip(progress, 0.0, 1.0))

    if style == "fade":
        return (frame.astype(np.float32) * t).astype(np.uint8)

    elif style == "slide_left":
        offset = int((1.0 - t) * out_w)
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    elif style == "slide_right":
        offset = int((1.0 - t) * -out_w)
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    elif style == "slide_up":
        offset = int((1.0 - t) * out_h)
        M = np.float32([[1, 0, 0], [0, 1, offset]])
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    elif style == "spin":
        angle = (1.0 - t) * 360.0
        scale = max(t, 0.01)
        center = (out_w // 2, out_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    return frame


# ─── 메인 렌더링 ───────────────────────────────────────


def render_shortform(
    input_path: str,
    output_path: str,
    config: ShortformConfig,
    progress_cb: Optional[Callable[[str, float], None]] = None,
):
    """숏폼 임팩트 영상 렌더링

    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 MP4 경로
        config: 효과 설정
        progress_cb: 진행률 콜백 (message, percent)
    """
    from src.video import create_video_writer, write_frame, finalize

    # 1. 이미지 로드 + 캔버스 준비
    if progress_cb:
        progress_cb("이미지 준비 중…", 5)

    raw = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {input_path}")

    # 입력 해상도에 맞춰 출력 해상도 자동 조정 (업스케일 방지)
    orig_w, orig_h = config.output_w, config.output_h
    h_raw, w_raw = raw.shape[:2]
    config.output_w, config.output_h = _clamp_output_to_native(
        w_raw, h_raw, config.output_w, config.output_h,
    )
    if (config.output_w, config.output_h) != (orig_w, orig_h):
        print(f"      ⚠ 입력 해상도({w_raw}×{h_raw})가 목표({orig_w}×{orig_h})에 부족")
        print(f"        → 원본 화질 유지를 위해 {config.output_w}×{config.output_h}로 출력")

    canvas = _prepare_canvas(raw, config.output_w, config.output_h)
    canvas_h, canvas_w = canvas.shape[:2]
    cx_base = canvas_w / 2.0
    cy_base = canvas_h / 2.0

    total_frames = int(config.duration * config.fps)

    # 2. 사전 계산
    if progress_cb:
        progress_cb("효과 계산 중…", 10)

    shake_keys = _generate_shake_keys(total_frames)
    vig_mask = _make_vignette_mask(config.output_h, config.output_w,
                                   config.vignette_strength)

    # 3. 영상 인코딩 시작
    writer = create_video_writer(
        output_path, width=config.output_w, height=config.output_h,
        fps=config.fps, crf=config.crf, use_nvenc=config.use_nvenc,
    )

    if progress_cb:
        progress_cb(f"렌더링 중… 0/{total_frames}", 15)

    intro_frames = int(config.intro_duration * config.fps)
    outro_start = total_frames - int(config.outro_duration * config.fps)

    for i in range(total_frames):
        t = i / config.fps

        # ── 비트 동기 줌 펄스 (심장박동 느낌) ──
        beat_phase = (t * config.bpm / 60.0) % 1.0
        pulse = np.exp(-6.0 * beat_phase)
        zoom = 1.0 + config.zoom_amplitude * pulse

        # ── 기저 드리프트 (느린 부유감) ──
        dp = t / config.duration
        drift_x = np.sin(2 * np.pi * dp * 0.7) * config.drift_range
        drift_y = np.cos(2 * np.pi * dp * 1.1) * config.drift_range * 0.6

        # ── 카메라 셰이크 ──
        sx, sy, sr = shake_keys[i] * config.shake_intensity

        # ── 합성 어파인 변환: 줌 + 이동 + 회전 ──
        frame_cx = cx_base + drift_x + sx
        frame_cy = cy_base + drift_y + sy
        rotation = sr * 0.3

        half_w = config.output_w / (2.0 * zoom)
        half_h = config.output_h / (2.0 * zoom)

        src_pts = np.float32([
            [frame_cx - half_w, frame_cy - half_h],
            [frame_cx + half_w, frame_cy - half_h],
            [frame_cx - half_w, frame_cy + half_h],
        ])

        if abs(rotation) > 0.01:
            cos_r = np.cos(np.radians(rotation))
            sin_r = np.sin(np.radians(rotation))
            for p in src_pts:
                dx, dy = p[0] - frame_cx, p[1] - frame_cy
                p[0] = frame_cx + dx * cos_r - dy * sin_r
                p[1] = frame_cy + dx * sin_r + dy * cos_r

        dst_pts = np.float32([
            [0, 0],
            [config.output_w, 0],
            [0, config.output_h],
        ])

        M = cv2.getAffineTransform(src_pts, dst_pts)
        frame = cv2.warpAffine(
            canvas, M, (config.output_w, config.output_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
        )

        # ── 크로매틱 애버레이션 (펄스에 동기) ──
        chroma = config.chroma_base + config.chroma_pulse * pulse
        frame = _apply_chromatic_aberration(frame, chroma)

        # ── 인트로 등장 효과 ──
        if i < intro_frames:
            frame = _apply_entry(frame, i / max(intro_frames, 1),
                                 config.entry_style, config.output_w, config.output_h)
        # ── 랜덤 글리치 ──
        elif np.random.random() < config.glitch_probability:
            frame = _apply_glitch(frame)

        # ── 비네팅 + 그레인 + 페이드 (float 도메인) ──
        frame_f = frame.astype(np.float32) * vig_mask

        if config.grain_intensity > 0:
            noise = np.random.randn(
                config.output_h, config.output_w, 1,
            ).astype(np.float32) * config.grain_intensity
            frame_f = frame_f + noise

        # ── 아웃트로 페이드 투 블랙 ──
        if i >= outro_start:
            fade = 1.0 - (i - outro_start) / max(total_frames - outro_start, 1)
            frame_f *= fade

        frame = np.clip(frame_f, 0, 255).astype(np.uint8)

        # ── 컬러 그레이딩 ──
        if config.color_grade != "none":
            from src.color_grade import apply_color_grade
            frame = apply_color_grade(frame, config.color_grade)

        # ── 텍스트 오버레이 ──
        if config.text_overlay:
            from src.text_overlay import render_text_on_frame
            frame = render_text_on_frame(
                frame, config.text_overlay,
                position=config.text_position,
                font_size=config.text_font_size,
            )

        write_frame(writer, frame)

        if progress_cb and (i % 5 == 0 or i == total_frames - 1):
            pct = 15 + 80 * (i + 1) / total_frames
            progress_cb(f"렌더링 중… {i+1}/{total_frames}", pct)

    finalize(writer)

    if progress_cb:
        file_bytes = os.path.getsize(output_path)
        if file_bytes >= 1024 * 1024:
            size_str = f"{file_bytes / (1024*1024):.1f} MB"
        else:
            size_str = f"{file_bytes / 1024:.1f} KB"
        progress_cb(f"완료! {size_str} — {output_path}", 100)
