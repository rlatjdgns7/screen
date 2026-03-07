"""멀티 이미지 시퀀스 렌더링 모듈

여러 이미지를 연속으로 효과 적용 + 트랜지션 연결하여 하나의 MP4로 출력.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from src.color_grade import apply_color_grade
from src.text_overlay import render_text_on_frame
from src.transitions import apply_transition, TRANSITIONS


# ─── 설정 ──────────────────────────────────────────────


@dataclass
class ClipConfig:
    """이미지 한 장의 연출 설정"""
    path: str = ""
    duration: float = 3.0
    effect: str = "kenburns"        # kenburns | shortform | static
    preset: str = "zoom_in"         # 켄번스 프리셋
    entry: str = "none"             # none | fade | slide_left | slide_right | slide_up | glitch | spin
    color_grade: str = "none"
    text: str = ""
    text_position: str = "bottom"   # top | center | bottom
    font_size: int = 48


@dataclass
class SequenceConfig:
    """멀티 이미지 시퀀스 설정"""
    clips: list[ClipConfig] = field(default_factory=list)
    transition: str = "crossfade"
    transition_duration: float = 0.5
    output_w: int = 1920
    output_h: int = 1080
    fps: int = 30
    crf: int = 18
    use_nvenc: bool = False


# ─── 등장 애니메이션 ──────────────────────────────────────


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _apply_entry(
    frame: np.ndarray,
    progress: float,
    entry_style: str,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """등장 애니메이션을 적용한다. progress: 0.0(시작) ~ 1.0(완료)"""
    if entry_style == "none" or progress >= 1.0:
        return frame

    t = _ease_out_cubic(np.clip(progress, 0.0, 1.0))

    if entry_style == "fade":
        return (frame.astype(np.float32) * t).astype(np.uint8)

    elif entry_style == "slide_left":
        offset = int((1.0 - t) * out_w)
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    elif entry_style == "slide_right":
        offset = int((1.0 - t) * -out_w)
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    elif entry_style == "slide_up":
        offset = int((1.0 - t) * out_h)
        M = np.float32([[1, 0, 0], [0, 1, offset]])
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    elif entry_style == "spin":
        angle = (1.0 - t) * 360.0
        scale = t
        center = (out_w // 2, out_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, max(scale, 0.01))
        return cv2.warpAffine(frame, M, (out_w, out_h), borderValue=(0, 0, 0))

    elif entry_style == "glitch":
        return _entry_glitch(frame, progress)

    return frame


def _entry_glitch(frame: np.ndarray, progress: float) -> np.ndarray:
    """글리치 스트립 등장 효과"""
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

    # 색수차
    chroma = int((1.0 - progress) * 15)
    if chroma > 0:
        b, g, r = cv2.split(result)
        M_r = np.float32([[1, 0, chroma], [0, 1, 0]])
        M_b = np.float32([[1, 0, -chroma], [0, 1, 0]])
        r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        result = cv2.merge([b, g, r])

    return result


# ─── 단일 클립 프레임 생성 ────────────────────────────────


def _render_static_frames(
    image: np.ndarray,
    n_frames: int,
    out_w: int,
    out_h: int,
) -> list[np.ndarray]:
    """정적 이미지 프레임 생성 (cover crop)"""
    h, w = image.shape[:2]
    scale = max(out_w / w, out_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # 중앙 크롭
    x = (new_w - out_w) // 2
    y = (new_h - out_h) // 2
    cropped = resized[y:y + out_h, x:x + out_w]

    return [cropped.copy() for _ in range(n_frames)]


def _render_kenburns_frames(
    image: np.ndarray,
    n_frames: int,
    out_w: int,
    out_h: int,
    preset: str = "zoom_in",
) -> list[np.ndarray]:
    """켄번스 효과 프레임 생성 (세그멘테이션 없이 단순 줌/패닝)"""
    h, w = image.shape[:2]

    # 캔버스 준비 (여유분 포함)
    headroom = 1.15
    scale = max(out_w / w, out_h / h) * headroom
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    canvas = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # 프리셋별 시작/끝 설정
    presets = {
        "zoom_in": {"s_zoom": 1.0, "e_zoom": 1.08, "s_cx": 0.5, "s_cy": 0.5, "e_cx": 0.5, "e_cy": 0.45},
        "zoom_out": {"s_zoom": 1.08, "e_zoom": 1.0, "s_cx": 0.5, "s_cy": 0.45, "e_cx": 0.5, "e_cy": 0.5},
        "pan_left_right": {"s_zoom": 1.03, "e_zoom": 1.03, "s_cx": 0.42, "s_cy": 0.5, "e_cx": 0.58, "e_cy": 0.5},
        "dramatic_push": {"s_zoom": 1.0, "e_zoom": 1.15, "s_cx": 0.5, "s_cy": 0.5, "e_cx": 0.5, "e_cy": 0.42},
        "slow_drift": {"s_zoom": 1.0, "e_zoom": 1.03, "s_cx": 0.48, "s_cy": 0.48, "e_cx": 0.52, "e_cy": 0.52},
    }
    p = presets.get(preset, presets["zoom_in"])

    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        # smoothstep easing
        et = t * t * (3.0 - 2.0 * t)

        zoom = p["s_zoom"] + (p["e_zoom"] - p["s_zoom"]) * et
        cx = (p["s_cx"] + (p["e_cx"] - p["s_cx"]) * et) * new_w
        cy = (p["s_cy"] + (p["e_cy"] - p["s_cy"]) * et) * new_h

        half_w = out_w / (2.0 * zoom)
        half_h = out_h / (2.0 * zoom)

        src_pts = np.float32([
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx - half_w, cy + half_h],
        ])
        dst_pts = np.float32([
            [0, 0], [out_w, 0], [0, out_h],
        ])

        M = cv2.getAffineTransform(src_pts, dst_pts)
        frame = cv2.warpAffine(
            canvas, M, (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT,
        )
        frames.append(frame)

    return frames


def _render_shortform_frames(
    image: np.ndarray,
    n_frames: int,
    out_w: int,
    out_h: int,
    fps: int = 30,
    bpm: float = 120.0,
) -> list[np.ndarray]:
    """숏폼 임팩트 프레임 생성 (줌펄스 + 셰이크, 가벼운 버전)"""
    h, w = image.shape[:2]

    headroom = 1.15
    scale = max(out_w / w, out_h / h) * headroom
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    canvas = cv2.resize(image, (new_w, new_h), interpolation=interp)

    cx_base = new_w / 2.0
    cy_base = new_h / 2.0

    frames = []
    for i in range(n_frames):
        t = i / fps
        beat_phase = (t * bpm / 60.0) % 1.0
        pulse = np.exp(-6.0 * beat_phase)
        zoom = 1.0 + 0.03 * pulse

        # 약한 드리프트
        dp = i / max(n_frames, 1)
        drift_x = np.sin(2 * np.pi * dp * 0.7) * 8.0
        drift_y = np.cos(2 * np.pi * dp * 1.1) * 5.0

        fcx = cx_base + drift_x
        fcy = cy_base + drift_y

        half_w = out_w / (2.0 * zoom)
        half_h = out_h / (2.0 * zoom)

        src_pts = np.float32([
            [fcx - half_w, fcy - half_h],
            [fcx + half_w, fcy - half_h],
            [fcx - half_w, fcy + half_h],
        ])
        dst_pts = np.float32([
            [0, 0], [out_w, 0], [0, out_h],
        ])

        M = cv2.getAffineTransform(src_pts, dst_pts)
        frame = cv2.warpAffine(
            canvas, M, (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT,
        )
        frames.append(frame)

    return frames


# ─── 메인 시퀀스 렌더링 ───────────────────────────────────


def render_sequence(
    config: SequenceConfig,
    output_path: str,
    progress_cb: Optional[Callable[[str, float], None]] = None,
):
    """멀티 이미지 시퀀스를 하나의 영상으로 렌더링한다.

    Args:
        config: 시퀀스 설정
        output_path: 출력 MP4 경로
        progress_cb: 진행률 콜백 (message, percent)
    """
    from src.video import create_video_writer, write_frame, finalize

    if not config.clips:
        raise ValueError("클립이 없습니다.")

    out_w = config.output_w
    out_h = config.output_h
    fps = config.fps
    trans_frames = int(config.transition_duration * fps)

    # 총 프레임 수 계산
    total_clip_frames = 0
    for clip in config.clips:
        total_clip_frames += int(clip.duration * fps)
    # 트랜지션 겹침 제거
    total_frames = total_clip_frames - trans_frames * max(0, len(config.clips) - 1)
    total_frames = max(total_frames, 1)

    if progress_cb:
        progress_cb(f"시퀀스 준비 중… ({len(config.clips)}개 이미지)", 5)

    writer = create_video_writer(
        output_path, width=out_w, height=out_h,
        fps=fps, crf=config.crf, use_nvenc=config.use_nvenc,
    )

    # 클립별 프레임 사전 생성
    all_clip_frames: list[list[np.ndarray]] = []
    for ci, clip in enumerate(config.clips):
        if progress_cb:
            pct = 5 + 40 * ci / len(config.clips)
            progress_cb(f"클립 {ci+1}/{len(config.clips)} 렌더링 중…", pct)

        raw = cv2.imread(clip.path, cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"이미지를 열 수 없습니다: {clip.path}")

        n_frames = int(clip.duration * fps)
        entry_frames = min(int(0.4 * fps), n_frames // 2)

        # 효과별 프레임 생성
        if clip.effect == "shortform":
            frames = _render_shortform_frames(raw, n_frames, out_w, out_h, fps)
        elif clip.effect == "kenburns":
            frames = _render_kenburns_frames(raw, n_frames, out_w, out_h, clip.preset)
        else:
            frames = _render_static_frames(raw, n_frames, out_w, out_h)

        # 등장 애니메이션 적용
        for fi in range(min(entry_frames, len(frames))):
            prog = fi / max(entry_frames - 1, 1)
            frames[fi] = _apply_entry(frames[fi], prog, clip.entry, out_w, out_h)

        # 컬러 그레이딩
        if clip.color_grade != "none":
            for fi in range(len(frames)):
                frames[fi] = apply_color_grade(frames[fi], clip.color_grade)

        # 텍스트 오버레이
        if clip.text:
            for fi in range(len(frames)):
                frames[fi] = render_text_on_frame(
                    frames[fi], clip.text,
                    position=clip.text_position,
                    font_size=clip.font_size,
                )

        all_clip_frames.append(frames)

    # 클립 연결 + 트랜지션
    if progress_cb:
        progress_cb("시퀀스 합성 중…", 50)

    written = 0

    for ci, frames in enumerate(all_clip_frames):
        is_last = (ci == len(all_clip_frames) - 1)

        if ci == 0:
            # 첫 클립: 트랜지션 없는 부분만 먼저 출력
            if is_last:
                end = len(frames)
            else:
                end = max(len(frames) - trans_frames, 0)

            for fi in range(end):
                write_frame(writer, frames[fi])
                written += 1
                if progress_cb and written % 5 == 0:
                    pct = 50 + 45 * written / total_frames
                    progress_cb(f"인코딩 중… {written}/{total_frames}", pct)

            # 트랜지션 구간의 A측 프레임 저장
            if not is_last:
                tail_a = frames[end:]
        else:
            prev_tail = tail_a
            # 트랜지션 프레임 생성
            actual_trans = min(len(prev_tail), trans_frames, len(frames))
            for ti in range(actual_trans):
                progress = ti / max(actual_trans - 1, 1)
                fa = prev_tail[ti] if ti < len(prev_tail) else prev_tail[-1]
                fb = frames[ti] if ti < len(frames) else frames[0]
                merged = apply_transition(fa, fb, config.transition, progress)
                write_frame(writer, merged)
                written += 1
                if progress_cb and written % 5 == 0:
                    pct = 50 + 45 * written / total_frames
                    progress_cb(f"인코딩 중… {written}/{total_frames}", pct)

            # 트랜지션 이후 나머지 프레임
            if is_last:
                remaining_start = actual_trans
                remaining_end = len(frames)
            else:
                remaining_start = actual_trans
                remaining_end = max(len(frames) - trans_frames, actual_trans)

            for fi in range(remaining_start, remaining_end):
                write_frame(writer, frames[fi])
                written += 1
                if progress_cb and written % 5 == 0:
                    pct = 50 + 45 * written / total_frames
                    progress_cb(f"인코딩 중… {written}/{total_frames}", pct)

            # 다음 트랜지션을 위한 꼬리
            if not is_last:
                tail_a = frames[remaining_end:]

    finalize(writer)

    if progress_cb:
        file_bytes = os.path.getsize(output_path)
        if file_bytes >= 1024 * 1024:
            size_str = f"{file_bytes / (1024*1024):.1f} MB"
        else:
            size_str = f"{file_bytes / 1024:.1f} KB"
        progress_cb(f"완료! {size_str} — {output_path}", 100)


# ─── JSON 로드 ────────────────────────────────────────────


def load_sequence_from_json(json_path: str) -> tuple[SequenceConfig, str]:
    """JSON 파일에서 시퀀스 설정을 로드한다.

    Returns:
        (SequenceConfig, output_path)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = []
    for c in data.get("clips", []):
        clips.append(ClipConfig(
            path=c.get("image", ""),
            duration=c.get("duration", 3.0),
            effect=c.get("effect", "kenburns"),
            preset=c.get("preset", "zoom_in"),
            entry=c.get("entry", "none"),
            color_grade=c.get("color_grade", "none"),
            text=c.get("text", ""),
            text_position=c.get("text_position", "bottom"),
            font_size=c.get("font_size", 48),
        ))

    res = data.get("resolution", [1920, 1080])
    config = SequenceConfig(
        clips=clips,
        transition=data.get("transition", "crossfade"),
        transition_duration=data.get("transition_duration", 0.5),
        output_w=res[0],
        output_h=res[1],
        fps=data.get("fps", 30),
        crf=data.get("crf", 18),
        use_nvenc=data.get("use_nvenc", False),
    )

    output = data.get("output", "sequence_output.mp4")
    return config, output
