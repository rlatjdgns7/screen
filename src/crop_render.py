"""원본 크롭 렌더러 — 원본 해상도 우선 크롭 + 필요 시 AI 업스케일

처리 흐름:
1. 원본 이미지의 해상도와 줌 범위를 비교
2. 원본이 충분히 크면 → 그대로 사용 (최고 화질, AI 없음)
3. 원본이 부족하면 → AI 업스케일로 캔버스 확보
4. 렌더링은 항상 다운스케일만 → 화질 보존
"""

import cv2
import numpy as np

from src.motion import KenBurnsConfig, EASING_FUNCTIONS, ease_in_out_sine, PRESETS
from src.renderer import crop_and_resize
from src.video import create_video_writer, write_frame, finalize


def get_crop_preset(
    name: str,
    img_w: int,
    img_h: int,
    out_w: int,
    out_h: int,
    duration: float = 5.0,
    fps: int = 30,
) -> KenBurnsConfig:
    """원본 크롭용 프리셋을 생성한다.

    extend_ratio 보정 없이 원본 이미지 좌표계를 직접 사용한다.
    단일 레이어(전경/배경 분리 없음)로 동작한다.
    """
    if name not in PRESETS:
        raise ValueError(f"알 수 없는 프리셋: {name}. 사용 가능: {list(PRESETS.keys())}")

    config = PRESETS[name](bbox_cx=0.5, bbox_cy=0.5)
    config.duration = duration
    config.fps = fps

    # 단일 레이어 모드: speed 배율 1.0
    config.bg_speed = 1.0
    config.fg_speed = 1.0

    return config


def _calc_needed_canvas(
    config: KenBurnsConfig,
    out_w: int,
    out_h: int,
) -> tuple[int, int]:
    """프리셋의 줌 범위를 기반으로 필요한 최소 캔버스 크기를 계산한다."""
    # zoom=1.0일 때 크롭 = 캔버스 전체, zoom=2.0이면 크롭 = 캔버스 절반
    # 최소 줌(가장 넓은 뷰)일 때 필요한 캔버스 = out / min_zoom ... 이 아니라
    # 캔버스에서 out_w/zoom, out_h/zoom 만큼 크롭하므로
    # 캔버스 >= out_w/min_zoom 이어야 함
    min_zoom = min(config.start_zoom, config.end_zoom)
    # 여유 5%
    need_w = int(out_w / min_zoom * 1.05)
    need_h = int(out_h / min_zoom * 1.05)
    return need_w, need_h


def get_crop_frame_params(
    config: KenBurnsConfig, frame_idx: int, total_frames: int
) -> tuple[float, float, float]:
    """프레임별 (cx, cy, zoom) 파라미터를 반환한다. (단일 레이어)"""
    t = frame_idx / max(total_frames - 1, 1)
    ease_fn = EASING_FUNCTIONS.get(config.easing, ease_in_out_sine)
    et = ease_fn(t)

    cx = config.start_cx + (config.end_cx - config.start_cx) * et
    cy = config.start_cy + (config.end_cy - config.start_cy) * et
    zoom = config.start_zoom + (config.end_zoom - config.start_zoom) * et

    return cx, cy, zoom


def render_crop_video(
    input_path: str,
    output_path: str,
    preset: str = "zoom_in",
    out_w: int = 1920,
    out_h: int = 1080,
    duration: float = 5.0,
    fps: int = 30,
    crf: int = 12,
    use_nvenc: bool = False,
    progress_cb=None,
):
    """원본 크롭 방식으로 영상을 렌더링한다.

    원본이 충분히 크면 AI 없이 바로 크롭하고,
    부족하면 AI 업스케일 후 크롭한다.
    렌더링은 항상 다운스케일만 발생하여 화질을 최대 보존한다.
    """
    # 원본 이미지 로드 (리사이즈 없음)
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {input_path}")

    img_h, img_w = img.shape[:2]

    # 모션 설정
    config = get_crop_preset(preset, img_w, img_h, out_w, out_h, duration, fps)
    total_frames = int(duration * fps)

    # 필요한 캔버스 크기 계산
    need_w, need_h = _calc_needed_canvas(config, out_w, out_h)

    if img_w >= need_w and img_h >= need_h:
        # 원본이 충분히 큼 → AI 없이 바로 크롭
        canvas = img
        if progress_cb:
            progress_cb(
                f"원본 크롭 ({img_w}x{img_h}) — AI 불필요, 최고 화질", 5
            )
    else:
        # 원본이 부족 → AI 업스케일로 캔버스 확보
        if progress_cb:
            progress_cb(
                f"AI 업스케일 중… ({img_w}x{img_h} → {need_w}x{need_h}+)", 5
            )
        from src.upscaler import upscale_image

        canvas = upscale_image(img, need_w, need_h)
        canvas_h, canvas_w = canvas.shape[:2]
        if progress_cb:
            progress_cb(
                f"업스케일 완료 ({canvas_w}x{canvas_h}) → 크롭 렌더링", 20
            )

    # 렌더링
    writer = create_video_writer(
        output_path, width=out_w, height=out_h,
        fps=fps, crf=crf, use_nvenc=use_nvenc,
    )

    render_start = 25 if img_w < need_w or img_h < need_h else 5

    for i in range(total_frames):
        cx, cy, zoom = get_crop_frame_params(config, i, total_frames)
        frame = crop_and_resize(canvas, cx, cy, zoom, out_w, out_h)
        write_frame(writer, frame)

        if progress_cb and (i % 5 == 0 or i == total_frames - 1):
            pct = render_start + (95 - render_start) * (i + 1) / total_frames
            progress_cb(f"렌더링 {i+1}/{total_frames}", pct)

    finalize(writer)

    if progress_cb:
        progress_cb("완료", 100)
