"""시네마틱 켄번스 효과 프로그램 — CLI 엔트리포인트

이미지 1장 → AI 프리업스케일 → HD 세그멘테이션 → HD 인페인팅 → 패럴랙스 렌더링 → MP4
"""

import argparse
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from src.segmenter import segment_image
from src.upscaler import upscale_image
from src.inpainter import inpaint_background
from src.motion import get_preset, get_frame_params, PRESETS
from src.renderer import render_frame
from src.video import create_video_writer, write_frame, finalize


def parse_args():
    parser = argparse.ArgumentParser(
        description="시네마틱 켄번스 효과 — 이미지를 패럴랙스 영상으로 변환",
    )
    parser.add_argument("--input", "-i", required=True, help="입력 이미지 경로")
    parser.add_argument("--output", "-o", default="output.mp4", help="출력 MP4 경로")
    parser.add_argument("--preset", "-p", default="zoom_in", choices=list(PRESETS.keys()))
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="영상 길이 (초)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=15, help="CRF 품질 (기본: 15)")
    parser.add_argument("--nvenc", action="store_true", help="NVENC GPU 인코딩")
    parser.add_argument("--model", default=None, help="세그멘테이션 모델 (기본: 자동)")
    parser.add_argument("--feather", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    extend_ratio = 0.20

    # ─── 1. AI 프리업스케일 (원본 → 출력 해상도 이상) ─────
    raw = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {args.input}")
    raw_h, raw_w = raw.shape[:2]
    print(f"[1/6] 원본 크기: {raw_w}x{raw_h}")

    if raw_w < args.width or raw_h < args.height:
        print(f"      AI 프리업스케일 중... (목표: {args.width}x{args.height} 이상)")
        original_hd = upscale_image(raw, args.width, args.height)
        hd_h, hd_w = original_hd.shape[:2]
        print(f"      프리업스케일 결과: {hd_w}x{hd_h}")
    else:
        original_hd = raw
        hd_h, hd_w = raw_h, raw_w
        print(f"      원본이 출력 해상도 이상 — 프리업스케일 생략")

    # ─── 2. HD 세그멘테이션 (고해상도에서 정밀한 마스크) ──
    print(f"[2/6] HD 세그멘테이션 중...")
    seg = segment_image(original_hd, model_name=args.model, feather_radius=args.feather)
    print(f"      이미지 크기: {hd_w}x{hd_h}")
    print(f"      대상 바운딩 박스: {seg.bbox}")

    # ─── 3. 모션 경로 설정 ─────────────────────────────────
    print(f"[3/6] 모션 프리셋: {args.preset}")
    config = get_preset(
        args.preset, bbox=seg.bbox, img_w=hd_w, img_h=hd_h,
        duration=args.duration, fps=args.fps, extend_ratio=extend_ratio,
    )
    total_frames = int(config.duration * config.fps)
    max_zoom = max(config.start_zoom, config.end_zoom)
    print(f"      총 프레임: {total_frames}, 최대줌: {max_zoom:.2f}x")

    # ─── 4. HD 인페인팅 + 줌 여유 업스케일 ────────────────
    print(f"[4/6] HD 배경 인페인팅 중...")
    bg_inpainted = inpaint_background(seg.original, seg.alpha, extend_ratio=extend_ratio)
    bg_inp_h, bg_inp_w = bg_inpainted.shape[:2]
    print(f"      인페인팅 결과: {bg_inp_w}x{bg_inp_h}")

    # 줌 여유분 확인: 인페인팅 캔버스가 충분하면 추가 업스케일 불필요
    need_w = int(args.width * max_zoom * 1.05)
    need_h = int(args.height * max_zoom * 1.05)

    if bg_inp_w >= need_w and bg_inp_h >= need_h:
        print(f"      캔버스 충분 ({bg_inp_w}x{bg_inp_h} >= {need_w}x{need_h}) — 추가 업스케일 생략")
        bg_image = bg_inpainted
        fg_up = seg.foreground
    else:
        # LANCZOS4로 부족분만 보간 (이중 AI 방지)
        scale = max(need_w / bg_inp_w, need_h / bg_inp_h)
        new_w = int(bg_inp_w * scale)
        new_h = int(bg_inp_h * scale)
        print(f"      줌 여유 보간 중... (LANCZOS4 {scale:.2f}x → {new_w}x{new_h})")
        bg_image = cv2.resize(bg_inpainted, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        fg_w, fg_h = seg.foreground.shape[1], seg.foreground.shape[0]
        fg_scale = max(new_w / fg_w, new_h / fg_h) if fg_w < new_w or fg_h < new_h else 1.0
        if fg_scale > 1.0:
            fg_new_w = int(fg_w * fg_scale)
            fg_new_h = int(fg_h * fg_scale)
            fg_up = cv2.resize(seg.foreground, (fg_new_w, fg_new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            fg_up = seg.foreground
        print(f"      배경: {bg_image.shape[1]}x{bg_image.shape[0]}")
        print(f"      대상: {fg_up.shape[1]}x{fg_up.shape[0]}")

    # ─── 5. 대상/배경 캔버스 크기 맞춤 ───────────────────
    bg_h, bg_w = bg_image.shape[:2]
    fg_h, fg_w = fg_up.shape[:2]
    if (fg_w, fg_h) != (bg_w, bg_h):
        dw = bg_w - fg_w
        dh = bg_h - fg_h
        if dw > 0 or dh > 0:
            pad_l, pad_r = max(dw // 2, 0), max(dw - dw // 2, 0)
            pad_t, pad_b = max(dh // 2, 0), max(dh - dh // 2, 0)
            fg_up = cv2.copyMakeBorder(
                fg_up, pad_t, pad_b, pad_l, pad_r,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0),
            )
        else:
            fg_up = cv2.resize(fg_up, (bg_w, bg_h), interpolation=cv2.INTER_LANCZOS4)

    # ─── 6. 렌더링 + 인코딩 ───────────────────────────────
    print(f"[5/6] 렌더링 중... ({args.width}x{args.height})")
    writer = create_video_writer(
        args.output, width=args.width, height=args.height,
        fps=args.fps, crf=args.crf, use_nvenc=args.nvenc,
    )

    try:
        for i in tqdm(range(total_frames), desc="      프레임", unit="f"):
            bg_cx, bg_cy, bg_zoom, fg_cx, fg_cy, fg_zoom = get_frame_params(
                config, i, total_frames
            )
            frame = render_frame(
                bg_image, fg_up,
                bg_cx, bg_cy, bg_zoom, fg_cx, fg_cy, fg_zoom,
                out_w=args.width, out_h=args.height,
            )
            write_frame(writer, frame)
    except Exception as e:
        finalize(writer)
        raise e

    finalize(writer)

    elapsed = time.time() - t0
    file_bytes = os.path.getsize(args.output)
    size_str = f"{file_bytes / (1024*1024):.1f} MB" if file_bytes >= 1024*1024 else f"{file_bytes / 1024:.1f} KB"
    print(f"[6/6] 완료!")
    print(f"      출력: {args.output}")
    print(f"      크기: {size_str}")
    print(f"      소요 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
