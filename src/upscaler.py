"""AI 업스케일링 모듈 — Real-ESRGAN + LANCZOS4 폴백"""

import cv2
import numpy as np

_esrgan_upsampler = None


def _get_esrgan(scale: int = 4):
    """Real-ESRGAN 업샘플러를 싱글톤으로 생성한다."""
    global _esrgan_upsampler
    if _esrgan_upsampler is not None:
        return _esrgan_upsampler

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )

        # 모델 가중치 경로 (자동 다운로드)
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.1.0/RealESRGAN_x4plus.pth"
        )

        gpu_id = 0 if torch.cuda.is_available() else None
        _esrgan_upsampler = RealESRGANer(
            scale=scale,
            model_path=model_url,
            model=model,
            tile=512,      # 타일링으로 VRAM 절약
            tile_pad=10,
            pre_pad=0,
            half=True,     # FP16
            gpu_id=gpu_id,
        )
        return _esrgan_upsampler
    except Exception:
        return None


def upscale_image(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """이미지를 목표 크기 이상으로 AI 업스케일한다.

    Real-ESRGAN을 우선 사용하고, 실패 시 LANCZOS4 폴백.

    Args:
        image: BGR 또는 BGRA 이미지
        target_w, target_h: 최소 목표 크기

    Returns:
        업스케일된 이미지
    """
    h, w = image.shape[:2]
    if w >= target_w and h >= target_h:
        return image

    scale_needed = max(target_w / w, target_h / h)

    # BGRA인 경우 BGR과 알파를 분리 처리
    has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False

    if has_alpha:
        bgr = image[:, :, :3]
        alpha_ch = image[:, :, 3]
    else:
        bgr = image
        alpha_ch = None

    # Real-ESRGAN 시도 (BGR only)
    upscaled_bgr = _try_esrgan_upscale(bgr, scale_needed)
    if upscaled_bgr is None:
        print("      업스케일 엔진: LANCZOS4 (폴백)")
        new_w = int(w * scale_needed * 1.05)
        new_h = int(h * scale_needed * 1.05)
        upscaled_bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        print("      업스케일 엔진: Real-ESRGAN (AI)")

    # 알파 채널 리사이즈 (있으면)
    if alpha_ch is not None:
        uh, uw = upscaled_bgr.shape[:2]
        alpha_up = cv2.resize(alpha_ch, (uw, uh), interpolation=cv2.INTER_LANCZOS4)
        result = np.dstack([upscaled_bgr, alpha_up])
        return result

    return upscaled_bgr


def _try_esrgan_upscale(bgr: np.ndarray, scale_needed: float) -> np.ndarray | None:
    """Real-ESRGAN으로 업스케일을 시도한다."""
    upsampler = _get_esrgan()
    if upsampler is None:
        return None

    try:
        # Real-ESRGAN outscale: 최대 3x (원본 질감 보존 ↔ 해상도 확보 균형)
        outscale = min(max(scale_needed * 1.05, 1.5), 3.0)
        output, _ = upsampler.enhance(bgr, outscale=outscale)
        return output
    except Exception:
        return None
