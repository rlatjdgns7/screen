# 시네마틱 켄번스 효과 자동 적용 프로그램 설계 보고서

## 1. 프로젝트 개요

### 1.1 목표
이미지를 입력하면 **AI 기반 세그멘테이션**으로 대상(Subject)과 배경(Background)을 자동 분리한 뒤,
분리된 레이어에 **서로 다른 속도의 줌/패닝**을 적용하여 **시네마틱 패럴랙스 켄번스 효과** 영상을 생성하는 프로그램.

### 1.2 켄번스 효과란?
- 다큐멘터리 감독 Ken Burns가 즐겨 사용한 기법
- 정지 이미지에 **천천히 줌인/줌아웃 + 패닝(이동)**을 적용하여 영상처럼 보이게 하는 연출
- **패럴랙스 켄번스**: 전경/배경 레이어를 분리하여 서로 다른 속도로 움직이게 함으로써 **3D 깊이감** 연출

### 1.3 최종 결과물
- 입력: 정지 이미지 1장 (JPG/PNG)
- 출력: 5~15초 분량의 MP4 영상 (1080p/4K, 24~30fps)

---

## 2. 핵심 기술 구성

```
┌─────────────────────────────────────────────────────────┐
│                    전체 파이프라인                         │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐   │
│  │  입력     │──▶│ 세그멘테이션  │──▶│ 인페인팅        │   │
│  │  이미지   │   │ (대상 분리)   │   │ (배경 복원)     │   │
│  └──────────┘   └──────────────┘   └───────┬────────┘   │
│                                            │            │
│                                            ▼            │
│                 ┌──────────────┐   ┌────────────────┐   │
│                 │  영상 렌더링  │◀──│ 켄번스 모션     │   │
│                 │  (MP4 출력)  │   │ 경로 생성       │   │
│                 └──────────────┘   └────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.1 단계별 핵심 기술

| 단계 | 기술 | 역할 |
|------|------|------|
| **1. 세그멘테이션** | SAM2 / RMBG-2.0 / U²-Net | 대상과 배경을 픽셀 단위로 분리 |
| **2. 인페인팅** | LaMa / Stable Diffusion Inpaint | 대상 제거 후 배경 빈 영역 복원 |
| **3. 모션 생성** | 커스텀 모션 엔진 | 줌/패닝 경로 계산 (이징 함수 적용) |
| **4. 패럴랙스 합성** | OpenCV / Pillow | 레이어별 다른 속도로 변환 후 합성 |
| **5. 영상 인코딩** | FFmpeg / MoviePy | 프레임 시퀀스를 MP4로 인코딩 |

---

## 3. 기술 스택 추천

### 3.1 Python 기반 (추천)

```
Python 3.10+
├── 세그멘테이션
│   ├── rembg (ONNX 기반, 설치 간편, CPU/GPU 모두 지원)
│   ├── segment-anything-2 (Meta SAM2, 최고 품질)
│   └── transformers + BRIA RMBG-2.0 (HuggingFace)
├── 인페인팅
│   ├── lama-cleaner (LaMa 모델, 로컬 실행)
│   ├── cv2.inpaint() (OpenCV 내장, 가볍지만 품질 제한)
│   └── diffusers (Stable Diffusion Inpaint)
├── 이미지 처리
│   ├── OpenCV (cv2) — 어파인 변환, 리사이즈, 블렌딩
│   ├── Pillow (PIL) — 이미지 I/O, 알파 합성
│   └── NumPy — 행렬 연산
├── 영상 생성
│   ├── moviepy — 프레임 → 영상 변환
│   └── ffmpeg-python — FFmpeg 파이프라인
└── GUI (선택)
    ├── Gradio — 웹 기반 데모 UI (가장 빠른 프로토타이핑)
    ├── Streamlit — 웹 대시보드
    └── PyQt6 / PySide6 — 데스크톱 앱
```

### 3.2 권장 환경

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 이상 |
| **RAM** | 8GB | 16GB 이상 |
| **GPU** | 없어도 가능 (CPU 모드) | NVIDIA RTX 3060+ (CUDA) |
| **저장소** | 2GB (모델 포함) | 5GB+ (고품질 모델) |
| **Python** | 3.10 | 3.10 ~ 3.12 |

---

## 4. 상세 설계

### 4.1 단계 1: 이미지 세그멘테이션 (대상/배경 분리)

#### 방법 A: `rembg` (가장 간편)

```python
from rembg import remove
from PIL import Image
import numpy as np

def separate_subject_background(image_path):
    """대상과 배경을 분리하여 각각의 이미지와 마스크를 반환"""
    img = Image.open(image_path).convert("RGBA")

    # 대상 추출 (배경 투명)
    subject = remove(img)

    # 알파 채널에서 마스크 추출
    alpha = np.array(subject)[:, :, 3]
    mask = (alpha > 128).astype(np.uint8) * 255

    return {
        "original": img,
        "subject": subject,           # 대상 (투명 배경)
        "mask": mask,                  # 이진 마스크
        "subject_bbox": get_bbox(mask) # 대상 바운딩 박스
    }

def get_bbox(mask):
    """마스크에서 대상의 바운딩 박스 계산"""
    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (x_min, y_min, x_max, y_max)
```

#### 방법 B: `SAM2` (최고 품질, GPU 권장)

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def segment_with_sam2(image_path):
    """SAM2를 사용한 고품질 세그멘테이션"""
    checkpoint = "sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    image = cv2.imread(image_path)
    predictor.set_image(image)

    # 이미지 중앙을 프롬프트로 자동 지정 (또는 얼굴 감지 활용)
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # 가장 높은 점수의 마스크 선택
    best_mask = masks[np.argmax(scores)]
    return best_mask
```

#### 대상 자동 감지 전략

```
우선순위:
1. 얼굴 감지 (dlib / MediaPipe) → 인물 사진
2. 전경 검출 (GrabCut 초기화 + SAM 정밀화) → 일반 사진
3. 현저성 맵 (Saliency Map) → 시각적으로 두드러진 영역
4. 이미지 중앙 기본값 → 폴백
```

---

### 4.2 단계 2: 배경 인페인팅 (빈 영역 복원)

대상을 분리하면 배경에 **빈 구멍**이 생깁니다.
켄번스 효과로 카메라가 이동할 때 이 빈 영역이 드러나므로, 반드시 **인페인팅**으로 복원해야 합니다.

#### 방법 A: OpenCV 내장 인페인팅 (빠르고 가벼움)

```python
import cv2

def inpaint_background_cv2(image, mask, method="telea"):
    """OpenCV 인페인팅으로 배경 복원"""
    # 마스크를 약간 팽창시켜 경계 아티팩트 제거
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    if method == "telea":
        result = cv2.inpaint(image, dilated_mask, 10, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(image, dilated_mask, 10, cv2.INPAINT_NS)

    return result
```

#### 방법 B: LaMa (고품질, GPU 권장)

```python
# lama-cleaner 라이브러리 사용
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config

def inpaint_background_lama(image, mask):
    """LaMa 모델로 고품질 인페인팅"""
    model = ModelManager("lama", device="cuda")
    config = Config(
        ldm_steps=25,
        hd_strategy="Resize",
        hd_strategy_resize_limit=1080
    )
    result = model(image, mask, config)
    return result
```

#### 인페인팅 + 여유 영역 확보

```
켄번스 효과 시 카메라 이동 범위를 고려하여
배경 이미지를 원본보다 10~20% 더 크게 확장(outpaint) 권장

┌────────────────────────────┐
│   확장 영역 (outpaint)      │
│  ┌──────────────────────┐  │
│  │                      │  │
│  │   원본 배경 영역       │  │
│  │   (인페인팅 완료)      │  │
│  │                      │  │
│  └──────────────────────┘  │
│   확장 영역 (outpaint)      │
└────────────────────────────┘
```

```python
def extend_background(bg_image, extend_ratio=0.15):
    """배경 이미지를 상하좌우로 확장"""
    h, w = bg_image.shape[:2]
    pad_h = int(h * extend_ratio)
    pad_w = int(w * extend_ratio)

    # 반사 패딩으로 자연스러운 확장
    extended = cv2.copyMakeBorder(
        bg_image, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_REFLECT_101
    )
    return extended, (pad_w, pad_h)  # 오프셋 반환
```

---

### 4.3 단계 3: 켄번스 모션 경로 생성

#### 모션 파라미터 정의

```python
from dataclasses import dataclass
from enum import Enum

class MotionPreset(Enum):
    ZOOM_IN_SUBJECT = "subject_zoom_in"      # 대상으로 줌인
    ZOOM_OUT_REVEAL = "zoom_out_reveal"       # 줌아웃하며 전체 공개
    PAN_LEFT_TO_RIGHT = "pan_lr"             # 좌→우 패닝
    PAN_RIGHT_TO_LEFT = "pan_rl"             # 우→좌 패닝
    DRAMATIC_PUSH_IN = "dramatic_push"       # 극적인 전진
    SLOW_DRIFT = "slow_drift"               # 느린 표류
    CIRCULAR_ORBIT = "circular_orbit"        # 원형 궤도

@dataclass
class KenBurnsConfig:
    duration: float = 8.0          # 영상 길이 (초)
    fps: int = 30                  # 프레임 레이트
    output_width: int = 1920       # 출력 해상도
    output_height: int = 1080

    # 줌 설정
    zoom_start: float = 1.0        # 시작 줌 레벨 (1.0 = 원본)
    zoom_end: float = 1.3          # 종료 줌 레벨

    # 패닝 설정 (정규화 좌표, 0~1)
    pan_start: tuple = (0.5, 0.5)  # 시작 중심점
    pan_end: tuple = (0.5, 0.5)    # 종료 중심점

    # 패럴랙스 설정
    bg_speed_ratio: float = 0.4    # 배경 이동 속도 비율 (대상 대비)
    fg_speed_ratio: float = 1.0    # 대상 이동 속도 비율

    # 이징 함수
    easing: str = "ease_in_out"    # 가감속 곡선
    motion_preset: MotionPreset = MotionPreset.ZOOM_IN_SUBJECT
```

#### 이징 함수 (부드러운 가감속)

```python
import math

def ease_in_out_cubic(t):
    """부드러운 시작과 끝 (시네마틱 느낌의 핵심)"""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def ease_in_out_sine(t):
    """사인 곡선 기반 가감속"""
    return -(math.cos(math.pi * t) - 1) / 2

def ease_out_expo(t):
    """지수 감속 (극적인 시작 후 안정)"""
    return 1 if t == 1 else 1 - pow(2, -10 * t)

EASING_FUNCTIONS = {
    "linear": lambda t: t,
    "ease_in_out": ease_in_out_cubic,
    "ease_in_out_sine": ease_in_out_sine,
    "ease_out_expo": ease_out_expo,
}
```

#### 모션 경로 계산기

```python
def generate_motion_path(config: KenBurnsConfig):
    """프레임별 줌/패닝 파라미터 생성"""
    total_frames = int(config.duration * config.fps)
    easing_fn = EASING_FUNCTIONS[config.easing]
    frames = []

    for i in range(total_frames):
        t = i / (total_frames - 1)  # 0.0 ~ 1.0
        eased_t = easing_fn(t)

        # 줌 보간
        zoom = config.zoom_start + (config.zoom_end - config.zoom_start) * eased_t

        # 패닝 보간
        pan_x = config.pan_start[0] + (config.pan_end[0] - config.pan_start[0]) * eased_t
        pan_y = config.pan_start[1] + (config.pan_end[1] - config.pan_start[1]) * eased_t

        frames.append({
            "frame": i,
            "t": eased_t,
            "zoom": zoom,
            "pan": (pan_x, pan_y),
            "bg_zoom": 1.0 + (zoom - 1.0) * config.bg_speed_ratio,
            "bg_pan": (
                0.5 + (pan_x - 0.5) * config.bg_speed_ratio,
                0.5 + (pan_y - 0.5) * config.bg_speed_ratio
            )
        })

    return frames
```

#### 프리셋 자동 설정

```python
def auto_configure(image_size, subject_bbox, preset: MotionPreset):
    """대상 위치에 따라 모션 자동 설정"""
    w, h = image_size
    sx1, sy1, sx2, sy2 = subject_bbox

    # 대상 중심 (정규화)
    subject_cx = ((sx1 + sx2) / 2) / w
    subject_cy = ((sy1 + sy2) / 2) / h

    # 대상 크기 비율
    subject_ratio = ((sx2 - sx1) * (sy2 - sy1)) / (w * h)

    if preset == MotionPreset.ZOOM_IN_SUBJECT:
        return KenBurnsConfig(
            zoom_start=1.0,
            zoom_end=1.5 if subject_ratio < 0.3 else 1.25,
            pan_start=(0.5, 0.5),
            pan_end=(subject_cx, subject_cy),
            easing="ease_in_out",
            bg_speed_ratio=0.3,
        )

    elif preset == MotionPreset.ZOOM_OUT_REVEAL:
        return KenBurnsConfig(
            zoom_start=1.5,
            zoom_end=1.0,
            pan_start=(subject_cx, subject_cy),
            pan_end=(0.5, 0.5),
            easing="ease_in_out",
            bg_speed_ratio=0.3,
        )

    elif preset == MotionPreset.DRAMATIC_PUSH_IN:
        return KenBurnsConfig(
            zoom_start=1.0,
            zoom_end=2.0,
            pan_start=(0.5, 0.5),
            pan_end=(subject_cx, subject_cy * 0.9),  # 약간 위로
            easing="ease_out_expo",
            bg_speed_ratio=0.2,
            duration=6.0,
        )

    elif preset == MotionPreset.SLOW_DRIFT:
        # 대상 위치 반대 방향으로 약간 이동
        drift_x = 0.5 + (0.5 - subject_cx) * 0.15
        drift_y = 0.5 + (0.5 - subject_cy) * 0.15
        return KenBurnsConfig(
            zoom_start=1.1,
            zoom_end=1.15,
            pan_start=(0.5, 0.5),
            pan_end=(drift_x, drift_y),
            easing="ease_in_out_sine",
            bg_speed_ratio=0.5,
            duration=10.0,
        )

    # ... 기타 프리셋
```

---

### 4.4 단계 4: 패럴랙스 합성 렌더링

#### 핵심 렌더링 엔진

```python
import cv2
import numpy as np

class ParallaxRenderer:
    def __init__(self, subject_rgba, background, config: KenBurnsConfig):
        """
        subject_rgba: 대상 이미지 (RGBA, 투명 배경)
        background: 인페인팅된 배경 이미지 (RGB)
        config: 켄번스 설정
        """
        self.subject = subject_rgba
        self.background = background
        self.config = config
        self.output_size = (config.output_width, config.output_height)

    def crop_and_zoom(self, image, zoom, center, output_size):
        """줌 레벨과 중심점에 따라 이미지를 크롭"""
        h, w = image.shape[:2]
        out_w, out_h = output_size

        # 줌 적용된 크롭 영역 계산
        crop_w = out_w / zoom
        crop_h = out_h / zoom

        cx = center[0] * w
        cy = center[1] * h

        x1 = int(max(0, cx - crop_w / 2))
        y1 = int(max(0, cy - crop_h / 2))
        x2 = int(min(w, x1 + crop_w))
        y2 = int(min(h, y1 + crop_h))

        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LANCZOS4)
        return resized

    def alpha_composite(self, background, foreground_rgba):
        """알파 블렌딩으로 전경과 배경 합성"""
        fg = foreground_rgba[:, :, :3].astype(float)
        alpha = foreground_rgba[:, :, 3:4].astype(float) / 255.0
        bg = background.astype(float)

        # 알파 블렌딩
        composite = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        return composite

    def render_frame(self, frame_params):
        """단일 프레임 렌더링"""
        # 배경 레이어 (느리게 이동)
        bg_frame = self.crop_and_zoom(
            self.background,
            frame_params["bg_zoom"],
            frame_params["bg_pan"],
            self.output_size
        )

        # 대상 레이어 (빠르게 이동)
        fg_frame = self.crop_and_zoom(
            self.subject,
            frame_params["zoom"],
            frame_params["pan"],
            self.output_size
        )

        # 합성
        composite = self.alpha_composite(bg_frame, fg_frame)
        return composite

    def render_all_frames(self, motion_path):
        """전체 프레임 렌더링"""
        frames = []
        for i, params in enumerate(motion_path):
            frame = self.render_frame(params)
            frames.append(frame)

            if i % 30 == 0:
                progress = (i / len(motion_path)) * 100
                print(f"렌더링 진행: {progress:.1f}%")

        return frames
```

---

### 4.5 단계 5: 영상 출력

```python
from moviepy.editor import ImageSequenceClip
import subprocess

def export_video_moviepy(frames, output_path, fps=30):
    """MoviePy를 사용한 영상 출력"""
    # BGR → RGB 변환
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    clip = ImageSequenceClip(rgb_frames, fps=fps)
    clip.write_videofile(
        output_path,
        codec="libx264",
        bitrate="8000k",
        audio=False,
        preset="slow",        # 고품질 인코딩
        ffmpeg_params=["-crf", "18"]  # 높은 품질 (낮을수록 고품질)
    )

def export_video_ffmpeg(frames, output_path, fps=30):
    """FFmpeg 파이프를 사용한 고성능 영상 출력"""
    h, w = frames[0].shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
```

---

## 5. 전체 통합 파이프라인

```python
class CinematicKenBurns:
    """시네마틱 켄번스 효과 메인 클래스"""

    def __init__(self, config=None):
        self.config = config or KenBurnsConfig()

    def process(self, image_path, output_path="output.mp4",
                preset=MotionPreset.ZOOM_IN_SUBJECT):
        """이미지 → 시네마틱 영상 전체 파이프라인"""

        print("[1/5] 이미지 로드 중...")
        original = cv2.imread(image_path)
        h, w = original.shape[:2]

        print("[2/5] 대상/배경 분리 중...")
        seg_result = separate_subject_background(image_path)
        subject_rgba = np.array(seg_result["subject"])
        mask = seg_result["mask"]
        bbox = seg_result["subject_bbox"]

        print("[3/5] 배경 인페인팅 중...")
        background = inpaint_background_cv2(original, mask)
        background = extend_background(background, extend_ratio=0.15)[0]
        # subject_rgba도 동일 비율로 확장 (투명 패딩)
        subject_rgba = extend_subject(subject_rgba, extend_ratio=0.15)

        print("[4/5] 모션 경로 생성 중...")
        self.config = auto_configure((w, h), bbox, preset)
        motion_path = generate_motion_path(self.config)

        print("[5/5] 영상 렌더링 중...")
        renderer = ParallaxRenderer(subject_rgba, background, self.config)
        frames = renderer.render_all_frames(motion_path)

        export_video_moviepy(frames, output_path, self.config.fps)
        print(f"완료! 출력 파일: {output_path}")


# 사용 예시
if __name__ == "__main__":
    app = CinematicKenBurns()
    app.process(
        image_path="photo.jpg",
        output_path="cinematic_output.mp4",
        preset=MotionPreset.DRAMATIC_PUSH_IN
    )
```

---

## 6. GUI 인터페이스 (Gradio)

```python
import gradio as gr

def process_image(image, preset_name, duration, zoom_intensity):
    """Gradio 인터페이스 핸들러"""
    preset_map = {
        "대상으로 줌인": MotionPreset.ZOOM_IN_SUBJECT,
        "줌아웃 전체 공개": MotionPreset.ZOOM_OUT_REVEAL,
        "극적 전진": MotionPreset.DRAMATIC_PUSH_IN,
        "좌→우 패닝": MotionPreset.PAN_LEFT_TO_RIGHT,
        "느린 표류": MotionPreset.SLOW_DRIFT,
    }

    # 임시 파일로 저장 후 처리
    temp_input = "temp_input.png"
    temp_output = "temp_output.mp4"
    image.save(temp_input)

    config = KenBurnsConfig(duration=duration)
    app = CinematicKenBurns(config)
    app.process(temp_input, temp_output, preset_map[preset_name])

    return temp_output

# Gradio UI 구성
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="이미지 업로드"),
        gr.Dropdown(
            choices=["대상으로 줌인", "줌아웃 전체 공개",
                     "극적 전진", "좌→우 패닝", "느린 표류"],
            value="대상으로 줌인",
            label="모션 프리셋"
        ),
        gr.Slider(3, 15, value=8, step=1, label="영상 길이 (초)"),
        gr.Slider(1.1, 2.0, value=1.3, step=0.05, label="줌 강도"),
    ],
    outputs=gr.Video(label="결과 영상"),
    title="🎬 시네마틱 켄번스 효과 생성기",
    description="이미지를 업로드하면 대상을 자동 분리하고 시네마틱 패럴랙스 켄번스 효과를 적용합니다."
)

demo.launch()
```

---

## 7. 프로젝트 디렉토리 구조

```
cinematic-ken-burns/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── default_config.yaml       # 기본 설정값
├── src/
│   ├── __init__.py
│   ├── main.py                   # 메인 엔트리포인트
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── rembg_segmenter.py    # rembg 기반 분리
│   │   ├── sam2_segmenter.py     # SAM2 기반 분리
│   │   └── auto_detect.py        # 대상 자동 감지
│   ├── inpainting/
│   │   ├── __init__.py
│   │   ├── cv2_inpainter.py      # OpenCV 인페인팅
│   │   ├── lama_inpainter.py     # LaMa 인페인팅
│   │   └── background_extend.py  # 배경 확장
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── easing.py             # 이징 함수 모음
│   │   ├── path_generator.py     # 모션 경로 생성
│   │   ├── presets.py            # 모션 프리셋 정의
│   │   └── auto_config.py        # 자동 모션 설정
│   ├── renderer/
│   │   ├── __init__.py
│   │   ├── parallax.py           # 패럴랙스 합성
│   │   ├── compositor.py         # 레이어 합성
│   │   └── video_export.py       # 영상 인코딩
│   └── ui/
│       ├── __init__.py
│       └── gradio_app.py         # Gradio 웹 인터페이스
├── tests/
│   ├── test_segmentation.py
│   ├── test_motion.py
│   └── test_renderer.py
└── examples/
    ├── sample_input.jpg
    └── sample_output.mp4
```

---

## 8. 의존성 (requirements.txt)

```txt
# 핵심
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0

# 세그멘테이션
rembg>=2.0.50
onnxruntime>=1.16.0           # CPU용
# onnxruntime-gpu>=1.16.0     # GPU용 (CUDA 필요)

# 인페인팅 (선택 - 고품질 원할 경우)
# lama-cleaner>=1.2.0

# 영상 출력
moviepy>=1.0.3
ffmpeg-python>=0.2.0

# GUI
gradio>=4.0.0

# 유틸리티
tqdm>=4.65.0
pyyaml>=6.0
```

---

## 9. 설치 및 실행 가이드

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. 의존성 설치
pip install -r requirements.txt

# 3. FFmpeg 설치 (영상 인코딩 필수)
# Windows: choco install ffmpeg 또는 공식 사이트에서 다운로드
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# 4. 실행
python src/main.py --input photo.jpg --output result.mp4 --preset zoom_in

# 5. 웹 UI 실행
python src/ui/gradio_app.py
# 브라우저에서 http://localhost:7860 접속
```

---

## 10. 품질 향상 팁

### 10.1 세그멘테이션 품질

| 문제 | 해결 방법 |
|------|----------|
| 머리카락/털 경계가 거칠다 | SAM2 사용 + 마스크 페더링(Gaussian Blur) 적용 |
| 대상이 여러 개다 | SAM2의 multi-point prompt 사용 |
| 반투명 요소 (유리, 물) | 알파 매팅 모델 (ViTMatte) 병행 |

### 10.2 인페인팅 품질

| 문제 | 해결 방법 |
|------|----------|
| 대상 영역이 넓어 복원이 부자연스럽다 | LaMa 또는 Stable Diffusion Inpaint 사용 |
| 경계에 아티팩트가 보인다 | 마스크 팽창(dilate) 범위를 15~25px로 증가 |
| 패턴/텍스처 연속성 부족 | 다중 패스 인페인팅 (점진적 확대) |

### 10.3 모션 품질

| 문제 | 해결 방법 |
|------|----------|
| 움직임이 뻣뻣하다 | ease_in_out_cubic 이징 사용, 가속/감속 강화 |
| 패럴랙스가 부자연스럽다 | bg_speed_ratio를 0.2~0.4로 조정 |
| 영상 시작/끝이 급하다 | 시작/끝 1초간 페이드 인/아웃 추가 |

### 10.4 영상 품질

```python
# 고품질 출력 설정
ffmpeg_params = [
    "-crf", "16",           # 높은 품질 (15~20 권장)
    "-preset", "slow",      # 인코딩 품질 우선
    "-profile:v", "high",   # H.264 High Profile
    "-level", "4.1",
    "-movflags", "+faststart"  # 웹 스트리밍 최적화
]
```

---

## 11. 고급 확장 기능

### 11.1 깊이 맵 기반 다중 레이어

```
단순 2레이어 (기본)          →    깊이 맵 기반 N레이어 (고급)

┌─────────────┐               ┌─────────────┐
│  대상 (FG)   │               │ 레이어 1 (가장 가까움)  │ × 1.0 속도
├─────────────┤               ├─────────────┤
│  배경 (BG)   │               │ 레이어 2 (중간)       │ × 0.6 속도
└─────────────┘               ├─────────────┤
                              │ 레이어 3 (먼 배경)     │ × 0.3 속도
                              └─────────────┘

깊이 추정 모델: MiDaS, Depth Anything v2
```

```python
# Depth Anything v2를 사용한 깊이 맵 추출
from transformers import pipeline

depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large")

def get_depth_layers(image, num_layers=3):
    """깊이 맵 기반으로 이미지를 N개 레이어로 분리"""
    depth = depth_estimator(image)["depth"]
    depth_array = np.array(depth)

    # 깊이값을 N개 구간으로 분할
    thresholds = np.linspace(depth_array.min(), depth_array.max(), num_layers + 1)
    layers = []
    for i in range(num_layers):
        mask = (depth_array >= thresholds[i]) & (depth_array < thresholds[i + 1])
        layers.append(mask)

    return layers
```

### 11.2 오디오 자동 추가

```python
# 앰비언트 사운드 자동 매칭 (선택적)
def add_ambient_audio(video_path, mood="cinematic"):
    """무료 앰비언트 사운드를 영상에 추가"""
    audio_map = {
        "cinematic": "assets/audio/cinematic_ambient.mp3",
        "nature": "assets/audio/nature_ambient.mp3",
        "urban": "assets/audio/urban_ambient.mp3",
    }
    # FFmpeg로 오디오 합성
    cmd = f'ffmpeg -i {video_path} -i {audio_map[mood]} -shortest -c:v copy output_with_audio.mp4'
    subprocess.run(cmd, shell=True)
```

### 11.3 배치 처리

```python
def batch_process(image_folder, output_folder, preset=MotionPreset.ZOOM_IN_SUBJECT):
    """폴더 내 모든 이미지를 일괄 처리"""
    import glob
    images = glob.glob(f"{image_folder}/*.{jpg,png,jpeg}")
    app = CinematicKenBurns()

    for img_path in images:
        name = Path(img_path).stem
        output = f"{output_folder}/{name}_cinematic.mp4"
        app.process(img_path, output, preset)
```

---

## 12. 성능 최적화

### 12.1 처리 시간 예상 (1920×1080, 8초, 30fps = 240프레임)

| 단계 | CPU (i7) | GPU (RTX 3060) |
|------|----------|----------------|
| 세그멘테이션 (rembg) | ~3초 | ~1초 |
| 인페인팅 (OpenCV) | ~1초 | ~1초 |
| 인페인팅 (LaMa) | ~15초 | ~3초 |
| 프레임 렌더링 | ~10초 | ~3초 |
| 영상 인코딩 | ~5초 | ~2초 (NVENC) |
| **총 합계** | **~20초** | **~7초** |

### 12.2 최적화 전략

```python
# 1. 멀티프로세싱으로 프레임 병렬 렌더링
from concurrent.futures import ProcessPoolExecutor

def render_parallel(renderer, motion_path, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        frames = list(executor.map(renderer.render_frame, motion_path))
    return frames

# 2. GPU 가속 (CuPy + OpenCV CUDA)
# CuPy로 NumPy 연산을 GPU에서 실행
import cupy as cp

def crop_and_zoom_gpu(image_gpu, zoom, center, output_size):
    """GPU 가속 크롭 & 줌"""
    # CuPy 배열로 변환하여 GPU에서 처리
    # ...

# 3. 메모리 절약: 프레임을 한 번에 저장하지 않고 스트리밍
def render_and_stream(renderer, motion_path, output_path, fps):
    """프레임을 생성하면서 바로 FFmpeg로 전송"""
    h, w = renderer.output_size[1], renderer.output_size[0]
    proc = start_ffmpeg_process(w, h, fps, output_path)

    for params in motion_path:
        frame = renderer.render_frame(params)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
```

---

## 13. 유사 서비스 / 레퍼런스

| 서비스 | 특징 | 비고 |
|--------|------|------|
| **CapCut** (캡컷) | AI 기반 자동 켄번스, 모바일 최적화 | 무료, 상업용 제한 |
| **Runway ML** | Gen-2 비디오 생성, 모션 브러시 | 유료, 웹 기반 |
| **D-ID** | AI 아바타 + 켄번스 | 유료 API |
| **Immersity AI** | 2D→3D 변환 + 패럴랙스 | 깊이맵 기반 |
| **LeiaPix** | 정지 이미지 → 3D 패럴랙스 | 무료 웹 도구 |

---

## 14. 개발 로드맵

```
Phase 1 (MVP) — 2~3주
├── [x] rembg 기반 대상/배경 분리
├── [x] OpenCV 인페인팅
├── [x] 기본 줌인/줌아웃 모션
├── [x] 2레이어 패럴랙스 합성
└── [x] MP4 출력

Phase 2 (품질 개선) — 2주
├── [ ] SAM2 세그멘테이션 옵션 추가
├── [ ] LaMa 인페인팅 통합
├── [ ] 다양한 모션 프리셋 (6종+)
├── [ ] 이징 함수 라이브러리 확장
└── [ ] Gradio 웹 UI

Phase 3 (고급 기능) — 3주
├── [ ] 깊이 맵 기반 다중 레이어
├── [ ] 3D 카메라 모션 시뮬레이션
├── [ ] 앰비언트 오디오 자동 매칭
├── [ ] 배치 처리 모드
├── [ ] GPU 가속 렌더링
└── [ ] 비디오 해상도 업스케일 (Real-ESRGAN)

Phase 4 (제품화) — 2주
├── [ ] 데스크톱 앱 패키징 (PyInstaller)
├── [ ] REST API 서버 모드
├── [ ] Docker 이미지
└── [ ] CI/CD 파이프라인
```

---

## 15. 요약

| 항목 | 내용 |
|------|------|
| **핵심 가치** | 이미지 1장 → 시네마틱 패럴랙스 영상 자동 생성 |
| **주요 기술** | AI 세그멘테이션 + 인페인팅 + 패럴랙스 켄번스 |
| **추천 스택** | Python + rembg + OpenCV + MoviePy + Gradio |
| **MVP 기간** | 2~3주 (Python 경험자 기준) |
| **최소 사양** | CPU만으로 가능 (GPU 있으면 5~10배 빠름) |
| **확장성** | 깊이 맵 다중 레이어, 3D 모션, 배치 처리 가능 |
