# 시네마틱 켄번스 효과 — 화질 저하 근본 원인 분석 및 해결 방안

## 1. 요약

현재 파이프라인은 **6개 지점**에서 화질이 누적 손실된다.
가장 치명적인 3가지 원인은:

| 순위 | 원인 | 심각도 | 현재 구현 | 최적 대안 |
|:---:|------|:------:|----------|----------|
| 1 | **인페인팅** | 치명적 | `cv2.inpaint (Telea)` — 주변 픽셀 번짐 | **LaMa** — AI 텍스처 인식 복원 |
| 2 | **업스케일링** | 심각 | `cv2.resize (LANCZOS4)` — 수학적 보간 | **Real-ESRGAN** — AI 디테일 생성 |
| 3 | **세그멘테이션** | 높음 | `rembg u2net` — IoU 0.64 | **BiRefNet** — IoU 0.87 |

아래에서 전체 파이프라인을 단계별로 분석한다.

---

## 2. 파이프라인 데이터 흐름과 화질 손실 지점

```
입력 이미지 (예: 1200×800)
    │
    ▼ ① 세그멘테이션 (rembg u2net)
    │   ├─ 대상 BGRA (1200×800)
    │   ├─ 알파 마스크 (1200×800) ← GaussianBlur 페더링으로 에지 흐려짐
    │   └─ 원본 BGR (1200×800)
    │
    ▼ ② 인페인팅 (cv2.inpaint TELEA)
    │   ├─ 배경 확장 20% → 1680×1120
    │   ├─ 마스크 팽창 20px
    │   └─ Telea 알고리즘 → ██ 대형 영역 번짐/뭉개짐 ██
    │
    ▼ ③ 소스 업스케일 (cv2.resize LANCZOS4)
    │   ├─ 필요 크기: 1920 × 1.4줌 × 1.15여유 = 3091px
    │   └─ 1680→3091 (1.84배) → ██ 보간으로 디테일 손실 ██
    │
    ▼ ④ 프레임별 렌더링 (warpAffine)
    │   ├─ 줌 크롭 → 1920×1080 출력
    │   ├─ 알파 블렌딩 (float32→uint8 반올림 오차)
    │   └─ 인페인팅 뭉개짐이 배경에 그대로 노출
    │
    ▼ ⑤ H.264 인코딩 (FFmpeg)
    │   ├─ YUV420P 크로마 서브샘플링 → 색상 해상도 75% 손실
    │   └─ CRF 18 양자화 → 미세 디테일 손실
    │
    ▼ 출력 MP4
```

---

## 3. 단계별 상세 분석

### 3.1 세그멘테이션 — `rembg u2net`

**현재 문제:**
- u2net 모델의 정확도가 낮음 (DIS5K 데이터셋 IoU: **0.39**)
- 머리카락, 반투명 오브젝트, 복잡한 윤곽에서 마스크가 거칠음
- GaussianBlur 페더링이 에지를 추가로 흐리게 함
- 마스크 부정확 → 합성 시 **헤일로(빛번짐) 아티팩트** 발생

**모델별 정확도 비교:**

| 모델 | IoU (평균) | Dice | 모델 크기 | VRAM | 라이선스 |
|------|:----------:|:----:|:---------:|:----:|:--------:|
| u2net (현재) | 0.64 | 0.73 | 176 MB | ~2 GB | Apache 2.0 |
| **isnet-general-use** | **0.82** | **0.89** | 179 MB | ~2-3 GB | Apache 2.0 |
| **birefnet-general** | **0.87** | **0.92** | 973 MB | ~3.5 GB | MIT |
| RMBG-2.0 | ~0.90+ | ~0.94+ | ~800 MB | ~3.5 GB | 비상업용 |

**해결책:**
```python
# 현재 (u2net)
session = new_session("u2net")

# 권장 — 같은 rembg, 모델만 교체
session = new_session("birefnet-general")      # 최고 품질
session = new_session("isnet-general-use")     # 가성비 (크기 비슷, 성능 대폭 향상)
```

---

### 3.2 인페인팅 — `cv2.inpaint` (가장 치명적)

**현재 문제:**
```
cv2.inpaint(extended, inpaint_mask, 7, cv2.INPAINT_TELEA)
```
- Telea 알고리즘은 **작은 흠집 제거용** (1990년대 논문 기반)
- 대상을 제거한 큰 영역에 적용하면 → **심한 번짐/뭉개짐**
- 의미론적 이해 없음 (텍스처, 패턴, 구조를 모름)
- 카메라가 이동하면서 인페인팅된 배경이 **그대로 노출** → 화질 저하의 주범

**cv2.inpaint vs LaMa 비교:**

| 항목 | cv2.inpaint (Telea) | LaMa |
|------|:-------------------:|:----:|
| 알고리즘 | 이웃 픽셀 확산 (수학적) | 푸리에 합성곱 (AI) |
| 대형 마스크 | 번짐/뭉개짐 | 텍스처 자연스럽게 복원 |
| 구조 인식 | 없음 | 글로벌 컨텍스트 이해 |
| 속도 (GPU) | ~10ms | ~2초 |
| VRAM | 0 (CPU만) | ~2-4 GB |

**해결책 — simple-lama-inpainting:**
```bash
pip install simple-lama-inpainting torch torchvision
```
```python
from simple_lama_inpainting import SimpleLama
from PIL import Image

lama = SimpleLama()
result = lama(Image.fromarray(bg_rgb), Image.fromarray(mask_gray))
```

---

### 3.3 업스케일링 — `cv2.resize LANCZOS4`

**현재 문제:**
- LANCZOS4는 **수학적 보간**일 뿐 — 존재하지 않는 디테일을 만들 수 없음
- 예: 1200px → 3091px (2.6배) → 모든 디테일이 부드럽게 퍼짐
- 인페인팅 아티팩트도 함께 업스케일되어 **더 크게 보임**

**보간법별 품질:**

| 방법 | 원리 | 2배 업스케일 품질 | 4배 업스케일 품질 |
|------|------|:----------------:|:----------------:|
| INTER_LINEAR | 이중선형 보간 | 흐림 | 매우 흐림 |
| INTER_LANCZOS4 (현재) | 싱크 함수 보간 | 약간 흐림 | 흐림 |
| **Real-ESRGAN** | AI 생성 모델 | **선명, 디테일 복원** | **선명, 텍스처 생성** |

**해결책 — Real-ESRGAN:**
```bash
pip install realesrgan basicsr
```
```python
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path="RealESRGAN_x4plus.pth",
    model=model,
    tile=512,       # VRAM 절약용 타일링
    half=True,      # FP16 → VRAM 절반
    gpu_id=0,
)
output, _ = upsampler.enhance(img, outscale=2)  # 2배 또는 4배
```

| 설정 | VRAM 사용량 | 속도 |
|------|:----------:|:----:|
| tile=0, half=False | ~6 GB | ~4초 |
| tile=512, half=True | **~2 GB** | ~6초 |
| tile=256, half=True | **~1.5 GB** | ~8초 |

RTX 4060 8GB에서 `tile=512, half=True` → 충분히 동작

---

### 3.4 프레임 렌더링 — `warpAffine`

**현재 문제:**
- 매 프레임마다 어파인 변환 → 보간 누적
- 줌 1.6x에서 출력 픽셀의 **84%가 보간으로 생성**
- float32→uint8 변환 시 반올림 오차 (프레임당 미미하지만 누적)

**해결책:**
- 소스가 충분히 크면 **다운스케일 크롭**이 됨 → `INTER_AREA` 사용 → 품질 손실 거의 없음
- 즉, **3.3의 업스케일링을 먼저 충분히 하면 이 단계 문제는 자동 해결**

---

### 3.5 영상 인코딩 — H.264 / YUV420P

**현재 문제:**
- `yuv420p` 크로마 서브샘플링: 색상 해상도 **75% 감소**
- CRF 18: 일반적으로 고화질이지만, preset "medium"은 속도/품질 타협

**개선 옵션:**

| 설정 | 현재 | 권장 (고화질) | 최고 화질 |
|------|:----:|:------------:|:--------:|
| 코덱 | libx264 | libx264 | **libx265** |
| CRF | 18 | **15** | **12** |
| preset | medium | **slow** | **slower** |
| 픽셀 포맷 | yuv420p | **yuv444p** | **yuv444p10le** |
| 프로파일 | baseline | **high444** | high444 |

```python
# 현재
["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]

# 권장 (고화질)
["-c:v", "libx264", "-preset", "slow", "-crf", "15",
 "-pix_fmt", "yuv444p", "-profile:v", "high444"]

# 최고 화질 (파일 크기 대폭 증가)
["-c:v", "libx265", "-preset", "slow", "-crf", "12",
 "-pix_fmt", "yuv444p10le", "-tag:v", "hvc1"]
```

> **참고:** yuv444p는 일부 플레이어에서 호환되지 않을 수 있음. 범용성이 필요하면 yuv420p + CRF 15로 타협.

---

### 3.6 파이프라인 순서 문제

**현재 순서 (문제):**
```
원본(작음) → 세그멘테이션 → 인페인팅(작은 해상도) → 업스케일 → 렌더링
                                    ↑
                         번짐이 업스케일되어 확대됨
```

**권장 순서:**
```
원본 → 세그멘테이션 → AI 업스케일(원본+마스크) → AI 인페인팅(고해상도) → 렌더링
                                                        ↑
                                             고해상도에서 인페인팅 → 품질 유지
```

---

## 4. 종합 해결 방안

### 4.1 즉시 적용 가능 (코드 변경만)

| # | 변경 | 효과 | 난이도 |
|:-:|------|:----:|:------:|
| 1 | 세그멘테이션 모델: `u2net` → `isnet-general-use` | 마스크 정확도 +28% | 1줄 변경 |
| 2 | 인코딩: CRF 18 → 15, preset slow | 인코딩 품질 향상 | 2줄 변경 |
| 3 | 인코딩: yuv420p → yuv444p | 색상 손실 제거 | 1줄 변경 |

### 4.2 라이브러리 추가 설치 필요

| # | 변경 | 효과 | 추가 설치 | VRAM |
|:-:|------|:----:|----------|:----:|
| 4 | **LaMa 인페인팅** | 배경 복원 품질 혁신적 향상 | `simple-lama-inpainting` + `torch` | ~2-4 GB |
| 5 | **Real-ESRGAN 업스케일** | AI 디테일 복원 | `realesrgan` + `basicsr` | ~2 GB (타일링) |
| 6 | **BiRefNet 세그멘테이션** | 최고 품질 마스크 | rembg 내장 (추가 없음) | ~3.5 GB |

### 4.3 RTX 4060 8GB VRAM 예산

```
LaMa 인페인팅:          ~2-3 GB  (1회, 세그먼트 후 실행)
Real-ESRGAN 업스케일:   ~2 GB    (tile=512, half=True)
BiRefNet 세그멘테이션:  ~3.5 GB  (1회 실행)
─────────────────────────────────
순차 실행이므로 최대 ~3.5 GB → RTX 4060 8GB에서 충분
```

---

## 5. 개선 전후 비교 (예상)

### 현재 파이프라인 (Phase 1 MVP)
```
입력 1200×800
  → u2net (IoU 0.64, 거친 마스크)
  → cv2.inpaint (번짐/뭉개짐)
  → LANCZOS4 업스케일 (흐림)
  → H.264 yuv420p CRF 18
  = 눈에 띄는 화질 저하
```

### 개선된 파이프라인
```
입력 1200×800
  → BiRefNet (IoU 0.87, 정밀 마스크)
  → Real-ESRGAN 2x 업스케일 (2400×1600, AI 디테일 복원)
  → LaMa 인페인팅 (고해상도에서 자연스러운 텍스처 복원)
  → 크롭만으로 1920×1080 출력 (추가 업스케일 불필요)
  → H.264 yuv444p CRF 15 preset slow
  = 원본에 가까운 화질 유지
```

### 핵심 차이

| 단계 | 현재 | 개선 후 |
|------|------|---------|
| 마스크 품질 | 거친 에지, 헤일로 | 정밀한 에지, 자연스러운 경계 |
| 배경 복원 | 번짐, 뭉개짐 | 텍스처 인식 자연 복원 |
| 업스케일 | 수학적 보간 (흐림) | AI 디테일 생성 (선명) |
| 색상 보존 | 75% 색상 손실 (420p) | 100% 보존 (444p) |

---

## 6. 구현 우선순위

```
[1순위] LaMa 인페인팅 교체          ← 가장 큰 화질 개선
[2순위] Real-ESRGAN 업스케일 교체    ← 두 번째로 큰 개선
[3순위] 인코딩 설정 강화             ← 즉시 적용, 간단
[4순위] BiRefNet 세그멘테이션 교체   ← 마스크 품질 대폭 향상
[5순위] 파이프라인 순서 재구성       ← 업스케일 → 인페인팅 순서로
```

---

## 7. 필요 패키지 설치 (한 번에)

```bash
# PyTorch GPU (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# AI 인페인팅
pip install simple-lama-inpainting

# AI 업스케일링
pip install realesrgan basicsr

# 세그멘테이션은 rembg 내장 모델 교체만 하면 됨 (추가 설치 불필요)
```

총 추가 디스크: ~1.5 GB (모델 포함)
총 추가 VRAM: 순차 실행 시 최대 ~3.5 GB

---

*작성일: 2026-03-03*
*환경: Python 3.10, RTX 4060 8GB, CUDA 12.9, Windows 11*
