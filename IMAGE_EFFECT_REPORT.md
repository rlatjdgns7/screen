# 유튜브 영상 이미지 연출 기법 & 자동화 방안 보고서

## 1. 문제 인식: "이미지를 그냥 넣으면 왜 심심한가?"

정지 이미지를 영상에 그대로 삽입하면 시청자의 시선이 0.5~1초 만에 정보를 소화하고, 이후 **시각적 자극이 없어** 이탈한다. 유튜브 숏폼/롱폼 모두 평균 체류 시간이 **첫 3초에 결정**되므로, 이미지 자체에 움직임과 변화를 부여하는 것이 핵심이다.

---

## 2. 이미지를 재미있게 넣는 연출 기법 12가지

### 2.1 모션 계열 (움직임 부여)

| # | 기법 | 설명 | 난이도 | 임팩트 |
|---|------|------|--------|--------|
| 1 | **켄번스 (Zoom & Pan)** | 천천히 줌인/줌아웃 + 이동. 다큐/브이로그 느낌 | 낮음 | 중 |
| 2 | **패럴랙스 (2.5D)** | 전경/배경 분리 후 다른 속도로 이동. 3D 깊이감 | 중간 | 높음 |
| 3 | **줌 펄스 (Beat Sync)** | 음악 비트에 맞춰 줌이 펄떡거림. 숏폼 필수 | 낮음 | 높음 |
| 4 | **셰이크 (Camera Shake)** | 미세한 카메라 흔들림. 라이브/긴장감 연출 | 낮음 | 중 |
| 5 | **슬라이드 인/아웃** | 이미지가 화면 밖에서 밀려 들어오거나 나감 | 낮음 | 중 |
| 6 | **회전 등장 (Spin In)** | 이미지가 회전하며 등장. 밈/코미디 채널 | 낮음 | 중 |

### 2.2 시각 효과 계열 (Look & Feel)

| # | 기법 | 설명 | 난이도 | 임팩트 |
|---|------|------|--------|--------|
| 7 | **글리치 (Glitch)** | 디지털 노이즈/깨짐 효과. 강렬한 등장 | 중간 | 높음 |
| 8 | **크로매틱 애버레이션** | RGB 색 채널 어긋남. 사이버펑크/테크 느낌 | 낮음 | 중 |
| 9 | **비네팅 (Vignette)** | 가장자리 어둡게. 시선 집중 + 시네마틱 | 낮음 | 낮음 |
| 10 | **필름 그레인 (Grain)** | 필름 노이즈 추가. 레트로/시네마틱 | 낮음 | 낮음 |
| 11 | **컬러 그레이딩** | 색감 변환 (틸/오렌지, 흑백, 하이콘트라스트) | 중간 | 중 |
| 12 | **마스크 리빌 (Mask Reveal)** | 텍스트/도형 안에서 이미지가 드러남 | 중간 | 높음 |

### 2.3 전환 계열 (이미지 간 연결)

| # | 기법 | 설명 | 비고 |
|---|------|------|------|
| A | **모프 트랜지션** | 이미지 A가 B로 자연스럽게 변형 | AI 필요 |
| B | **와이프/스월 트랜지션** | 닦아내듯/소용돌이 전환 | 매우 일반적 |
| C | **줌 스루 (Zoom Through)** | 이미지 안으로 줌인하면 다음 이미지가 나타남 | 인기 높음 |
| D | **매치 컷** | 비슷한 구도의 두 이미지를 연결 | 고급 연출 |

---

## 3. 장르별 추천 조합

### 3.1 교육/강의 채널
```
켄번스(느린 줌인) + 비네팅 + 슬라이드 인
→ 차분하면서도 지루하지 않은 리듬
```

### 3.2 숏폼/밈/엔터테인먼트
```
줌 펄스(BPM 동기) + 글리치 + 크로매틱 애버레이션 + 셰이크
→ 강렬한 비트감, 짧은 시간 내 시각 폭격
```

### 3.3 브이로그/시네마틱
```
패럴랙스(2.5D) + 필름 그레인 + 컬러 그레이딩
→ 영화 같은 깊이감과 분위기
```

### 3.4 뉴스/정보 전달
```
슬라이드 인 + 약한 줌 + 마스크 리빌
→ 깔끔하고 전문적인 느낌
```

---

## 4. 현재 프로그램 분석

### 4.1 이미 구현된 기능

| 기능 | 모듈 | 위치 |
|------|------|------|
| 켄번스 줌/패닝 | `src/motion.py` | 5가지 프리셋 |
| 패럴랙스 (2레이어) | `src/renderer.py` | 전경/배경 분리 합성 |
| AI 세그멘테이션 | `src/segmenter.py` | 대상/배경 자동 분리 |
| AI 인페인팅 | `src/inpainter.py` | 배경 빈 영역 복원 |
| 줌 펄스 (BPM 동기) | `src/shortform.py` | 비트 동기 심장박동 줌 |
| 카메라 셰이크 | `src/shortform.py` | Hermite 보간 유기적 흔들림 |
| 글리치 | `src/shortform.py` | 수평 스트립 변위 + 인트로 글리치 |
| 크로매틱 애버레이션 | `src/shortform.py` | RGB 채널 분리 시프트 |
| 비네팅 | `src/shortform.py` | 가장자리 어둡게 |
| 필름 그레인 | `src/shortform.py` | 노이즈 오버레이 |
| 페이드 인/아웃 | `src/shortform.py` | 아웃트로 페이드 투 블랙 |

### 4.2 아직 없는 기능 (= 추가 기회)

| 기능 | 우선순위 | 구현 난이도 | 기대 효과 |
|------|----------|------------|----------|
| **슬라이드 인/아웃** | 높음 | 낮음 | 이미지 등장 연출 |
| **회전 등장** | 중간 | 낮음 | 코미디/밈 채널 |
| **컬러 그레이딩 프리셋** | 높음 | 낮음 | 분위기 전환 |
| **마스크 리빌** | 중간 | 중간 | 텍스트/도형 등장 |
| **줌 스루 트랜지션** | 높음 | 중간 | 멀티 이미지 연결 |
| **깊이맵 다중 레이어** | 낮음 | 높음 | 극한 3D 효과 |
| **텍스트 오버레이** | 높음 | 낮음 | 자막/강조 텍스트 |
| **이미지 시퀀스 지원** | 높음 | 중간 | 여러 이미지 연속 처리 |

---

## 5. 자동화 방안: 구체적 구현 설계

### 5.1 Phase 1 — 즉시 추가 가능 (1~2일)

#### 5.1.1 슬라이드 인/아웃 효과

`src/shortform.py`에 등장 애니메이션 추가:

```python
# 등장 방향 설정
class EntryDirection(Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

def _apply_slide_in(frame, canvas, progress, direction, out_w, out_h):
    """이미지가 화면 밖에서 밀려 들어오는 효과"""
    ease_t = ease_in_out_cubic(progress)
    if direction == "left":
        offset_x = int((1.0 - ease_t) * -out_w)
        offset_y = 0
    elif direction == "right":
        offset_x = int((1.0 - ease_t) * out_w)
        offset_y = 0
    # ...
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(frame, M, (out_w, out_h))
```

**적용 위치**: `ShortformConfig`에 `entry_style: str = "glitch"` 필드를 추가하고, 렌더링 루프의 인트로 구간에서 분기 처리.

#### 5.1.2 컬러 그레이딩 프리셋

```python
COLOR_GRADES = {
    "cinematic_teal_orange": {"shadows": [0, 30, 60], "highlights": [30, -10, -30]},
    "vintage_warm": {"gamma": 0.9, "saturation": 0.8, "warmth": 20},
    "high_contrast_bw": {"saturation": 0.0, "contrast": 1.4},
    "cyberpunk": {"shadows": [40, 0, 60], "highlights": [-20, 10, 0]},
    "pastel": {"saturation": 0.6, "brightness": 15, "contrast": 0.9},
}

def apply_color_grade(frame, grade_name):
    """LUT 없이 NumPy 연산으로 컬러 그레이딩"""
    grade = COLOR_GRADES[grade_name]
    # HSV 변환 → 채도/밝기 조정 → BGR 복원
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= grade.get("saturation", 1.0)
    hsv[:,:,2] *= grade.get("contrast", 1.0)
    hsv[:,:,2] += grade.get("brightness", 0)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
```

**적용 위치**: `ShortformConfig`에 `color_grade: str = "none"` 필드 추가 → 렌더링 루프 마지막 단계에서 적용.

#### 5.1.3 텍스트 오버레이

```python
def _render_text_overlay(frame, text, position, font_scale, color, thickness):
    """영상 위에 텍스트 렌더링 (PIL 사용 시 한글 지원)"""
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("malgun.ttf", size=font_scale)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

### 5.2 Phase 2 — 멀티 이미지 시퀀스 (3~5일)

현재 프로그램은 **이미지 1장 → 영상 1개** 구조이다.
유튜브 영상에서는 보통 여러 이미지를 연속으로 보여줘야 한다.

#### 5.2.1 설계: 시퀀스 파이프라인

```
입력: [img1.jpg, img2.jpg, img3.jpg, ...]
        ↓
각 이미지별 효과 적용 (개별 클립 생성)
        ↓
클립 간 트랜지션 적용 (줌 스루, 크로스 디졸브, 글리치 전환)
        ↓
하나의 MP4로 합체
```

#### 5.2.2 새 모듈: `src/sequence.py`

```python
@dataclass
class ImageClip:
    """이미지 한 장의 연출 설정"""
    path: str
    duration: float = 3.0
    effect: str = "kenburns"      # kenburns | shortform | static
    preset: str = "zoom_in"
    color_grade: str = "none"
    entry: str = "slide_left"     # slide_left | slide_right | glitch | fade | none
    text_overlay: str = ""

@dataclass
class SequenceConfig:
    """멀티 이미지 시퀀스 설정"""
    clips: list[ImageClip]
    transition: str = "crossfade"   # crossfade | zoom_through | glitch | cut
    transition_duration: float = 0.5
    output_w: int = 1920
    output_h: int = 1080
    fps: int = 30
    bgm_path: str = ""              # BGM 경로 (있으면 BPM 자동 감지)

def render_sequence(config: SequenceConfig, output_path: str):
    """멀티 이미지 시퀀스를 하나의 영상으로 렌더링"""
    writer = create_video_writer(output_path, ...)

    for i, clip in enumerate(config.clips):
        # 1. 이미지별 효과 렌더링 (기존 파이프라인 재활용)
        frames = render_single_clip(clip, config)

        # 2. 진입 애니메이션 적용
        frames = apply_entry_animation(frames, clip.entry)

        # 3. 이전 클립과의 트랜지션
        if i > 0:
            frames = apply_transition(prev_tail_frames, frames, config.transition)

        for frame in frames:
            write_frame(writer, frame)

        prev_tail_frames = frames[-int(config.transition_duration * config.fps):]

    finalize(writer)
```

#### 5.2.3 트랜지션 구현

```python
def _transition_crossfade(frames_a, frames_b, n_frames):
    """크로스 디졸브: A가 서서히 사라지고 B가 나타남"""
    result = []
    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1)
        blended = cv2.addWeighted(frames_a[i], 1.0 - alpha, frames_b[i], alpha, 0)
        result.append(blended)
    return result

def _transition_zoom_through(frames_a, frames_b, n_frames):
    """줌 스루: A 안으로 빨려 들어가면 B가 나타남"""
    result = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        # A: 점점 줌인 + 밝아짐 (화이트 아웃)
        zoom = 1.0 + t * 2.0
        brightness = int(t * 200)
        a_zoomed = zoom_center(frames_a[0], zoom)
        a_bright = cv2.add(a_zoomed, np.full_like(a_zoomed, brightness))
        # B: 줌아웃하며 등장
        if t > 0.4:
            bt = (t - 0.4) / 0.6
            b_zoom = 2.0 - bt * 1.0
            b_frame = zoom_center(frames_b[0], b_zoom)
            blended = cv2.addWeighted(a_bright, 1.0 - bt, b_frame, bt, 0)
            result.append(blended)
        else:
            result.append(a_bright)
    return result

def _transition_glitch(frames_a, frames_b, n_frames):
    """글리치 전환: A에 글리치 → 순간 전환 → B"""
    result = []
    mid = n_frames // 2
    for i in range(n_frames):
        if i < mid:
            intensity = i / mid
            result.append(_apply_glitch_intensity(frames_a[-1], intensity))
        else:
            intensity = 1.0 - (i - mid) / (n_frames - mid)
            result.append(_apply_glitch_intensity(frames_b[0], intensity))
    return result
```

### 5.3 Phase 3 — GUI 통합 (2~3일)

#### 5.3.1 새 탭: "이미지 시퀀스"

`gui.py`에 세 번째 탭을 추가:

```
┌──────────────────────────────────────────────┐
│ [켄번스 효과] [숏폼 임팩트] [이미지 시퀀스]    │
├──────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────┐        │
│  │ img1.jpg  [줌인] [슬라이드 등장]  │ [↑][↓] │
│  │ img2.jpg  [셰이크] [글리치 등장]  │ [↑][↓] │
│  │ img3.jpg  [패럴랙스] [페이드 등장] │ [↑][↓] │
│  │                                  │        │
│  │ [+ 이미지 추가]  [전체 삭제]       │        │
│  └──────────────────────────────────┘        │
│                                              │
│  트랜지션: [크로스 디졸브 ▼]  길이: [0.5초]    │
│  컬러 그레이딩: [시네마틱 ▼]                   │
│  BGM: [없음]  [찾아보기...]                   │
│                                              │
│  [========================] 45%              │
│  렌더링 중... 3/8 이미지                      │
│                                              │
│          [ 영상 생성 ]                        │
└──────────────────────────────────────────────┘
```

#### 5.3.2 드래그 앤 드롭

```python
# tkinterdnd2 라이브러리로 파일 드래그 앤 드롭 지원
def _on_drop(self, event):
    files = self._parse_drop_data(event.data)
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            self._add_clip(f)
```

### 5.4 Phase 4 — 고급 자동화 (1~2주)

#### 5.4.1 JSON 프리셋으로 배치 자동화

외부 JSON 파일로 영상 구성을 정의하고 CLI에서 일괄 처리:

```json
{
  "output": "my_video.mp4",
  "resolution": [1920, 1080],
  "fps": 30,
  "clips": [
    {
      "image": "intro.jpg",
      "duration": 2.0,
      "effect": "shortform",
      "entry": "glitch",
      "color_grade": "cyberpunk",
      "text": "EPISODE 1"
    },
    {
      "image": "main.jpg",
      "duration": 5.0,
      "effect": "kenburns",
      "preset": "dramatic_push"
    },
    {
      "image": "outro.jpg",
      "duration": 3.0,
      "effect": "shortform",
      "entry": "fade",
      "color_grade": "vintage_warm"
    }
  ],
  "transition": "zoom_through",
  "transition_duration": 0.5
}
```

CLI 사용:
```bash
python main.py --sequence config.json
```

#### 5.4.2 BGM BPM 자동 감지

```python
import librosa

def detect_bpm(audio_path):
    """BGM 파일에서 BPM 자동 감지"""
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)

def get_beat_timestamps(audio_path):
    """비트 타임스탬프 추출 → 이미지 전환 타이밍으로 활용"""
    y, sr = librosa.load(audio_path)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr)
```

활용: 비트 타이밍에 맞춰 이미지 전환 시점을 자동 결정.

#### 5.4.3 AI 자동 효과 선택

이미지 내용을 분석하여 적절한 효과를 자동 추천:

```python
def auto_select_effect(image_path):
    """이미지 분석 → 최적 효과 자동 선택"""
    img = cv2.imread(image_path)

    # 얼굴 감지 → 인물 사진이면 켄번스 줌인
    faces = detect_faces(img)
    if faces:
        return {"effect": "kenburns", "preset": "zoom_in"}

    # 가로로 넓은 풍경 → 패닝
    h, w = img.shape[:2]
    if w / h > 1.8:
        return {"effect": "kenburns", "preset": "pan_left_right"}

    # 고채도/화려한 이미지 → 숏폼 임팩트
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_saturation = hsv[:,:,1].mean()
    if avg_saturation > 120:
        return {"effect": "shortform", "entry": "glitch"}

    # 기본값
    return {"effect": "kenburns", "preset": "slow_drift"}
```

---

## 6. 구현 우선순위 로드맵

```
Phase 1 (즉시, 1~2일) ─────────────────────────
  [1] 슬라이드 인/아웃 등장 애니메이션
  [2] 컬러 그레이딩 프리셋 5종
  [3] 회전 등장 효과
      → ShortformConfig에 필드 추가 + 렌더 루프 분기

Phase 2 (단기, 3~5일) ─────────────────────────
  [4] 멀티 이미지 시퀀스 모듈 (src/sequence.py)
  [5] 트랜지션 3종 (크로스 디졸브, 줌 스루, 글리치)
  [6] 텍스트 오버레이 (PIL 한글 렌더링)
      → 새 모듈 생성 + main.py에 --sequence 옵션

Phase 3 (중기, 2~3일) ─────────────────────────
  [7] GUI 세 번째 탭 (이미지 시퀀스 편집)
  [8] 드래그 앤 드롭 이미지 추가
  [9] 클립 순서 변경 / 개별 효과 설정 UI
      → gui.py 탭 추가

Phase 4 (장기, 1~2주) ─────────────────────────
  [10] JSON 프리셋 배치 자동화
  [11] BGM BPM 자동 감지 + 비트 동기 전환
  [12] AI 자동 효과 선택 (이미지 분석 기반)
       → 완전 자동화 파이프라인
```

---

## 7. 기존 코드 수정 포인트 요약

### 7.1 `src/shortform.py` 수정

```python
# ShortformConfig에 추가할 필드
entry_style: str = "glitch"       # glitch | slide_left | slide_right | fade | spin | none
color_grade: str = "none"         # none | cinematic | vintage | cyberpunk | pastel | bw
text_overlay: str = ""            # 텍스트 (빈 문자열이면 미표시)
text_position: str = "center"     # center | top | bottom
```

렌더링 루프에서:
- 인트로 구간: `entry_style`에 따라 `_apply_intro_glitch` 대신 분기
- 마지막 단계: `color_grade`가 "none"이 아니면 컬러 그레이딩 적용
- 텍스트: `text_overlay`가 있으면 PIL로 텍스트 렌더링

### 7.2 `gui.py` 수정

- 숏폼 탭에 콤보박스 추가: 등장 효과 / 컬러 그레이딩
- 텍스트 입력 필드 추가
- (Phase 3) 세 번째 탭 "이미지 시퀀스" 추가

### 7.3 새 파일 생성

| 파일 | 역할 |
|------|------|
| `src/color_grade.py` | 컬러 그레이딩 프리셋 + 적용 함수 |
| `src/transitions.py` | 트랜지션 효과 (크로스 디졸브, 줌 스루, 글리치) |
| `src/sequence.py` | 멀티 이미지 시퀀스 렌더링 파이프라인 |
| `src/text_overlay.py` | PIL 기반 텍스트 렌더링 (한글 지원) |

---

## 8. 핵심 요약

| 항목 | 내용 |
|------|------|
| **현재 상태** | 켄번스 + 숏폼 임팩트 (단일 이미지) — 이미 효과 10종 구현 |
| **가장 큰 개선점** | 멀티 이미지 시퀀스 + 트랜지션 (실제 유튜브 영상은 이미지 여러 장) |
| **가성비 높은 추가** | 슬라이드 등장 + 컬러 그레이딩 (1~2일이면 체감 효과 큼) |
| **장기 목표** | JSON 배치 + BGM 비트 동기 + AI 자동 효과 선택 |
| **기존 코드 영향** | 최소한 — 새 모듈 추가 위주, 기존 파이프라인은 그대로 재활용 |
