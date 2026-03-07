"""이미지 → 영상 효과 — 원클릭 GUI

탭 1: 시네마틱 켄번스 효과 (AI 세그멘테이션 + 패럴랙스)
탭 2: 숏폼 임팩트 (줌펄스 + 셰이크 + 글리치, AI 없이 초고속)
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk

import traceback


def _show_error_dialog(parent, title: str, message: str):
    """복사 가능한 에러 다이얼로그"""
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.configure(bg="#1e1e2e")
    dlg.resizable(True, True)
    dlg.grab_set()

    # 아이콘 + 제목
    header = ttk.Frame(dlg)
    header.pack(fill="x", padx=16, pady=(12, 4))
    ttk.Label(header, text=title, font=("맑은 고딕", 12, "bold"),
              foreground="#f38ba8", background="#1e1e2e").pack(side="left")

    # 에러 텍스트 (선택/복사 가능)
    text_frame = ttk.Frame(dlg)
    text_frame.pack(fill="both", expand=True, padx=16, pady=8)

    text = tk.Text(text_frame, wrap="word", font=("Consolas", 10),
                   bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4",
                   selectbackground="#585b70", selectforeground="#cdd6f4",
                   relief="flat", borderwidth=0, padx=8, pady=8)
    scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)

    text.insert("1.0", message)
    text.configure(state="disabled")  # 편집 불가, 선택/복사는 가능

    # 버튼 영역
    btn_frame = ttk.Frame(dlg)
    btn_frame.pack(fill="x", padx=16, pady=(0, 12))

    def _copy_all():
        parent.clipboard_clear()
        parent.clipboard_append(message)
        copy_btn.configure(text="복사됨!")
        dlg.after(1500, lambda: copy_btn.configure(text="전체 복사"))

    copy_btn = ttk.Button(btn_frame, text="전체 복사", command=_copy_all)
    copy_btn.pack(side="left")
    ttk.Button(btn_frame, text="닫기", command=dlg.destroy).pack(side="right")

    # 크기/위치
    dlg.update_idletasks()
    w, h = 560, 340
    x = parent.winfo_x() + (parent.winfo_width() - w) // 2
    y = parent.winfo_y() + (parent.winfo_height() - h) // 2
    dlg.geometry(f"{w}x{h}+{x}+{y}")
    dlg.minsize(400, 200)


from src.segmenter import segment_image
from src.upscaler import upscale_image
from src.inpainter import inpaint_background
from src.motion import get_preset, get_frame_params, PRESETS
from src.renderer import render_frame
from src.video import create_video_writer, write_frame, finalize
from src.color_grade import GRADE_LABELS
from src.transitions import TRANSITION_LABELS

PRESET_LABELS = {
    "zoom_in": "줌인 (대상으로 접근)",
    "zoom_out": "줌아웃 (대상에서 후퇴)",
    "pan_left_right": "패닝 (좌→우)",
    "dramatic_push": "극적 전진 (강한 줌)",
    "slow_drift": "느린 표류 (미세 이동)",
}

RESOLUTIONS = {
    "1920x1080 (FHD)": (1920, 1080),
    "1280x720 (HD)": (1280, 720),
    "3840x2160 (4K)": (3840, 2160),
    "1080x1920 (세로)": (1080, 1920),
    "720x1280 (세로HD)": (720, 1280),
}

SHORTFORM_RESOLUTIONS = {
    "1080x1920 (세로FHD)": (1080, 1920),
    "720x1280 (세로HD)": (720, 1280),
    "1920x1080 (FHD)": (1920, 1080),
    "1280x720 (HD)": (1280, 720),
}

INTENSITY_LABELS = {
    "부드러움": 0.5,
    "보통": 1.0,
    "강렬함": 1.5,
    "극한": 2.0,
}

ENTRY_LABELS = {
    "글리치": "glitch",
    "페이드": "fade",
    "슬라이드(좌→)": "slide_left",
    "슬라이드(→우)": "slide_right",
    "슬라이드(아래↑)": "slide_up",
    "회전 등장": "spin",
    "없음": "none",
}

EFFECT_LABELS = {
    "켄번스 (줌/패닝)": "kenburns",
    "숏폼 (줌펄스)": "shortform",
    "정적 (효과 없음)": "static",
}

SEQ_PRESET_LABELS = {
    "줌인": "zoom_in",
    "줌아웃": "zoom_out",
    "패닝 (좌→우)": "pan_left_right",
    "극적 전진": "dramatic_push",
    "느린 표류": "slow_drift",
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("이미지 → 영상 효과")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        # ── 공통 변수 ──
        self._input_path = tk.StringVar()
        self._output_path = tk.StringVar()
        self._progress = tk.DoubleVar(value=0.0)
        self._status = tk.StringVar(value="이미지를 선택하세요")
        self._running = False
        self._thumb_label = None
        self._photo = None

        # ── 켄번스 변수 ──
        self._preset = tk.StringVar(value="zoom_in")
        self._resolution = tk.StringVar(value="1920x1080 (FHD)")
        self._duration = tk.DoubleVar(value=5.0)
        self._fps = tk.IntVar(value=30)
        self._crf = tk.IntVar(value=15)
        self._nvenc = tk.BooleanVar(value=False)

        # ── 숏폼 변수 ──
        self._sf_bpm = tk.IntVar(value=120)
        self._sf_intensity = tk.StringVar(value="보통")
        self._sf_resolution = tk.StringVar(value="1080x1920 (세로FHD)")
        self._sf_duration = tk.DoubleVar(value=5.0)
        self._sf_fps = tk.IntVar(value=30)
        self._sf_crf = tk.IntVar(value=18)
        self._sf_nvenc = tk.BooleanVar(value=False)
        self._sf_entry = tk.StringVar(value="글리치")
        self._sf_color_grade = tk.StringVar(value="없음")
        self._sf_text = tk.StringVar(value="")

        # ── 시퀀스 변수 ──
        self._seq_clips: list[dict] = []  # [{path, effect, preset, entry, grade, text}, ...]
        self._seq_transition = tk.StringVar(value="크로스 디졸브")
        self._seq_trans_dur = tk.DoubleVar(value=0.5)
        self._seq_clip_dur = tk.DoubleVar(value=3.0)
        self._seq_resolution = tk.StringVar(value="1920x1080 (FHD)")
        self._seq_fps = tk.IntVar(value=30)
        self._seq_crf = tk.IntVar(value=18)
        self._seq_nvenc = tk.BooleanVar(value=False)

        self._build_ui()
        self.update_idletasks()
        self.minsize(540, 780)
        self.geometry("540x780")
        self._center()

    # ─── UI 빌드 ───────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("맑은 고딕", 10))
        style.configure("Header.TLabel", background=bg, foreground=accent, font=("맑은 고딕", 14, "bold"))
        style.configure("Status.TLabel", background=bg, foreground="#a6adc8", font=("맑은 고딕", 9))
        style.configure("TButton", font=("맑은 고딕", 10))
        style.configure("Go.TButton", font=("맑은 고딕", 13, "bold"))
        style.configure("TCheckbutton", background=bg, foreground=fg, font=("맑은 고딕", 10))
        style.configure("TCombobox", font=("맑은 고딕", 10))
        style.configure("TScale", background=bg)
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", font=("맑은 고딕", 10, "bold"), padding=[14, 4])

        pad = {"padx": 12, "pady": 4}
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # 헤더
        ttk.Label(main, text="이미지 → 영상 효과", style="Header.TLabel").pack(pady=(0, 8))

        # 미리보기
        self._thumb_frame = ttk.Frame(main, width=400, height=225)
        self._thumb_frame.pack(pady=(0, 8))
        self._thumb_frame.pack_propagate(False)
        self._thumb_label = ttk.Label(self._thumb_frame, text="미리보기", anchor="center")
        self._thumb_label.pack(fill="both", expand=True)

        # 입력
        row = ttk.Frame(main); row.pack(fill="x", **pad)
        ttk.Label(row, text="입력 이미지").pack(side="left")
        ttk.Entry(row, textvariable=self._input_path, width=40, state="readonly").pack(side="left", padx=(8, 4))
        ttk.Button(row, text="찾아보기…", command=self._browse_input).pack(side="left")

        # 출력
        row = ttk.Frame(main); row.pack(fill="x", **pad)
        ttk.Label(row, text="출력  영상 ").pack(side="left")
        ttk.Entry(row, textvariable=self._output_path, width=40, state="readonly").pack(side="left", padx=(8, 4))
        ttk.Button(row, text="저장위치…", command=self._browse_output).pack(side="left")

        # ── 하단 공통 (먼저 배치해야 항상 보임) ──
        bottom = ttk.Frame(main)
        bottom.pack(side="bottom", fill="x")

        self._go_btn = ttk.Button(bottom, text="영상 생성", style="Go.TButton", command=self._run)
        self._go_btn.pack(pady=(8, 4), ipadx=30, ipady=6)
        ttk.Label(bottom, textvariable=self._status, style="Status.TLabel").pack(**pad)
        self._pbar = ttk.Progressbar(bottom, variable=self._progress, maximum=100)
        self._pbar.pack(fill="x", **pad)
        ttk.Separator(bottom, orient="horizontal").pack(fill="x", pady=(8, 0), side="bottom")

        # ── 탭 ──
        self._notebook = ttk.Notebook(main)
        self._notebook.pack(fill="both", expand=True, pady=(8, 0))

        kb_tab = ttk.Frame(self._notebook)
        self._notebook.add(kb_tab, text="  켄번스 효과  ")
        self._build_kenburns_tab(kb_tab, pad)

        sf_tab = ttk.Frame(self._notebook)
        self._notebook.add(sf_tab, text="  숏폼 임팩트  ")
        self._build_shortform_tab(sf_tab, pad)

        seq_tab = ttk.Frame(self._notebook)
        self._notebook.add(seq_tab, text="  이미지 시퀀스  ")
        self._build_sequence_tab(seq_tab, pad)

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _build_kenburns_tab(self, parent, pad):
        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="모션 프리셋").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self._preset, state="readonly", width=28,
                             values=list(PRESET_LABELS.keys()))
        combo.pack(side="left", padx=(8, 4))
        self._preset_desc = ttk.Label(row, text=PRESET_LABELS["zoom_in"], style="Status.TLabel")
        self._preset_desc.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="해상도       ").pack(side="left")
        ttk.Combobox(row, textvariable=self._resolution, state="readonly", width=20,
                     values=list(RESOLUTIONS.keys())).pack(side="left", padx=(8, 0))

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="영상 길이(초)").pack(side="left")
        self._dur_label = ttk.Label(row, text="5.0초")
        ttk.Scale(row, from_=1, to=15, variable=self._duration, orient="horizontal", length=200,
                  command=lambda _: self._dur_label.configure(text=f"{self._duration.get():.1f}초")
                  ).pack(side="left", padx=(8, 4))
        self._dur_label.pack(side="left")

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="FPS").pack(side="left")
        ttk.Spinbox(row, from_=15, to=60, textvariable=self._fps, width=5).pack(side="left", padx=(8, 16))
        ttk.Label(row, text="CRF (품질)").pack(side="left")
        ttk.Spinbox(row, from_=10, to=30, textvariable=self._crf, width=5).pack(side="left", padx=(8, 16))
        ttk.Checkbutton(row, text="NVENC GPU 인코딩", variable=self._nvenc).pack(side="left")

    def _build_shortform_tab(self, parent, pad):
        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="BPM          ").pack(side="left")
        ttk.Spinbox(row, from_=60, to=200, increment=5, textvariable=self._sf_bpm, width=5
                    ).pack(side="left", padx=(8, 16))
        ttk.Label(row, text="효과 강도").pack(side="left")
        ttk.Combobox(row, textvariable=self._sf_intensity, state="readonly", width=10,
                     values=list(INTENSITY_LABELS.keys())).pack(side="left", padx=(8, 0))

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="등장 효과    ").pack(side="left")
        ttk.Combobox(row, textvariable=self._sf_entry, state="readonly", width=14,
                     values=list(ENTRY_LABELS.keys())).pack(side="left", padx=(8, 16))
        ttk.Label(row, text="컬러 그레이딩").pack(side="left")
        ttk.Combobox(row, textvariable=self._sf_color_grade, state="readonly", width=20,
                     values=list(GRADE_LABELS.values())).pack(side="left", padx=(8, 0))

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="텍스트       ").pack(side="left")
        ttk.Entry(row, textvariable=self._sf_text, width=40).pack(side="left", padx=(8, 0))

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="해상도       ").pack(side="left")
        ttk.Combobox(row, textvariable=self._sf_resolution, state="readonly", width=20,
                     values=list(SHORTFORM_RESOLUTIONS.keys())).pack(side="left", padx=(8, 0))

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="영상 길이(초)").pack(side="left")
        self._sf_dur_label = ttk.Label(row, text="5.0초")
        ttk.Scale(row, from_=1, to=15, variable=self._sf_duration, orient="horizontal", length=200,
                  command=lambda _: self._sf_dur_label.configure(text=f"{self._sf_duration.get():.1f}초")
                  ).pack(side="left", padx=(8, 4))
        self._sf_dur_label.pack(side="left")

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="FPS").pack(side="left")
        ttk.Spinbox(row, from_=15, to=60, textvariable=self._sf_fps, width=5).pack(side="left", padx=(8, 16))
        ttk.Label(row, text="CRF (품질)").pack(side="left")
        ttk.Spinbox(row, from_=10, to=30, textvariable=self._sf_crf, width=5).pack(side="left", padx=(8, 16))
        ttk.Checkbutton(row, text="NVENC GPU 인코딩", variable=self._sf_nvenc).pack(side="left")

    def _build_sequence_tab(self, parent, pad):
        # 클립 리스트
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True, **pad)

        self._seq_listbox = tk.Listbox(
            list_frame, height=6, bg="#181825", fg="#cdd6f4",
            selectbackground="#585b70", selectforeground="#cdd6f4",
            font=("Consolas", 9), relief="flat",
        )
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._seq_listbox.yview)
        self._seq_listbox.configure(yscrollcommand=scrollbar.set)
        self._seq_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 클립 버튼
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", **pad)
        ttk.Button(btn_frame, text="+ 이미지 추가", command=self._seq_add_clips).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="선택 삭제", command=self._seq_remove_clip).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="위로", command=lambda: self._seq_move(-1)).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="아래로", command=lambda: self._seq_move(1)).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="전체 삭제", command=self._seq_clear).pack(side="right")

        # 클립별 설정
        clip_frame = ttk.Frame(parent)
        clip_frame.pack(fill="x", **pad)
        ttk.Label(clip_frame, text="효과").pack(side="left")
        self._seq_effect = tk.StringVar(value="켄번스 (줌/패닝)")
        ttk.Combobox(clip_frame, textvariable=self._seq_effect, state="readonly", width=16,
                     values=list(EFFECT_LABELS.keys())).pack(side="left", padx=(4, 12))
        ttk.Label(clip_frame, text="프리셋").pack(side="left")
        self._seq_preset = tk.StringVar(value="줌인")
        ttk.Combobox(clip_frame, textvariable=self._seq_preset, state="readonly", width=12,
                     values=list(SEQ_PRESET_LABELS.keys())).pack(side="left", padx=(4, 12))

        clip_frame2 = ttk.Frame(parent)
        clip_frame2.pack(fill="x", **pad)
        ttk.Label(clip_frame2, text="등장").pack(side="left")
        self._seq_entry = tk.StringVar(value="페이드")
        ttk.Combobox(clip_frame2, textvariable=self._seq_entry, state="readonly", width=14,
                     values=list(ENTRY_LABELS.keys())).pack(side="left", padx=(4, 12))
        ttk.Label(clip_frame2, text="컬러").pack(side="left")
        self._seq_grade = tk.StringVar(value="없음")
        ttk.Combobox(clip_frame2, textvariable=self._seq_grade, state="readonly", width=20,
                     values=list(GRADE_LABELS.values())).pack(side="left", padx=(4, 0))

        clip_frame3 = ttk.Frame(parent)
        clip_frame3.pack(fill="x", **pad)
        ttk.Label(clip_frame3, text="텍스트").pack(side="left")
        self._seq_text = tk.StringVar(value="")
        ttk.Entry(clip_frame3, textvariable=self._seq_text, width=30).pack(side="left", padx=(4, 12))
        ttk.Button(clip_frame3, text="선택 항목에 적용", command=self._seq_apply_settings).pack(side="left")

        # 전체 설정
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4, padx=12)

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="트랜지션    ").pack(side="left")
        ttk.Combobox(row, textvariable=self._seq_transition, state="readonly", width=16,
                     values=list(TRANSITION_LABELS.values())).pack(side="left", padx=(4, 12))
        ttk.Label(row, text="길이(초)").pack(side="left")
        ttk.Spinbox(row, from_=0.1, to=2.0, increment=0.1, textvariable=self._seq_trans_dur,
                    width=5, format="%.1f").pack(side="left", padx=(4, 0))

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="클립 길이(초)").pack(side="left")
        ttk.Spinbox(row, from_=1.0, to=15.0, increment=0.5, textvariable=self._seq_clip_dur,
                    width=5, format="%.1f").pack(side="left", padx=(4, 12))
        ttk.Label(row, text="해상도").pack(side="left")
        ttk.Combobox(row, textvariable=self._seq_resolution, state="readonly", width=18,
                     values=list(RESOLUTIONS.keys())).pack(side="left", padx=(4, 0))

        row = ttk.Frame(parent); row.pack(fill="x", **pad)
        ttk.Label(row, text="FPS").pack(side="left")
        ttk.Spinbox(row, from_=15, to=60, textvariable=self._seq_fps, width=5).pack(side="left", padx=(4, 12))
        ttk.Label(row, text="CRF").pack(side="left")
        ttk.Spinbox(row, from_=10, to=30, textvariable=self._seq_crf, width=5).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(row, text="NVENC", variable=self._seq_nvenc).pack(side="left")

    # ─── 시퀀스 이벤트 ─────────────────────────────────

    def _seq_add_clips(self):
        paths = filedialog.askopenfilenames(
            title="이미지 선택 (여러 개 가능)",
            filetypes=[("이미지", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("모든 파일", "*.*")],
        )
        for p in paths:
            clip = {
                "path": p,
                "effect": EFFECT_LABELS.get(self._seq_effect.get(), "kenburns"),
                "preset": SEQ_PRESET_LABELS.get(self._seq_preset.get(), "zoom_in"),
                "entry": ENTRY_LABELS.get(self._seq_entry.get(), "fade"),
                "grade": self._seq_grade_key(),
                "text": self._seq_text.get(),
            }
            self._seq_clips.append(clip)
        self._seq_refresh_list()

        # 시퀀스 탭에서는 출력 경로 자동 설정
        if self._seq_clips and not self._output_path.get():
            first = self._seq_clips[0]["path"]
            parent_dir = str(Path(first).parent)
            self._output_path.set(os.path.join(parent_dir, "sequence_output.mp4"))

    def _seq_grade_key(self) -> str:
        label = self._seq_grade.get()
        for k, v in GRADE_LABELS.items():
            if v == label:
                return k
        return "none"

    def _seq_remove_clip(self):
        sel = self._seq_listbox.curselection()
        if sel:
            idx = sel[0]
            self._seq_clips.pop(idx)
            self._seq_refresh_list()

    def _seq_move(self, direction: int):
        sel = self._seq_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        new_idx = idx + direction
        if 0 <= new_idx < len(self._seq_clips):
            self._seq_clips[idx], self._seq_clips[new_idx] = self._seq_clips[new_idx], self._seq_clips[idx]
            self._seq_refresh_list()
            self._seq_listbox.selection_set(new_idx)

    def _seq_clear(self):
        self._seq_clips.clear()
        self._seq_refresh_list()

    def _seq_apply_settings(self):
        sel = self._seq_listbox.curselection()
        if not sel:
            messagebox.showinfo("선택 필요", "설정을 적용할 클립을 선택하세요.")
            return
        idx = sel[0]
        self._seq_clips[idx]["effect"] = EFFECT_LABELS.get(self._seq_effect.get(), "kenburns")
        self._seq_clips[idx]["preset"] = SEQ_PRESET_LABELS.get(self._seq_preset.get(), "zoom_in")
        self._seq_clips[idx]["entry"] = ENTRY_LABELS.get(self._seq_entry.get(), "fade")
        self._seq_clips[idx]["grade"] = self._seq_grade_key()
        self._seq_clips[idx]["text"] = self._seq_text.get()
        self._seq_refresh_list()

    def _seq_refresh_list(self):
        self._seq_listbox.delete(0, tk.END)
        for i, clip in enumerate(self._seq_clips):
            name = Path(clip["path"]).name
            effect = clip.get("effect", "kenburns")
            entry = clip.get("entry", "none")
            grade = clip.get("grade", "none")
            text = clip.get("text", "")
            label = f"{i+1}. {name}  [{effect}] [{entry}]"
            if grade != "none":
                label += f" [{grade}]"
            if text:
                label += f' "{text}"'
            self._seq_listbox.insert(tk.END, label)

    # ─── 이벤트 핸들러 ─────────────────────────────────

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    def _on_preset_change(self, _event=None):
        self._preset_desc.configure(text=PRESET_LABELS.get(self._preset.get(), ""))

    def _on_tab_change(self, _event=None):
        tab_idx = self._notebook.index(self._notebook.select())
        if tab_idx == 2:
            # 시퀀스 탭: 클립이 있으면 첫 클립 기준 출력 경로
            if self._seq_clips:
                parent = str(Path(self._seq_clips[0]["path"]).parent)
                self._output_path.set(os.path.join(parent, "sequence_output.mp4"))
            return
        inp = self._input_path.get()
        if inp:
            stem = Path(inp).stem
            parent = str(Path(inp).parent)
            suffix = "_kenburns" if tab_idx == 0 else "_shortform"
            self._output_path.set(os.path.join(parent, f"{stem}{suffix}.mp4"))

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("이미지", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("모든 파일", "*.*")],
        )
        if path:
            self._input_path.set(path)
            stem = Path(path).stem
            tab_idx = self._notebook.index(self._notebook.select())
            suffix = "_kenburns" if tab_idx == 0 else "_shortform"
            self._output_path.set(os.path.join(str(Path(path).parent), f"{stem}{suffix}.mp4"))
            self._load_thumbnail(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="저장 위치", defaultextension=".mp4",
                                            filetypes=[("MP4 영상", "*.mp4")])
        if path:
            self._output_path.set(path)

    def _load_thumbnail(self, path: str):
        try:
            img = Image.open(path)
            img.thumbnail((400, 225))
            self._photo = ImageTk.PhotoImage(img)
            self._thumb_label.configure(image=self._photo, text="")
        except Exception:
            self._thumb_label.configure(image="", text="미리보기 실패")

    def _set_status(self, msg: str, pct: float | None = None):
        self._status.set(msg)
        if pct is not None:
            self._progress.set(pct)
        self.update_idletasks()

    # ─── 실행 ──────────────────────────────────────────

    def _run(self):
        if self._running:
            return
        tab_idx = self._notebook.index(self._notebook.select())
        inp = self._input_path.get()
        out = self._output_path.get()

        if tab_idx == 2:
            # 시퀀스 탭: 클립 리스트 확인
            if not self._seq_clips:
                messagebox.showwarning("입력 필요", "이미지를 추가하세요.")
                return
            if not out:
                if self._seq_clips:
                    parent = str(Path(self._seq_clips[0]["path"]).parent)
                    out = os.path.join(parent, "sequence_output.mp4")
                    self._output_path.set(out)
                else:
                    messagebox.showwarning("출력 필요", "저장 위치를 지정하세요.")
                    return
        else:
            if not inp:
                messagebox.showwarning("입력 필요", "이미지를 선택하세요.")
                return
            if not out:
                messagebox.showwarning("출력 필요", "저장 위치를 지정하세요.")
                return

        self._running = True
        self._go_btn.configure(state="disabled")

        if tab_idx == 0:
            threading.Thread(target=self._pipeline, args=(inp, out), daemon=True).start()
        elif tab_idx == 1:
            threading.Thread(target=self._shortform_pipeline, args=(inp, out), daemon=True).start()
        else:
            threading.Thread(target=self._sequence_pipeline, args=(out,), daemon=True).start()

    # ─── 켄번스 파이프라인 ─────────────────────────────

    def _pipeline(self, inp: str, out: str):
        import cv2
        import numpy as np

        try:
            w, h = RESOLUTIONS[self._resolution.get()]
            preset_name = self._preset.get()
            duration = self._duration.get()
            fps = self._fps.get()
            crf = self._crf.get()
            nvenc = self._nvenc.get()

            # 1. AI 프리업스케일 (출력 해상도 이상으로)
            extend_ratio = 0.20
            raw = cv2.imread(inp, cv2.IMREAD_COLOR)
            raw_h, raw_w = raw.shape[:2]

            if raw_w < w or raw_h < h:
                self._set_status(f"AI 프리업스케일 중… ({raw_w}x{raw_h} → {w}x{h}+)", 5)
                original_hd = upscale_image(raw, w, h)
            else:
                original_hd = raw
            hd_h, hd_w = original_hd.shape[:2]

            # 2. HD 세그멘테이션
            self._set_status("HD 세그멘테이션 중… (첫 실행 시 모델 다운로드)", 12)
            seg = segment_image(original_hd)
            self._set_status("세그멘테이션 완료", 18)

            # 3. 모션 설정
            config = get_preset(preset_name, bbox=seg.bbox, img_w=hd_w, img_h=hd_h,
                                duration=duration, fps=fps, extend_ratio=extend_ratio)
            total_frames = int(config.duration * config.fps)
            max_zoom = max(config.start_zoom, config.end_zoom)

            # 4. HD 인페인팅 + 줌 여유 업스케일
            self._set_status("HD 배경 인페인팅 중…", 22)
            bg_inpainted = inpaint_background(seg.original, seg.alpha, extend_ratio=extend_ratio)
            bg_inp_h, bg_inp_w = bg_inpainted.shape[:2]

            need_w = int(w * max_zoom * 1.05)
            need_h = int(h * max_zoom * 1.05)

            if bg_inp_w >= need_w and bg_inp_h >= need_h:
                bg_image = bg_inpainted
                fg_up = seg.foreground
                self._set_status(f"캔버스 충분 ({bg_inp_w}x{bg_inp_h})", 42)
            else:
                # LANCZOS4로 부족분만 보간 (이중 AI 방지)
                scale = max(need_w / bg_inp_w, need_h / bg_inp_h)
                new_w = int(bg_inp_w * scale)
                new_h = int(bg_inp_h * scale)
                self._set_status(f"줌 여유 보간 중… (LANCZOS4 {scale:.2f}x)", 32)
                bg_image = cv2.resize(bg_inpainted, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                fg_w_s, fg_h_s = seg.foreground.shape[1], seg.foreground.shape[0]
                fg_scale = max(new_w / fg_w_s, new_h / fg_h_s) if fg_w_s < new_w or fg_h_s < new_h else 1.0
                if fg_scale > 1.0:
                    fg_up = cv2.resize(seg.foreground, (int(fg_w_s * fg_scale), int(fg_h_s * fg_scale)),
                                       interpolation=cv2.INTER_LANCZOS4)
                else:
                    fg_up = seg.foreground
                self._set_status(f"보간 완료: {bg_image.shape[1]}x{bg_image.shape[0]}", 42)

            # 5. 캔버스 크기 맞춤
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

            # 6. 렌더링 + 인코딩
            self._set_status(f"렌더링 중… 0/{total_frames}", 45)
            writer = create_video_writer(out, width=w, height=h, fps=fps, crf=crf, use_nvenc=nvenc)

            for i in range(total_frames):
                bg_cx, bg_cy, bg_zoom, fg_cx, fg_cy, fg_zoom = get_frame_params(config, i, total_frames)
                frame = render_frame(bg_image, fg_up,
                                     bg_cx, bg_cy, bg_zoom, fg_cx, fg_cy, fg_zoom,
                                     out_w=w, out_h=h)
                write_frame(writer, frame)
                pct = 45 + 50 * (i + 1) / total_frames
                if i % 5 == 0 or i == total_frames - 1:
                    self._set_status(f"렌더링 중… {i+1}/{total_frames}", pct)

            finalize(writer)

            file_bytes = os.path.getsize(out)
            size_str = f"{file_bytes / (1024*1024):.1f} MB" if file_bytes >= 1024*1024 else f"{file_bytes / 1024:.1f} KB"
            self._set_status(f"완료! {size_str} — {out}", 100)

            if messagebox.askyesno("완료", f"영상이 생성되었습니다.\n{out}\n\n파일을 열까요?"):
                os.startfile(out)

        except Exception as e:
            err_detail = f"{e}\n\n── 상세 트레이스백 ──\n{traceback.format_exc()}"
            self._set_status(f"오류: {e}", 0)
            self.after(0, lambda: _show_error_dialog(self, "오류 발생", err_detail))
        finally:
            self._running = False
            self._go_btn.configure(state="normal")

    # ─── 숏폼 파이프라인 ───────────────────────────────

    def _shortform_pipeline(self, inp: str, out: str):
        try:
            from src.shortform import render_shortform, ShortformConfig

            w, h = SHORTFORM_RESOLUTIONS[self._sf_resolution.get()]
            intensity = INTENSITY_LABELS[self._sf_intensity.get()]

            # 등장 효과 키 변환
            entry_key = ENTRY_LABELS.get(self._sf_entry.get(), "glitch")

            # 컬러 그레이딩 키 변환 (한글 레이블 → 키)
            grade_label = self._sf_color_grade.get()
            grade_key = "none"
            for k, v in GRADE_LABELS.items():
                if v == grade_label:
                    grade_key = k
                    break

            config = ShortformConfig(
                duration=self._sf_duration.get(),
                fps=self._sf_fps.get(),
                bpm=float(self._sf_bpm.get()),
                zoom_amplitude=0.04 * intensity,
                shake_intensity=6.0 * intensity,
                chroma_base=1.5 * intensity,
                chroma_pulse=3.0 * intensity,
                vignette_strength=0.45,
                grain_intensity=12.0 * intensity,
                glitch_probability=0.04 * intensity,
                drift_range=15.0,
                entry_style=entry_key,
                color_grade=grade_key,
                text_overlay=self._sf_text.get(),
                text_position="bottom",
                text_font_size=48,
                output_w=w,
                output_h=h,
                crf=self._sf_crf.get(),
                use_nvenc=self._sf_nvenc.get(),
            )

            render_shortform(inp, out, config, progress_cb=self._set_status)

            if messagebox.askyesno("완료", f"영상이 생성되었습니다.\n{out}\n\n파일을 열까요?"):
                os.startfile(out)

        except Exception as e:
            err_detail = f"{e}\n\n── 상세 트레이스백 ──\n{traceback.format_exc()}"
            self._set_status(f"오류: {e}", 0)
            self.after(0, lambda: _show_error_dialog(self, "오류 발생", err_detail))
        finally:
            self._running = False
            self._go_btn.configure(state="normal")


    # ─── 시퀀스 파이프라인 ──────────────────────────────

    def _sequence_pipeline(self, out: str):
        try:
            from src.sequence import render_sequence, SequenceConfig, ClipConfig

            w, h = RESOLUTIONS[self._seq_resolution.get()]

            # 트랜지션 키 변환
            trans_label = self._seq_transition.get()
            trans_key = "crossfade"
            for k, v in TRANSITION_LABELS.items():
                if v == trans_label:
                    trans_key = k
                    break

            clips = []
            for c in self._seq_clips:
                clips.append(ClipConfig(
                    path=c["path"],
                    duration=self._seq_clip_dur.get(),
                    effect=c.get("effect", "kenburns"),
                    preset=c.get("preset", "zoom_in"),
                    entry=c.get("entry", "fade"),
                    color_grade=c.get("grade", "none"),
                    text=c.get("text", ""),
                    text_position="bottom",
                    font_size=48,
                ))

            config = SequenceConfig(
                clips=clips,
                transition=trans_key,
                transition_duration=self._seq_trans_dur.get(),
                output_w=w,
                output_h=h,
                fps=self._seq_fps.get(),
                crf=self._seq_crf.get(),
                use_nvenc=self._seq_nvenc.get(),
            )

            render_sequence(config, out, progress_cb=self._set_status)

            if messagebox.askyesno("완료", f"영상이 생성되었습니다.\n{out}\n\n파일을 열까요?"):
                os.startfile(out)

        except Exception as e:
            err_detail = f"{e}\n\n── 상세 트레이스백 ──\n{traceback.format_exc()}"
            self._set_status(f"오류: {e}", 0)
            self.after(0, lambda: _show_error_dialog(self, "오류 발생", err_detail))
        finally:
            self._running = False
            self._go_btn.configure(state="normal")


if __name__ == "__main__":
    App().mainloop()
