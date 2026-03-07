"""영상 인코딩 모듈 — FFmpeg subprocess pipe (고화질 설정)"""

import os
import subprocess
import shutil
import tempfile


def _find_ffmpeg() -> str:
    """FFmpeg 실행 파일 경로를 찾는다."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("FFmpeg를 찾을 수 없습니다. PATH에 FFmpeg를 추가하세요.")
    return path


def create_video_writer(
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    crf: int = 15,
    use_nvenc: bool = False,
) -> subprocess.Popen:
    """FFmpeg 파이프 기반 비디오 라이터를 생성한다."""
    ffmpeg = _find_ffmpeg()

    # 해상도 짝수 보장 (H.264 필수)
    width = max(width - width % 2, 2)
    height = max(height - height % 2, 2)

    # 출력 디렉토리 확인
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    input_args = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
    ]

    if use_nvenc:
        output_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "constqp",
            "-qp", str(crf),
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    else:
        output_args = [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            output_path,
        ]

    cmd = input_args + output_args

    # stderr를 임시 파일로 캡처 (PIPE 데드락 방지)
    stderr_path = os.path.join(
        tempfile.gettempdir(), f"ffmpeg_err_{os.getpid()}.log"
    )
    stderr_fh = open(stderr_path, "w+b")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=stderr_fh,
    )
    process._stderr_fh = stderr_fh
    process._stderr_path = stderr_path
    return process


def _read_stderr(process: subprocess.Popen) -> str:
    """FFmpeg stderr 로그를 읽고 임시 파일을 정리한다."""
    if not hasattr(process, "_stderr_fh"):
        return ""
    try:
        process._stderr_fh.seek(0)
        text = process._stderr_fh.read().decode(errors="replace").strip()
        process._stderr_fh.close()
    except Exception:
        text = ""
    try:
        os.unlink(process._stderr_path)
    except OSError:
        pass
    return text


def write_frame(process: subprocess.Popen, frame) -> None:
    """프레임을 FFmpeg 프로세스에 전송한다."""
    try:
        process.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError):
        process.wait()
        stderr = _read_stderr(process)
        raise RuntimeError(
            f"FFmpeg 파이프 끊김 (프로세스가 비정상 종료됨)\n"
            f"── FFmpeg 에러 로그 ──\n{stderr or '(출력 없음)'}"
        ) from None


def finalize(process: subprocess.Popen) -> None:
    """인코딩을 완료하고 FFmpeg 프로세스를 종료한다."""
    process.stdin.close()
    process.wait()
    stderr = _read_stderr(process)
    if process.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 인코딩 실패 (exit code {process.returncode})\n"
            f"── FFmpeg 에러 로그 ──\n{stderr or '(출력 없음)'}"
        )
