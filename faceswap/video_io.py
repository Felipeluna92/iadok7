"""Leitura de vídeo (OpenCV) e escrita em alta qualidade via FFmpeg, preservando o áudio."""
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_frames(path: str, start: int = 0, end: int = None):
    cap = cv2.VideoCapture(str(path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while end is None or idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    def __init__(self, out_path: str, width: int, height: int, fps: float, crf: int = 16,
                 preset: str = "slow", audio_from: str = None):
        self.out_path = Path(out_path)
        self.audio_from = audio_from
        self._tmp = Path(tempfile.mkstemp(suffix=".mp4", dir=self.out_path.parent)[1])
        self._log = tempfile.TemporaryFile()
        cmd = [FFMPEG, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self._tmp)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)

    def write(self, frame) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> Path:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            self._log.seek(0)
            raise RuntimeError("FFmpeg falhou: " + self._log.read().decode(errors="ignore"))
        if self.audio_from:
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(self._tmp), "-i", str(self.audio_from),
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                   "-shortest", "-movflags", "+faststart", str(self.out_path)]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                self._tmp.unlink(missing_ok=True)
                return self.out_path
        self._tmp.replace(self.out_path)
        return self.out_path
