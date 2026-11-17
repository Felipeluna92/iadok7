"""Download dos modelos e criação das sessões ONNX Runtime (GPU quando disponível)."""
import os
import threading
import urllib.request
import zipfile
from pathlib import Path

import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("FACESWAP_MODELS", ROOT / "models"))

_FF = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
_BUFFALO = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# nome -> (url, arquivo dentro de MODELS_DIR)
MODELS = {
    "det_10g": (_BUFFALO, "buffalo_l/det_10g.onnx"),
    "w600k_r50": (_BUFFALO, "buffalo_l/w600k_r50.onnx"),
    "inswapper_128": (_FF + "inswapper_128.onnx", "inswapper_128.onnx"),
    "bisenet_resnet_34": (_FF + "bisenet_resnet_34.onnx", "bisenet_resnet_34.onnx"),
    "gfpgan_1.4": (_FF + "gfpgan_1.4.onnx", "gfpgan_1.4.onnx"),
    "gpen_bfr_512": (_FF + "gpen_bfr_512.onnx", "gpen_bfr_512.onnx"),
    "codeformer": (_FF + "codeformer.onnx", "codeformer.onnx"),
}

_lock = threading.Lock()
_sessions = {}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[modelos] baixando {url}")

    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {dest.name}: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    tmp.replace(dest)


def model_path(name: str) -> Path:
    url, rel = MODELS[name]
    path = MODELS_DIR / rel
    if path.exists():
        return path
    if url.endswith(".zip"):
        archive = MODELS_DIR / Path(url).name
        if not archive.exists():
            _download(url, archive)
        target = MODELS_DIR / Path(url).stem
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.endswith(".onnx"):
                    out = target / Path(member).name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(out, "wb") as dst:
                        dst.write(src.read())
        archive.unlink(missing_ok=True)
    else:
        _download(url, path)
    return path


def providers() -> list:
    available = ort.get_available_providers()
    chosen = []
    if "CUDAExecutionProvider" in available:
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()  # carrega CUDA/cuDNN instalados via pip
            except Exception:
                pass
        chosen.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "cudnn_conv_algo_search": "HEURISTIC",
            "arena_extend_strategy": "kSameAsRequested",
        }))
    if "DmlExecutionProvider" in available:
        chosen.append("DmlExecutionProvider")
    chosen.append("CPUExecutionProvider")
    return chosen


def session(name: str) -> ort.InferenceSession:
    with _lock:
        if name not in _sessions:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _sessions[name] = ort.InferenceSession(
                str(model_path(name)), sess_options=opts, providers=providers())
        return _sessions[name]


def active_device() -> str:
    p = providers()[0]
    name = p[0] if isinstance(p, tuple) else p
    return {"CUDAExecutionProvider": "GPU (CUDA)", "DmlExecutionProvider": "GPU (DirectML)"}.get(name, "CPU")
