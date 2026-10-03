"""Выбор execution-провайдеров ONNX Runtime и резолвинг CUDA DLL (Windows).

CUDA используется автоматически, если доступна; при любых проблемах —
CPU-фолбэк (приложение не падает).
"""
import ctypes
import glob
import logging
import os

logger = logging.getLogger("app.ai.providers")

# DLL, от которых зависит onnxruntime_providers_cuda.dll (Windows)
_CUDA_DLLS = (
    "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_8.dll",
    "cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "cudnn64_9.dll",
)


def _candidate_dirs() -> list[str]:
    """Где ищем CUDA/cuDNN DLL: .env → bin/cuda проекта → venv (pip-колёса
    nvidia-*) → torch из соседнего окружения (например, ComfyUI) → тулкит."""
    from app.config import BASE_DIR, settings

    dirs: list[str] = []
    if settings.cuda_dll_path:
        dirs.append(settings.cuda_dll_path)
    dirs.append(str(BASE_DIR / "bin" / "cuda"))

    # pip-колёса nvidia-*-cu12/cu13 внутри venv
    for pattern in ("nvidia/*/bin", "nvidia/**/bin"):
        dirs.extend(glob.glob(os.path.join(os.sys.prefix, "Lib", "site-packages", pattern)))

    # чужие окружения с torch (ComfyUI и т.п.) — там лежат нужные DLL
    dirs.extend(glob.glob(
        os.path.expanduser(
            "~/AppData/Local/Comfy-Desktop/ComfyUI-Installs/*/ComfyUI/.venv/Lib/site-packages/torch/lib"
        )
    ))

    dirs.extend(glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin"))
    dirs.extend(glob.glob("C:/Program Files/NVIDIA/CUDNN/*/bin"))
    return dirs


def setup_cuda_dlls() -> None:
    """Подключает директории с CUDA-библиотеками ДО создания ORT-сессий.

    os.add_dll_directory не помогает зависимостям провайдера ORT, поэтому
    директории добавляются в PATH, а сами DLL предзагружаются через ctypes.
    Вызывать до первого InferenceSession.
    """
    if os.name != "nt":
        return
    for d in _candidate_dirs():
        if not os.path.isdir(d):
            continue
        found = [dll for dll in _CUDA_DLLS if os.path.isfile(os.path.join(d, dll))]
        if not found:
            continue
        logger.info("CUDA DLL: %s (%d библиотек)", d, len(found))
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        for dll in found:
            try:
                ctypes.WinDLL(os.path.join(d, dll))
            except OSError:
                pass  # битая/несовместимая — ORT сам сообщит при загрузке EP
        return  # достаточно первой директории с полным набором
    logger.info("CUDA DLL не найдены — при недоступности CUDA будет использован CPU")


def resolve_providers(device: str = "auto") -> list[str]:
    """CUDA при наличии, иначе CPU. Порядок = приоритет."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda" and not has_cuda:
        logger.warning("CUDA запрошена (AI_DEVICE=cuda), но недоступна — используется CPU")
        return ["CPUExecutionProvider"]
    if has_cuda:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def provider_label(providers: list[str]) -> str:
    return providers[0] if providers else "?"
