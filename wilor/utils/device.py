import importlib.util

import torch


def import_musa_if_available() -> bool:
    if importlib.util.find_spec("torch_musa") is None:
        return False
    try:
        import torch_musa  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "musa")


def get_torch_device() -> torch.device:
    if import_musa_if_available():
        try:
            if torch.musa.is_available():
                return torch.device("musa")
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
