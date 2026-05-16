import torch


def get_torch_device() -> torch.device:
    if hasattr(torch, "musa"):
        try:
            if torch.musa.is_available():
                return torch.device("musa")
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
