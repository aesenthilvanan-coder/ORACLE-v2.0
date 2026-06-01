import os
import torch
import logging

logger = logging.getLogger(__name__)

# Required for numpy/torch on M1 Mac
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return best available compute device."""
    if torch.cuda.is_available():
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        logger.info("Using Apple MPS device")
        return torch.device("mps")
    logger.info("Using CPU device")
    return torch.device("cpu")


def move_batch(batch, device: torch.device):
    """Recursively move a dict/list/tensor batch to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [move_batch(x, device) for x in batch]
        return type(batch)(moved)
    return batch


def memory_summary(device: torch.device) -> str:
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
        return f"CUDA memory: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved"
    if device.type == "mps":
        return "MPS memory stats not available"
    return "CPU device — no memory stats"
