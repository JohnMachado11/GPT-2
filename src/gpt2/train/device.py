import torch


def pick_device():
    # the fastest thing available to train on: an NVIDIA GPU, then a Mac's GPU, then the CPU
    if torch.cuda.is_available():
        return "cuda"       # NVIDIA GPU (Colab, rented GPUs)
    if torch.backends.mps.is_available():
        return "mps"        # Apple GPU (Mac)
    return "cpu"


def supports_bf16(device):
    # bf16 pays off only on NVIDIA GPUs that do it in hardware: compute capability 8.0 and up (A100, L4,
    # H100, RTX 30xx/40xx). Older ones -- free Colab's T4 is 7.5 -- would only imitate it, slower than
    # plain 32-bit. Mac and CPU runs stay 32-bit too.
    return device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
