import pytest
import torch

from gpt2.train.device import pick_device, supports_bf16


@pytest.mark.parametrize("cuda, mps, expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_pick_device_prefers_nvidia_then_mac_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert pick_device() == expected


@pytest.mark.parametrize("capability, expected", [((7, 5), False), ((8, 0), True), ((8, 9), True), ((9, 0), True)])
def test_bf16_only_on_nvidia_gpus_that_do_it_in_hardware(monkeypatch, capability, expected):
    # 7.5 is free Colab's T4; 8.0 the A100, 8.9 the L4, 9.0 the H100
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    assert supports_bf16("cuda") is expected


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_no_bf16_on_a_mac_or_the_cpu(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))    # even beside an H100
    assert supports_bf16(device) is False
