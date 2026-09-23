"""YOLO 微批梯度求和 vs Lightning 默认 /accumulate。"""

from types import SimpleNamespace

import torch
from torch import nn

from lovely_deep_learning.callback.ema import ModelEMACallback


def test_detect_training_step_multiplies_loss_by_accumulate():
    """不实例化完整 YAML 模块：只复现 training_step 里对 backward loss 的缩放。"""
    acc = 4
    loss = torch.tensor(16.0, requires_grad=True)
    trainer = SimpleNamespace(accumulate_grad_batches=acc)
    backward = loss * max(int(getattr(trainer, "accumulate_grad_batches", 1) or 1), 1)
    assert float(backward.detach()) == 64.0


def test_ema_decay_ramp_matches_ultralytics():
    cb = ModelEMACallback(decay=0.9999, tau=2000)
    assert abs(cb.decay_at(1) - 0.9999 * (1 - __import__("math").exp(-1 / 2000))) < 1e-12
    assert cb.decay_at(10_000) > cb.decay_at(1)


def test_ema_update_moves_toward_student():
    torch.manual_seed(0)
    student = nn.Linear(2, 2)
    cb = ModelEMACallback(decay=0.5, tau=1)
    pl_module = SimpleNamespace(model=student)
    trainer = SimpleNamespace(sanity_checking=False, accumulate_grad_batches=1, is_last_batch=True)
    cb.on_fit_start(trainer, pl_module)  # type: ignore[arg-type]
    assert cb.ema is not None
    old = cb.ema.weight.detach().clone()
    with torch.no_grad():
        student.weight.fill_(3.0)
    cb.on_train_batch_end(trainer, pl_module, None, None, batch_idx=0)  # type: ignore[arg-type]
    assert not torch.equal(cb.ema.weight, old)
    assert torch.isfinite(cb.ema.weight).all()


def test_ema_swap_and_restore_roundtrip():
    torch.manual_seed(1)
    student = nn.Linear(1, 1)
    with torch.no_grad():
        student.weight.fill_(1.0)
    cb = ModelEMACallback(decay=0.0, tau=1)
    pl_module = SimpleNamespace(model=student)
    trainer = SimpleNamespace(sanity_checking=False, accumulate_grad_batches=1, is_last_batch=True)
    cb.on_fit_start(trainer, pl_module)  # type: ignore[arg-type]
    with torch.no_grad():
        student.weight.fill_(5.0)
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)  # type: ignore[arg-type]
    assert float(cb.ema.weight) == 5.0  # decay_at(1) with decay=0 → d=0, copy student
    with torch.no_grad():
        student.weight.fill_(9.0)
    cb.on_validation_start(trainer, pl_module)  # type: ignore[arg-type]
    assert float(student.weight) == 5.0
    cb.on_train_batch_start(trainer, pl_module, None, 0)  # type: ignore[arg-type]
    assert float(student.weight) == 9.0
