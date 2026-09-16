"""SequentialLR 子调度器须绑到同一 optimizer。"""

import torch
from torch import nn

from lovely_deep_learning.module.base import _instantiate_lr_scheduler


def test_instantiate_sequential_lr_binds_nested_linear_lr():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = _instantiate_lr_scheduler(
        optimizer,
        {
            "class_path": "torch.optim.lr_scheduler.SequentialLR",
            "init_args": {
                "schedulers": [
                    {
                        "class_path": "torch.optim.lr_scheduler.LinearLR",
                        "init_args": {
                            "start_factor": 0.01,
                            "end_factor": 1.0,
                            "total_iters": 3,
                        },
                    },
                    {
                        "class_path": "torch.optim.lr_scheduler.LinearLR",
                        "init_args": {
                            "start_factor": 1.0,
                            "end_factor": 0.01,
                            "total_iters": 7,
                        },
                    },
                ],
                "milestones": [3],
            },
        },
    )
    assert type(scheduler).__name__ == "SequentialLR"
    assert abs(optimizer.param_groups[0]["lr"] - 1.0e-4) < 1e-9
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 1.0e-4
