"""Diagnose COCO scratch stall: one val batch loss/fg/grads + a few SGD steps."""
from __future__ import annotations

import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lovely_deep_learning.dataset.object_detect import ObjectDetectDataset
from lovely_deep_learning.loss.object_detect import DetectionLossYOLOv8
from lovely_deep_learning.model.DAGNet import DAGNet
from lovely_deep_learning.module.base import _instantiate_lr_scheduler
from lightning.pytorch.cli import instantiate_class


def main() -> None:
    root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((root / "configs/experiments/object_detect_COCO.yaml").read_text())
    model_cfg = cfg["model"]["init_args"]["model"]
    data_cfg = cfg["data"]["init_args"]

    print("=== cat dtype check ===")
    a = torch.ones(2, 1).long()
    b = torch.ones(2, 1).long()
    c = torch.rand(2, 4)
    try:
        print("long+float cat", torch.cat((a, b, c), 1).dtype)
    except Exception as e:
        print("long+float cat failed:", e)

    print("=== SequentialLR 15 epochs ===")
    dummy = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(dummy.parameters(), lr=0.01)
    sch = _instantiate_lr_scheduler(opt, cfg["model"]["init_args"]["lr_scheduler"])
    print("epoch-1 (init)", opt.param_groups[0]["lr"])
    for i in range(15):
        sch.step()
        print(f"after step {i}", opt.param_groups[0]["lr"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    from torchvision.transforms.v2 import Compose, ToDtype
    from lovely_deep_learning.transforms.letterbox import LetterBox

    val_tf = Compose(
        [
            LetterBox(size=[640, 640], fill=114, center=True, scaleup=True, interpolation=2),
            ToDtype(dtype=torch.float32, scale=True),
        ]
    )
    ds = ObjectDetectDataset(
        csv_paths=[str(root / "datasets/COCO/val.csv")],
        key_map={"img_path": "path_img", "object_label_path": "path_label_detect_yolo"},
        transform=val_tf,
        mosaic_prob=0.0,
    )
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=ObjectDetectDataset.get_collate_fn_for_dataloader(),
    )
    net_in, net_out = next(iter(loader))
    imgs = ObjectDetectDataset.stack_batch_images(net_in).to(device)
    print("imgs", tuple(imgs.shape), "min/max/mean", float(imgs.min()), float(imgs.max()), float(imgs.mean()))

    n_gt = 0
    box_minmax = []
    for i, no in enumerate(net_out):
        boxes = no["bboxes_xyxy_abs_tv_transformed"]
        if hasattr(boxes, "data"):
            boxes = boxes.data
        cls = no["cls_tv_transformed"]
        n_gt += int(cls.shape[0])
        if boxes.numel():
            box_minmax.append(
                (
                    float(boxes[:, 0].min()),
                    float(boxes[:, 1].min()),
                    float(boxes[:, 2].max()),
                    float(boxes[:, 3].max()),
                    int(cls.shape[0]),
                )
            )
        print(
            f" sample{i} n={int(cls.shape[0])} batch_idx={'batch_idx' in no} "
            f"cls_max={int(cls.max()) if cls.numel() else None}"
        )
    print("box ranges (x1min,y1min,x2max,y2max,n)", box_minmax[:4], "n_gt", n_gt)

    net_out_gpu = []
    net_in_gpu = []
    for ni, no in zip(net_in, net_out):
        ni2 = dict(ni)
        ni2["img"] = ni["img"].to(device)
        net_in_gpu.append(ni2)
        no2 = dict(no)
        for k, v in no.items():
            if torch.is_tensor(v):
                no2[k] = v.to(device)
            elif hasattr(v, "data") and torch.is_tensor(v.data):
                no2[k] = type(v)(v.data.to(device), format=getattr(v, "format", None), canvas_size=getattr(v, "canvas_size", None)) if hasattr(v, "format") else v
            else:
                no2[k] = v
        net_out_gpu.append(no2)

    dag = DAGNet(**model_cfg).to(device)
    crit = DetectionLossYOLOv8(
        nc=80, reg_max=16, stride=[8, 16, 32], box_gain=7.5, cls_gain=0.5, dfl_gain=1.5, tal_topk=10
    )
    dag.train()
    preds = dag([imgs])
    print("preds type/len", type(preds), len(preds) if hasattr(preds, "__len__") else None)
    inner = preds[0] if isinstance(preds, tuple) else preds
    print("inner type", type(inner), "len", len(inner) if hasattr(inner, "__len__") else None)
    if isinstance(inner, (list, tuple)):
        for i, t in enumerate(inner):
            print(f"  feat{i}", tuple(t.shape), t.dtype, "reqgrad", t.requires_grad)

    vec = crit.forward_loss_vec(preds, net_out=net_out_gpu, net_in=net_in_gpu)
    print("loss vec box/cls/dfl", [float(x) for x in vec], "sum", float(vec.sum()))

    # fg via a second forward internals: decode assign
    feats = crit._feature_list(preds)
    print("feats after _feature_list", type(feats), len(feats) if not torch.is_tensor(feats) else feats.shape)

    opt = instantiate_class(
        filter(lambda p: p.requires_grad, dag.parameters()),
        {"class_path": "torch.optim.SGD", "init_args": {"lr": 0.01, "momentum": 0.937, "nesterov": True, "weight_decay": 5.0e-4}},
    )
    print("n param groups tensors", sum(len(g["params"]) for g in opt.param_groups))
    n_params = sum(p.numel() for p in dag.parameters() if p.requires_grad)
    print("n trainable", n_params)

    loss = vec.sum()
    loss.backward()
    gnorm = 0.0
    n_zero = 0
    n_none = 0
    n_ok = 0
    for p in dag.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            n_none += 1
            continue
        n_ok += 1
        g = p.grad.detach()
        gnorm += float(g.norm())
        if float(g.abs().max()) == 0:
            n_zero += 1
    print("grad: ok", n_ok, "none", n_none, "zero_max", n_zero, "sum_norms", gnorm)

    opt.zero_grad(set_to_none=True)
    print("=== 40 SGD steps same batch ===")
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        preds = dag([imgs])
        vec = crit.forward_loss_vec(preds, net_out=net_out_gpu, net_in=net_in_gpu)
        vec.sum().backward()
        opt.step()
        if step in (0, 1, 4, 9, 19, 39):
            print(f"step {step:02d} box/cls/dfl", [round(float(x), 3) for x in vec], "sum", round(float(vec.sum()), 3))


if __name__ == "__main__":
    main()
