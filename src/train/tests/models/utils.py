import yaml


def load_yaml_config(file_path):
    try:
        # 用with语句自动管理文件关闭
        with open(file_path, "r", encoding="utf-8") as f:
            # safe_load() 安全解析YAML内容
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except yaml.YAMLError as e:
        print(f"YAML解析错误：{e}")
        return None


resnet18_config = {
    "structure": {
        "inputs": [{"name": "input", "shape": [3, 224, 224]}],
        "outputs": [{"name": "classification", "from": ["fc"]}],
        "layers": [
            # Stem
            {
                "name": "conv1",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": 7,
                    "stride": 2,
                    "padding": 3,
                    "bias": False,
                },
                "from": ["input"],
            },
            {
                "name": "bn1",
                "module": "torch.nn.BatchNorm2d",
                "args": {"num_features": 64},
                "from": ["conv1"],
            },
            {
                "name": "relu",
                "module": "torch.nn.ReLU",
                "args": {"inplace": True},
                "from": ["bn1"],
            },
            {
                "name": "maxpool",
                "module": "torch.nn.MaxPool2d",
                "args": {"kernel_size": 3, "stride": 2, "padding": 1},
                "from": ["relu"],
            },
            # Layer1
            {
                "name": "layer1",
                "module": "torch.nn.Sequential",
                "from": ["maxpool"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 64, "out_channels": 64, "stride": 1},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 64, "out_channels": 64, "stride": 1},
                    },
                ],
            },
            # Layer2
            {
                "name": "layer2",
                "module": "torch.nn.Sequential",
                "from": ["layer1"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 64, "out_channels": 128, "stride": 2},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 128, "out_channels": 128, "stride": 1},
                    },
                ],
            },
            # Layer3
            {
                "name": "layer3",
                "module": "torch.nn.Sequential",
                "from": ["layer2"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 128, "out_channels": 256, "stride": 2},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 256, "out_channels": 256, "stride": 1},
                    },
                ],
            },
            # Layer4
            {
                "name": "layer4",
                "module": "torch.nn.Sequential",
                "from": ["layer3"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 256, "out_channels": 512, "stride": 2},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 512, "out_channels": 512, "stride": 1},
                    },
                ],
            },
            # Head
            {
                "name": "avgpool",
                "module": "torch.nn.AdaptiveAvgPool2d",
                "args": {"output_size": [1, 1]},
                "from": ["layer4"],
            },
            {
                "name": "flatten",
                "module": "torch.nn.Flatten",
                "args": {},
                "from": ["avgpool"],
            },
            {
                "name": "fc",
                "module": "torch.nn.Linear",
                "args": {"in_features": 512, "out_features": 1000},
                "from": ["flatten"],
            },
        ],
    },
    "weight": {
        "path": "pretrained_models/resnet18-f37072fd.pth",
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "map_location": "cpu",
        "strict": False,
    },
}


yolov8_n_config = {
    "structure": {
        "inputs": [{"name": "input", "shape": [3, 640, 640]}],
        "outputs": [{"name": "detect", "from": ["22"]}],
        "layers": [
            # Stem
            {
                "name": "0",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 3,
                    "c2": 16,
                    "k": 3,
                    "s": 2,
                },
                "from": ["input"],
            },
            {
                "name": "1",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 16,
                    "c2": 32,
                    "k": 3,
                    "s": 2,
                },
                "from": ["0"],
            },
            {
                "name": "2",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 32,
                    "c2": 32,
                    "n": 1,
                    "shortcut": True,
                },
                "from": ["1"],
            },
            {
                "name": "3",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 32,
                    "c2": 64,
                    "k": 3,
                    "s": 2,
                },
                "from": ["2"],
            },
            {
                "name": "4",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 64,
                    "c2": 64,
                    "n": 2,
                    "shortcut": True,
                },
                "from": ["3"],
            },
            {
                "name": "5",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 64,
                    "c2": 128,
                    "k": 3,
                    "s": 2,
                },
                "from": ["4"],
            },
            {
                "name": "6",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 128,
                    "c2": 128,
                    "n": 2,
                    "shortcut": True,
                },
                "from": ["5"],
            },
            {
                "name": "7",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 128,
                    "c2": 256,
                    "k": 3,
                    "s": 2,
                },
                "from": ["6"],
            },
            {
                "name": "8",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 256,
                    "c2": 256,
                    "n": 1,
                    "shortcut": True,
                },
                "from": ["7"],
            },
            {
                "name": "9",
                "module": "lovely_deep_learning.nn.block.SPPF",
                "args": {
                    "c1": 256,
                    "c2": 256,
                    "k": 5,
                },
                "from": ["8"],
            },
            {
                "name": "10",
                "module": "torch.nn.modules.upsampling.Upsample",
                "args": {
                    "size": None,
                    "scale_factor": 2,
                    "mode": "nearest",
                },
                "from": ["9"],
            },
            {
                "name": "11",
                "module": "ultralytics.nn.modules.conv.Concat",
                "args": {
                    "dimension": 1,
                },
                "from": ["10", "6"],
            },
            {
                "name": "12",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 384,
                    "c2": 128,
                    "n": 1,
                },
                "from": ["11"],
            },
            {
                "name": "13",
                "module": "torch.nn.modules.upsampling.Upsample",
                "args": {
                    "size": None,
                    "scale_factor": 2,
                    "mode": "nearest",
                },
                "from": ["12"],
            },
            {
                "name": "14",
                "module": "ultralytics.nn.modules.conv.Concat",
                "args": {
                    "dimension": 1,
                },
                "from": ["13", "4"],
            },
            {
                "name": "15",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 192,
                    "c2": 64,
                    "n": 1,
                },
                "from": ["14"],
            },
            {
                "name": "16",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 64,
                    "c2": 64,
                    "k": 3,
                    "s": 2,
                },
                "from": ["15"],
            },
            {
                "name": "17",
                "module": "ultralytics.nn.modules.conv.Concat",
                "args": {
                    "dimension": 1,
                },
                "from": ["16", "12"],
            },
            {
                "name": "18",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 192,
                    "c2": 128,
                    "n": 1,
                },
                "from": ["17"],
            },
            {
                "name": "19",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 128,
                    "c2": 128,
                    "k": 3,
                    "s": 2,
                },
                "from": ["18"],
            },
            {
                "name": "20",
                "module": "ultralytics.nn.modules.conv.Concat",
                "args": {
                    "dimension": 1,
                },
                "from": ["19", "9"],
            },
            {
                "name": "21",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 384,
                    "c2": 256,
                    "n": 1,
                },
                "from": ["20"],
            },
            {
                "name": "22",
                "module": "lovely_deep_learning.nn.head.Detect",
                "args": {
                    "nc": 80,
                    "ch": [64, 128, 256],
                    "legacy": True,
                    "stride": [8, 16, 32],
                    "shape": [136, 64, 80,80],
                },
                "from": ["15", "18", "21"],
            },
        ],
    },
    "weight": {
        "path": "pretrained_models/yolov8n.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "map_location": "cpu",
        "strict": False,
        "src_key_prefix": "layers.",
        "src_key_slice_start": 12,
    },
}

def _efficientnet_v2_s_layers(*, bn_eps: float = 1e-3, stochastic_depth_prob: float = 0.2):
    """
    Build EfficientNetV2-S layers config compatible with DAGNet.

    Mirrors TorchVision's EfficientNet(features/avgpool/classifier) naming so that
    `{"layers."+k: v for k,v in official.state_dict().items()}` can be loaded directly.
    """
    # From torchvision.models.efficientnet._efficientnet_conf("efficientnet_v2_s")
    stage_confs = [
        ("fused", 1, 3, 1, 24, 24, 2),
        ("fused", 4, 3, 2, 24, 48, 4),
        ("fused", 4, 3, 2, 48, 64, 4),
        ("mbconv", 4, 3, 2, 64, 128, 6),
        ("mbconv", 6, 3, 1, 128, 160, 9),
        ("mbconv", 6, 3, 2, 160, 256, 15),
    ]
    last_channel = 1280

    total_stage_blocks = sum(n for *_, n in stage_confs)
    stage_block_id = 0

    # features = [stem] + [stage1..stage6] + [lastconv]
    features_children = []

    # stem: Conv(3->24, k3 s2 p1) + BN(eps=1e-3) + SiLU
    features_children.append(
        {
            "name": "0",
            "module": "torch.nn.Sequential",
            "children": [
                {
                    "name": "0",
                    "module": "torch.nn.Conv2d",
                    "args": {
                        "in_channels": 3,
                        "out_channels": 24,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "bias": False,
                    },
                },
                {
                    "name": "1",
                    "module": "torch.nn.BatchNorm2d",
                    "args": {"num_features": 24, "eps": bn_eps},
                },
                {"name": "2", "module": "torch.nn.SiLU", "args": {"inplace": True}},
            ],
        }
    )

    # stages
    stage_idx = 1
    for block_type, expand_ratio, kernel, stride, in_ch, out_ch, num_layers in stage_confs:
        stage_children = []
        for layer_i in range(num_layers):
            block_in = in_ch if layer_i == 0 else out_ch
            block_stride = stride if layer_i == 0 else 1
            sd_prob = stochastic_depth_prob * float(stage_block_id) / float(total_stage_blocks)
            stage_block_id += 1

            if block_type == "fused":
                module = "lovely_deep_learning.nn.efficientnet_v2.FusedMBConv"
            else:
                module = "lovely_deep_learning.nn.efficientnet_v2.MBConv"

            stage_children.append(
                {
                    "name": str(layer_i),
                    "module": module,
                    "args": {
                        "input_channels": block_in,
                        "out_channels": out_ch,
                        "expand_ratio": expand_ratio,
                        "kernel": kernel,
                        "stride": block_stride,
                        "stochastic_depth_prob": sd_prob,
                        "bn_eps": bn_eps,
                    },
                }
            )

        features_children.append(
            {
                "name": str(stage_idx),
                "module": "torch.nn.Sequential",
                "children": stage_children,
            }
        )
        stage_idx += 1

    # lastconv: Conv(256->1280, k1 s1 p0) + BN(eps=1e-3) + SiLU
    features_children.append(
        {
            "name": str(stage_idx),
            "module": "torch.nn.Sequential",
            "children": [
                {
                    "name": "0",
                    "module": "torch.nn.Conv2d",
                    "args": {
                        "in_channels": 256,
                        "out_channels": last_channel,
                        "kernel_size": 1,
                        "stride": 1,
                        "padding": 0,
                        "bias": False,
                    },
                },
                {
                    "name": "1",
                    "module": "torch.nn.BatchNorm2d",
                    "args": {"num_features": last_channel, "eps": bn_eps},
                },
                {"name": "2", "module": "torch.nn.SiLU", "args": {"inplace": True}},
            ],
        }
    )

    layers = [
        {
            "name": "features",
            "module": "torch.nn.Sequential",
            "from": ["input"],
            "children": features_children,
        },
        {
            "name": "avgpool",
            "module": "torch.nn.AdaptiveAvgPool2d",
            "args": {"output_size": 1},
            "from": ["features"],
        },
        {
            "name": "flatten",
            "module": "torch.nn.Flatten",
            "args": {"start_dim": 1},
            "from": ["avgpool"],
        },
        {
            "name": "classifier",
            "module": "torch.nn.Sequential",
            "from": ["flatten"],
            "children": [
                {"name": "0", "module": "torch.nn.Dropout", "args": {"p": 0.2, "inplace": True}},
                {
                    "name": "1",
                    "module": "torch.nn.Linear",
                    "args": {"in_features": last_channel, "out_features": 1000},
                },
            ],
        },
    ]
    return layers


efficientnet_v2_s_config = {
    "structure": {
        "inputs": [{"name": "input", "shape": [3, 384, 384]}],
        "outputs": [{"name": "classification", "from": ["classifier"]}],
        "layers": _efficientnet_v2_s_layers(),
    },
    "weight": {
        "path": "pretrained_models/efficientnet_v2_s-dd5fe13b.pth",
        "url": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
        "map_location": "cpu",
        "strict": False,
        "src_key_prefix": "layers.",
        "src_key_slice_start": 0,
    },
}


def _swin_v2_t_layers(
    *,
    patch_size: list[int] = [4, 4],
    embed_dim: int = 96,
    depths: list[int] = [2, 2, 6, 2],
    num_heads: list[int] = [3, 6, 12, 24],
    window_size: list[int] = [8, 8],
    stochastic_depth_prob: float = 0.2,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    mlp_ratio: float = 4.0,
    norm_eps: float = 1e-5,
):
    total_stage_blocks = sum(depths)
    stage_block_id = 0

    features_children = []
    features_children.append(
        {
            "name": "0",
            "module": "lovely_deep_learning.nn.swin_v2.PatchEmbed",
            "args": {"embed_dim": embed_dim, "patch_size": patch_size, "norm_eps": norm_eps},
        }
    )

    feature_idx = 1
    for i_stage, stage_depth in enumerate(depths):
        dim = embed_dim * (2**i_stage)
        stage_children = []
        for i_layer in range(stage_depth):
            sd_prob = stochastic_depth_prob * float(stage_block_id) / float(total_stage_blocks - 1)
            shift_size = [0 if i_layer % 2 == 0 else w // 2 for w in window_size]
            stage_children.append(
                {
                    "name": str(i_layer),
                    "module": "lovely_deep_learning.nn.swin_v2.SwinTransformerBlockV2",
                    "args": {
                        "dim": dim,
                        "num_heads": num_heads[i_stage],
                        "window_size": window_size,
                        "shift_size": shift_size,
                        "mlp_ratio": mlp_ratio,
                        "dropout": dropout,
                        "attention_dropout": attention_dropout,
                        "stochastic_depth_prob": sd_prob,
                        "norm_eps": norm_eps,
                    },
                }
            )
            stage_block_id += 1

        features_children.append(
            {"name": str(feature_idx), "module": "torch.nn.Sequential", "children": stage_children}
        )
        feature_idx += 1

        if i_stage < len(depths) - 1:
            features_children.append(
                {
                    "name": str(feature_idx),
                    "module": "lovely_deep_learning.nn.swin_v2.PatchMergingV2",
                    "args": {"dim": dim, "norm_eps": norm_eps},
                }
            )
            feature_idx += 1

    layers = [
        {"name": "features", "module": "torch.nn.Sequential", "from": ["input"], "children": features_children},
        {
            "name": "norm",
            "module": "torch.nn.LayerNorm",
            "args": {"normalized_shape": embed_dim * 2 ** (len(depths) - 1), "eps": norm_eps},
            "from": ["features"],
        },
        {"name": "permute", "module": "torchvision.ops.misc.Permute", "args": {"dims": [0, 3, 1, 2]}, "from": ["norm"]},
        {"name": "avgpool", "module": "torch.nn.AdaptiveAvgPool2d", "args": {"output_size": 1}, "from": ["permute"]},
        {"name": "flatten", "module": "torch.nn.Flatten", "args": {"start_dim": 1}, "from": ["avgpool"]},
        {"name": "head", "module": "torch.nn.Linear", "args": {"in_features": embed_dim * 2 ** (len(depths) - 1), "out_features": 1000}, "from": ["flatten"]},
    ]
    return layers


swin_v2_t_config = {
    "structure": {
        "inputs": [{"name": "input", "shape": [3, 256, 256]}],
        "outputs": [{"name": "classification", "from": ["head"]}],
        "layers": _swin_v2_t_layers(),
    },
    "weight": {
        "path": "pretrained_models/swin_v2_t-b137f0e2.pth",
        "url": "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
        "map_location": "cpu",
        "strict": False,
        "src_key_prefix": "layers.",
        "src_key_slice_start": 0,
    },
}

demo_config_two_inputs_one_outputs = {
    "structure": {
        "inputs": [
            {"name": "img1", "shape": (3, 32, 32)},
            {"name": "img2", "shape": (3, 32, 32)},
        ],
        "outputs": [{"name": "class", "shape": (10,), "from": ["fc"]}],
        "layers": [
            {
                "name": "conv1",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 3,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "padding": 1,
                },
                "from": ["img1"],
            },
            {
                "name": "conv2",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 3,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "padding": 1,
                },
                "from": ["img2"],
            },
            {
                "name": "concat",
                "module": "lovely_deep_learning.nn.conv.Concat",
                "args": {"dim": 1},
                "from": ["conv1", "conv2"],
            },
            {
                "name": "conv3",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 32,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "padding": 1,
                },
                "from": ["concat"],
            },
            {
                "name": "flatten",
                "module": "torch.nn.Flatten",
                "args": {},
                "from": ["conv3"],
            },
            {
                "name": "fc",
                "module": "torch.nn.LazyLinear",
                "args": {"out_features": 10},
                "from": ["flatten"],
            },
        ],
    },
}
