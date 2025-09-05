import torch

from lovely_deep_learning.models.DAGNet import DAGNet


def test_DAGNet_demo_model_two_inputs_one_outputs():
    config = {
        "inputs": [
            {"name": "img1", "shape": (3, 32, 32)},
            {"name": "img2", "shape": (3, 32, 32)}
        ],
        "outputs": [
            {"name": "class", "shape": (10,),"from":["fc"]}
        ],
        "layers": [
            {"name": "conv1", "module": "torch.nn.Conv2d", "args": {"in_channels":3, "out_channels":16, "kernel_size":3, "padding":1}, "from":["img1"]},
            {"name": "conv2", "module": "torch.nn.Conv2d", "args": {"in_channels":3, "out_channels":16, "kernel_size":3, "padding":1}, "from":["img2"]},
            {"name": "concat", "module": "torch.concat", "args": {"dim":1}, "from":["conv1","conv2"]},
            {"name": "conv3", "module": "torch.nn.Conv2d", "args": {"in_channels":32, "out_channels":32, "kernel_size":3, "padding":1}, "from":["concat"]},
            {"name": "flatten", "module": "torch.nn.Flatten", "args": {}, "from":["conv3"]},
            {"name": "fc", "module": "torch.nn.Linear", "args": {"in_features":32*32*32, "out_features":10}, "from":["flatten"]}
        ]
    }

    model = DAGNet(config)

    x1 = torch.randn(1,3,32,32)
    x2 = torch.randn(1,3,32,32)

    out = model([x1, x2])
    
    assert out[0].shape == (1,10)


def test_DAGNet_equal_ResNet18():
    pass