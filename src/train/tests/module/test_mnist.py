import torch
from lovely_deep_learning.module.mnist import MNISTModule


class TestMNISTModule:

    def setup_method(self):
        self.model = MNISTModule()

    def test_forward(self):
        x = torch.randn(4, 1, 28, 28)

        logits = self.model(x)

        assert logits.shape == (4, 10)

    def test_training_step(self):
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))

        batch = (x, y)

        loss = self.model.training_step(batch, 0)

        assert loss is not None