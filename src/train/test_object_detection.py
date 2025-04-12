import pytest
from src.datasets.object_detection import DirectionalCornerDetectionDataset

class TestDirectionalCornerDetectionDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dataset = DirectionalCornerDetectionDataset()

    def test_getitem_valid_data_and_label(self):
        index = 0
        net_in, net_out = self.dataset[index]
        assert "img" in net_in
        assert "img_tensor" in net_in
        assert "target_raw" in net_out
        assert "target_valid" in net_out
        assert "target_tensor" in net_out

    def test_getitem_invalid_label(self):
        index = 1
        net_in, net_out = self.dataset[index]
        assert "img" in net_in
        assert "img_tensor" in net_in
        assert "target_raw" not in net_out
        assert "target_valid" not in net_out
        assert "target_tensor" not in net_out

    def test_getitem_no_label(self):
        index = 2
        net_in, net_out = self.dataset[index]
        assert "img" in net_in
        assert "img_tensor" in net_in
        assert "target_raw" not in net_out
        assert "target_valid" not in net_out
        assert "target_tensor" not in net_out
