from abc import ABC, abstractmethod


class BaseMetrics(ABC):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self,cfgs:dict):
        self.metrics = None  # 字典，所有可能考虑的指标
        self.metric = None  # 标量，最关注的指标
        pass

    @abstractmethod
    def update_metrics_batch(self):
        return

    @abstractmethod
    def update_metrics_epoch(self):
        """_summary_
        """
        pass

    @abstractmethod
    def get_report(self):
        pass

    @abstractmethod
    def reset_metrics(self):
        pass
    
    def get_metric(self):
        return self.metric

    def get_dict_metrics(self):
        return self.metrics
