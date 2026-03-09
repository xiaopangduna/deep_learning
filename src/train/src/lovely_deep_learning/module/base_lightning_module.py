from abc import ABC, abstractmethod
import pytorch_lightning as pl
import lightning as L
from pathlib import Path
import torch_pruning as tp
import torch


class AbstractLightningModule(L.LightningModule, ABC):
    @abstractmethod
    def set_model(self):
        """
        This method sets the model for the module. Subclasses can choose to override this method to provide their own implementation.
        """
        print("This is the default implementation of set_model")
    @staticmethod
    def get_model(self):
        """
        This method sets the model for the module. Subclasses can choose to override this method to provide their own implementation.
        """
        print("This is the default implementation of set_model")

    def get_pruner(self):
        """
        This method sets the pruner for the module. Subclasses can choose to override this method to provide their own implementation.
        """
        print("This is the default implementation of set_pruner")

    def export(self):
        """
        This method sets the pruner for the module. Subclasses can choose to override this method to provide their own implementation.
        """
        print("This is the default implementation of set_pruner")

    def profile(self):
        """
        This method is used to perform profiling on the module.
        Subclasses can choose to override this method to provide their own implementation.
        """
        pass

