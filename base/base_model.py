import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with total and trainable parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        base_str = super().__str__()

        return (
            f"{base_str}\n"
            f"Total parameters: {total_params}\n"
            f"Trainable parameters: {trainable_params}\n"
            f"Non-trainable parameters: {non_trainable_params}"
        )
