"""
Video Module for uncertainty weighting.
"""

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """Learnable loss weights (Kendall et al., 2018)."""

    def __init__(self, num_tasks: int = 2):
        """
        Initialize base video classification model.

        :param num_tasks: Number of tasks.
        """
        super().__init__()
        self._log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses) -> torch.Tensor:
        """
        Forward pass for learnable loss weights.

        :param losses: Losses.
        :return: Total loss value.
        """
        total = 0.0
        for i, loss in enumerate(losses):
            if loss is None:
                continue
            log_var = self._log_vars[i].clamp(-5, 5)
            precision = torch.exp(-log_var)
            total += 0.5 * precision * loss + 0.5 * log_var

        if total is None:
            total = torch.tensor(0.0, device=self._log_vars.device, requires_grad=True)

        return total
