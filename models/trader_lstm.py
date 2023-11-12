"""Trader model and utilities."""

import torch, torch.nn as nn, torch.nn.functional as F

class TraderLSTM(nn.Module):
    """LSTM model."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        """Iniitalize Trader LSTM object.

        Args:
            input_size (int, optional): Number of features in the input data at each time step. Defaults to 1.
            hidden_size (int, optional): hidden units in LSTM layer. Defaults to 64.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
        """
        super(TraderLSTM, self).__init__()

        self.lstm =     nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear =   nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out, _ =    self.lstm(X)
        return      self.linear(out)