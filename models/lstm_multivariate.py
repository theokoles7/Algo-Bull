"""Multivariate LSTM model and utilities."""

import torch, torch.nn as nn
from tqdm   import tqdm

from data   import StockMultivariate
from utils  import ARGS, LOGGER

class LSTMMultivariate(nn.Module):
    """Multivariate LSTM model."""

    def __init__(self, num_classes: int, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        """Initialize Multivariate LSTM model.

        Args:
            num_classes (int): Number of classes in data set
            input_size (int, optional): Variability. Defaults to 1.
            hidden_size (int, optional): Hidden units in LSTM layer. Defaults to 64.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
        """
        super().__init__()

        self.num_classes =      num_classes
        self.num_layers =       num_layers
        self.input_size =       input_size
        self.hidden_size =      hidden_size

        # Define model
        self.lstm =             nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_1 =             nn.Linear(hidden_size, 128)
        self.fc_2 =             nn.Linear(128, num_classes)

        # Define loss function
        self.loss_func =        nn.MSELoss(reduction="mean")

        # Define optimizer
        self.optimizer =        torch.optim.Adam(self.parameters(), lr=ARGS.learning_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and provide output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Initialize hidden and cell states
        H =         torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        C =         torch.zeros(self.num_layers, X.size(0), self.hidden_size)

        X, (H, C) = self.lstm(X, (H, C))
        H =         H.view(-1, self.hidden_size)

        return self.fc_2(nn.ReLU(self.fc_1(nn.ReLU(H))))
    
    def execute(self, data: StockMultivariate) -> tuple[float, int]:
        """Execute model operations on provided data.

        Args:
            data (StockMultivariate): Multivariate stock dataset

        Returns:
            tuple[float, int]: [best_loss, best_epoch]
        """
        # Get data
        self.X_train, self.y_train, self.X_test, self.y_test = data.get_sets()
        for epoch in range(1, ARGS.epochs + 1):
            
            self.train_model(self.X_train, self.y_train)
    
    def train_model(self, pbar: tqdm = None) -> float:
        """Conduct training phase.

        Args:
            pbar (tqdm): Progress bar

        Returns:
            float: Average training loss
        """
        # Put model into training mode
        self.train()

        # Make predictions
        predictions = self(self.X_train)

        # Claculate loss
        self.optimizer.zero_grad()
        loss = self.loss_func(predictions, self.y_train)
        self.optimizer.step()

        return loss.item() / len(self.X_train)
    
    def validate_model(self, pbar: tqdm = None) -> float:
        """Conduct validation phase.

        Args:
            pbar (tqdm): Progress bar

        Returns:
            float: Average validation loss
        """
        # Put model into evaulation mode
        self.eval()

        # Make predictions
        predictions = self(self.X_test)

        # Calculate loss
        loss =          self.loss_func(predictions, self.y_test)

        return loss.item() / len(self.X_test)