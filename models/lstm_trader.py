"""Univariate LSTM model and utilities."""

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from matplotlib             import pyplot   as plt
from termcolor              import colored
from tqdm                   import tqdm

from data.stock  import Stock
from utils                  import ARGS, LOGGER

class LSTMTrader(nn.Module):
    """Univariate LSTM model."""

    logger = LOGGER.getChild('lstm')

    def __init__(self, output_dir: str, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        """Iniitalize Univariate LSTM object.

        Args:
            output_dir (str): Path to output (graphs) directory
            input_size (int, optional): Number of features in the input data at each time step. Defaults to 1.
            hidden_size (int, optional): Hidden units in LSTM layer. Defaults to 64.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
        """
        self.logger.info(f"Initializing model")

        super(LSTMTrader, self).__init__()

        self.input_size = input_size

        # Define model
        self.lstm =     nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear =   nn.Linear(hidden_size, input_size)

        # Place model on GPU if available
        self.device =   'cuda' if torch.cuda.is_available() else 'cpu'
        self =          self.to(self.device)
        self.logger.info(f"Using device: {self.device}")

        # Define loss function
        self.loss_func = nn.MSELoss(reduction="mean")

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=ARGS.learning_rate)

        # Note output directory
        self.output_dir = output_dir

        self.logger.info(f"Model initialization complete")
        self.logger.debug(f"Model:\n{self}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out, _ =    self.lstm(X)
        return      self.linear(out)
    
    def execute(self, data: Stock) -> tuple[float, int]:
        """Execute model operations on provided data.

        Args:
            data (StockUnivariate): Univariate stock dataset

        Returns:
            tuple[float, int]: [best_loss, best_epoch]
        """
        # Get data loaders
        self.train_data, self.test_data = data.get_loaders(ARGS.batch_size)
        self.logger.debug(f"Train loader:\n{self.train_data}")
        self.logger.debug(f"Test loader:\n{self.test_data}")

        # Note dates
        self.first_date =   data.first_date
        self.last_date =    data.last_date

        # Initialize metrics
        best_loss, best_epoch = 9999, 0
        train_history, val_history = [], []

        # Execute
        for epoch in range(1, ARGS.epochs + 1):

            # Initialize epoch metrics
            self.running_train_loss, self.running_val_loss = 0, 0

            # Initialize progress bar
            with tqdm(
                total =     len(self.train_data) + len(self.test_data), 
                desc =      f"Epoch {epoch:>4}/{ARGS.epochs}", 
                leave =     False, 
                colour =    'magenta'
            ) as pbar:

                # Train
                train_history.append(self.train_model(pbar))

                # Validate
                val_history.append(self.validate_model(pbar))

                # Update running metrics
                if val_history[-1] < best_loss:
                    best_loss, best_epoch = val_history[-1], epoch

                # Update progress bar
                pbar.set_postfix(status="Complete")

            # Log epoch metrics
            self.logger.info(f"EPOCH {epoch:4}/{ARGS.epochs:<4} | Train loss: {train_history[-1]:.8f} | Validation loss: {val_history[-1]:.8f}")

        # Log general execution metrics
        self.logger.info(f"Best loss score of {best_loss} @ epoch {best_epoch}")

        # Plot loss
        self.plot_loss(train_history, val_history)

        # # Convert to np-array
        # seq_to_plot = data.get_x_test().squeeze().cpu().numpy()

        # # Make forecast
        # forecast, combined_index = self.forecast(seq_to_plot, data.raw_test, data.last_date)

        # self.logger.debug(f"Seq_To_Plot shape: {np.shape(seq_to_plot)}")
        # self.logger.debug(f"Forecast shape: {np.shape(forecast)}")

        # # Gather forecast data for plotting
        # self.logger.debug(f"Forecast changed shape: {np.shape(np.expand_dims(forecast, axis=0))}")
        # if data.input_size > 1:
        #     original_cases =    data.scaler.inverse_transform(np.expand_dims(seq_to_plot[-1], axis=0)[0]).flatten()
        #     forecasted_cases =  data.scaler.inverse_transform(np.expand_dims(forecast, axis=0)).flatten()

        # else:
        #     original_cases =    data.scaler.inverse_transform(np.expand_dims(seq_to_plot[-1], axis=0)).flatten()
        #     forecasted_cases =  data.scaler.inverse_transform(np.expand_dims(forecast, axis=0)).flatten()

        # # Plot & save forecast
        # self.plot_forecast(data.raw_test, original_cases, forecasted_cases, combined_index, data.first_date, data.last_date)

        return round(best_loss, 8), best_epoch

    def forecast(self, seq_to_plot: torch.Tensor, raw_test: torch.Tensor, last_date: str) -> tuple[torch.Tensor, torch.Tensor]:

        self.logger.info(f"Making {ARGS.look_ahead} day forecast")

        # Grab last month of data
        historical_data = seq_to_plot[-1]

        # Initialize forecast
        forecast = []

        # Make forecast
        with tqdm(
            total =     ARGS.look_ahead * 2,
            desc =      colored("Forecasting", "green"),
            leave =     False,
            colour =    "magenta"
        ) as pbar, torch.no_grad():
            
            for i in range(ARGS.look_ahead * 2):

                # Prepare historical data
                history =               torch.as_tensor(historical_data).view(1, -1, self.input_size).float().to(self.device)

                # Make predictionsh
                prediction =            self(history).cpu().numpy()[0, 0]
                forecast.append(prediction[0])
                
                # Roll sequence
                historical_data =       np.roll(historical_data, shift=-1)
                historical_data[-1] =   prediction

                # Update progress bar
                pbar.update(1)

        # Generate & append future dates
        future_dates =                  pd.date_range(start=last_date + pd.DateOffset(1), periods=ARGS.look_ahead)
        combined_index =                raw_test.index.append(future_dates)

        return forecast, combined_index

    def plot_forecast(
            self, 
            raw_test: torch.Tensor, 
            original_cases: pd.DatetimeIndex, 
            forecasted_cases: np.ndarray, 
            combined_index: torch.Tensor,
            first_date: str,
            last_date: str
        ) -> None:
        """Plot and save forecast.

        Args:
            raw_test (torch.Tensor): Origina, raw date sequence
            original_cases (pd.DatetimeIndex): Original variable sequence
            forecasted_cases (np.ndarray): Forecasted sequence
            combined_index (torch.Tensor): Combined date index
            first_date (str): First data of sequence
            last_date (str): Last date of sequence
        """
        # Clear plot
        plt.clf()

        # Plot forecast
        plt.rcParams['figure.figsize'] = [14, 4]

        plt.plot(raw_test.index[-100:-30], raw_test.Open[-100:-30], label="Test Data",     color="blue")
        plt.plot(raw_test.index[-30:],     original_cases[-30:],          label="Actual Values", color="green")
        plt.plot(combined_index[-60:],     forecasted_cases,        label="Forecast",      color="red")

        plt.xlabel("Time Step")
        plt.ylabel("Value (USD)")
        plt.legend()
        plt.grid(True)
        plt.title(f"${ARGS.ticker} ({first_date} - {last_date})")
        plt.savefig(f"{self.output_dir}/forecast.jpg")

        self.logger.info(f"Forecast plot saved to {self.output_dir}/forecast.jpg")

    def plot_loss(self, train_loss: list, val_loss: list) -> None:
        """Plot and save loss graph.

        Args:
            train_loss (list): List of training loss scores
            val_loss (list): List of validation loss scores
        """
        x = np.linspace(1, ARGS.epochs, ARGS.epochs)
        plt.plot(x, train_loss, scalex=True, label="Training Loss")
        plt.plot(x, val_loss,                label="Validation Loss")

        plt.title(f"${ARGS.ticker} ({self.first_date} - {self.last_date})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(f"{self.output_dir}/loss.jpg")
        self.logger.info(f"Loss plot saved to {self.output_dir}/loss.jpg")
        
    def train_model(self, pbar: tqdm) -> float:
        """Conduct training phase.

        Args:
            pbar (tqdm): Progress bar

        Returns:
            float: Average training loss
        """
        # Put model into training mode
        self.train()
        pbar.set_postfix(status=colored("Training", "cyan"))

        # Initialize metric
        running_loss = 0.0

        # Train
        for batch_x, batch_y in self.train_data:

            # Move data to GPU if available
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Make predictions
            self.logger.debug(f"Batch_X Input shape: {batch_x.shape}")
            predictions = self(batch_x)

            # Calculate loss
            self.logger.debug(f"Predictions shape: {predictions.shape}")
            self.logger.debug(f"Batch_Y shape: {batch_y.shape}")
            loss = self.loss_func(predictions, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            # Update progress bar
            pbar.update(1)

        # Return avg loss
        return running_loss / len(self.train_data)
    
    def validate_model(self, pbar: tqdm) -> float:
        """Conduct validation phase.

        Args:
            pbar (tqdm): Progress bar

        Returns:
            float: Average validation loss
        """
        # Put model into validation mode
        self.eval()
        pbar.set_postfix(status=colored("Validating", "yellow"))

        # Initialize metric
        running_loss = 0.0

        # Validate
        with torch.no_grad():

            for batch_x, batch_y in self.test_data:

                # Move data to GPU if available
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Make predictions
                predictions = self(batch_x)

                # Calculate loss
                loss = self.loss_func(predictions, batch_y)
                running_loss += loss.item()

                # Update progress bar
                pbar.update(1)

            # Return average loss
            return running_loss / len(self.test_data)

