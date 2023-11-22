"""Dataset class and utilities."""

import datetime, math

import numpy as np, torch, yfinance as yf
from sklearn.preprocessing  import MinMaxScaler, StandardScaler
from torch.utils.data       import DataLoader, TensorDataset

from utils.logger           import LOGGER

class StockMultivariate():
    """Dataset class."""

    logger = LOGGER.getChild('stock-data')

    def __init__(self, ticker: str, start_date: datetime.date | str = "1900-01-01", end_date: datetime.date | str = datetime.date.today().strftime("%Y-%m-%d")):
        """Initialize Stock object.

        Args:
            ticker (str): Ticker/stock symbol (i.e., AAPL, GOOG, AMZN)
            start_date (date | str, optional): Beginning of historical data being pulled. Defaults to "1900-01-01".
            end_date (date | str, optional): End of historical data. Defaults to today's date.
        """
        # Download stock data
        self.logger.info(f"Downloading {ticker} from {start_date} to {end_date}")

        self.data = yf.download(
            ticker, 
            start = start_date, 
            end =   end_date
        )

        X, y = self.data.drop(columns=['Close']), self.data.Close.values

        ss = StandardScaler()
        mm = MinMaxScaler()

        X = ss.fit_transform(X)
        y = mm.fit_transform(y.reshape(-1, 1))

        X, y = self.split_seq(X, y, 100, 50)

        X_train =   X[:math.ceil(len(X) * .8)]
        X_test =    X[math.ceil(len(X) * .8):]

        self.y_train =   y[:math.ceil(len(X) * .8)]
        self.y_test =    y[math.ceil(len(X) * .8):]

        self.X_train = torch.reshape(
            torch.Tensor(X_train),
            (X_train.shape[0], 100,
             X_train.shape[2])
        )

        self.X_test = torch.reshape(
            torch.Tensor(X_test),
            (X_test.shape[0], 100,
             X_test.shape[2])
        )

        print(f"X_train: {self.X_train.shape}")
        print(f"X_test: {self.X_test.shape}")
        print(f"y_train: {self.y_train.shape}")
        print(f"y_test: {self.y_test.shape}")

    def split_seq(self, input_seq, output_seq, steps_in, steps_out) -> tuple[np.array, np.array]:
        X, y = list(), list()

        for i in range(len(input_seq)):

            in_end =    i + steps_in
            out_end =   in_end + steps_out - 1

            if out_end > len(input_seq): break

            x_seq, y_seq = input_seq[i: in_end], output_seq[in_end - 1:out_end, -1]

            X.append(x_seq), y.append(y_seq)

        return np.array(X), np.array(y)

        # self.logger.debug(f"Downloaded data:\n{self.data}")

        # # Split data into training and testing sets
        # self.logger.info("Splitting data into train and test sets")

        # self.train_data =   self.data[:math.ceil(len(self.data) * .8)].iloc[:,:1]
        # self.test_data =    self.data[math.ceil(len(self.data) * .8):].iloc[:,:1]

        # # Store raw data
        # self.raw_train = self.train_data
        # self.raw_test =  self.test_data

        # # Record first and last dates
        # self.first_date = self.test_data.index[0]
        # self.last_date =  self.test_data.index[-1]

        # self.logger.debug(f"Train data:\n{self.train_data}")
        # self.logger.debug(f"Test data:\n{self.test_data}")

        # # Reshape
        # self.logger.info("Reshaping sets")

        # self.train_data =   np.reshape(self.train_data.Open.values, (-1, 1))
        # self.test_data =    np.reshape(self.test_data.Open.values, (-1, 1))

        # self.logger.debug(f"Training set shape: {self.train_data.shape}")
        # self.logger.debug(f"Testing set shape: {self.test_data.shape}")

        # # Normalization
        # self.logger.info("Normalizing data")
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.train_data =   self.scaler.fit_transform(self.train_data)
        # self.test_data =    self.scaler.fit_transform(self.test_data)

        # self.logger.debug(f"Scaled train set:\n{self.train_data}")
        # self.logger.debug(f"Scaled test set:\n{self.test_data}")

        # # Sequencing
        # self.logger.info("Converting sets to sequences")

        # self.X_train, self.y_train, self.X_test, self.y_test = [], [], [], []

        # for i in range(len(self.train_data) - 50):
        #     self.X_train.append(self.train_data[i:i+50])
        #     self.y_train.append(self.train_data[i+1:i+50+1])

        # for i in range(len(self.test_data) - 30):
        #     self.X_test.append(self.test_data[i:i+30])
        #     self.y_test.append(self.test_data[i+1:i+30+1])

        # self.X_train, self.y_train, self.X_test, self.y_test = (
        #     np.array(self.X_train), 
        #     np.array(self.y_train), 
        #     np.array(self.X_test), 
        #     np.array(self.y_test)
        # )

        # self.X_train, self.y_train, self.X_test, self.y_test = (
        #     torch.tensor(self.X_train, dtype=torch.float32), 
        #     torch.tensor(self.y_train, dtype=torch.float32), 
        #     torch.tensor(self.X_test,  dtype=torch.float32), 
        #     torch.tensor(self.y_test,  dtype=torch.float32)
        # )

        # self.logger.debug(f"X_train.shape: {self.X_train.shape}")
        # self.logger.debug(f"y_train.shape: {self.y_train.shape}")
        # self.logger.debug(f"X_test.shape: {self.X_test.shape}")
        # self.logger.debug(f"y_test.shape: {self.y_test.shape}")

    def get_last_date(self) -> str:
        """Provide last date in sequence.

        Returns:
            str: Last date
        """
        return self.last_date

    def get_loaders(self, batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
        """Provide train and test data loaders.

        Args:
            batch_size (int, optional): Loader batch size. Defaults to 16.

        Returns:
            tuple[DataLoader, DataLoader]: Train and test data loaders
        """
        self.logger.info("Creating train and test loaders")
        
        return [
            DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(self.X_test,  self.y_test),  batch_size=batch_size, shuffle=False)
        ]
    
    def get_x_test(self) -> torch.Tensor:
        """Provide X_test tensor.

        Returns:
            torch.Tensor: X_test tensor
        """
        return self.X_test