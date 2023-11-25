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

    def get_sets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Provide training and test data sets.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Train & test data sets
        """
        return self.X_train, self.y_train, self.X_test, self.y_test