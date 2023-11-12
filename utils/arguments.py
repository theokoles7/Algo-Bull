"""Parse and provide command line arguments."""

import argparse, datetime


parser = argparse.ArgumentParser(
    'algo-bull',
    description='Algorithmic trading bot'
)

# BEGIN ARGUMENTS =========================================================================

# Stock -----------------------------------------
stock = parser.add_argument_group('Stock')

stock.add_argument(
    '--ticker',
    type =      str,
    default =   "AAPL",
    help =      "Stock security to run model on (Defaults to AAPL)"
)

stock.add_argument(
    '--start_date',
    type =      str,
    default =   "1900-01-01",
    help =      "Start date of historical data (Format: YYYY-MM-DD) (Defaults to 1900-01-01)"
)

stock.add_argument(
    '--end_date',
    type =      str,
    default =   datetime.date.today().strftime("%Y-%m-%d"),
    help =      "End date of historical data (Format: YYYY-MM-DD) (Defaults to today's date)"
)

# Model -----------------------------------------
model = parser.add_argument_group('Model')

model.add_argument(
    '--look_ahead',
    type =      int,
    default =   30,
    help =      "Prediction look ahead window"
)

model.add_argument(
    '--hidden_size',
    type =      int,
    default =   64,
    help =      "Hidden units in LSTM layer (Defaults to 64)"
)

model.add_argument(
    '--layers',
    type =      int,
    default =   2,
    help =      "Number of LSTM layers (Defaults to 2)"
)

model.add_argument(
    '--learning_rate',
    type =      float,
    default =   0.1,
    help =      "Optimizer learning rate"
)

model.add_argument(
    '--batch_size',
    type =      int,
    default =   16,
    help =      "Data batch size"
)

model.add_argument(
    '--epochs',
    type =      int,
    default =   50,
    help =      "Number of epochs model will train for"
)

# LOGGING ---------------------------------------
logging = parser.add_argument_group('Logging')

logging.add_argument(
    '--logging_path',
    type =      str,
    default =   "logs",
    help =      "Path at which log files will be written (Defaults to ./logs/)"
)

logging.add_argument(
    '--logging_level',
    type =      str,
    choices =   ["none", "debug", "info", "warning", "error", "critical"],
    default =   "info",
    help =      "Maximum logging level (None > Debug > Info > Warning > Error > Critical)"
)

# OUTPUT
output = parser.add_argument_group('Output')

output.add_argument(
    '--output_path',
    type =      str,
    default =   "output",
    help =      "Path at which application will save output"
)

# END ARGUMENTS ===========================================================================

ARGS = parser.parse_args()