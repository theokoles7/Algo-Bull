"""Logger utilities."""

import datetime, logging, os, sys

from utils.arguments import ARGS

# Intialize logger
LOGGER = logging.getLogger("algo-bull")

# Set general logger level
LOGGER.setLevel(logging.getLevelName(ARGS.logging_level.upper()))

# Define console handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.getLevelName(ARGS.logging_level.upper()))
stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s: %(message)s'))
LOGGER.addHandler(stdout_handler)

# Verify that logging path exists
os.makedirs(f"{ARGS.logging_path}/{ARGS.ticker}/{ARGS.variability}/{ARGS.epochs}_epochs", exist_ok=True)

# Define file handler
file_handler = logging.FileHandler(f"{ARGS.logging_path}/{ARGS.ticker}/{ARGS.variability}/{ARGS.epochs}_epochs/trading-bot-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setLevel(logging.getLevelName(ARGS.logging_level.upper()))
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s: %(message)s'))
LOGGER.addHandler(file_handler)