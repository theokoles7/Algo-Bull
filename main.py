"""Drive application."""

import datetime, numpy as np, os, pandas as pd, torch, torch.nn as nn, traceback
from matplotlib                 import pyplot as plt
from sklearn.metrics            import accuracy_score
from termcolor                  import colored
from tqdm                       import tqdm
from torchmetrics.functional    import r2_score

from data.stock_univariate      import StockUnivariate
from data.stock_multivariate    import StockMultivariate
from models.trader_lstm_univariate         import TraderLSTMUnivariate
from utils                      import ARGS, LOGGER

if __name__ == '__main__':
    try:
        LOGGER.info(f"\n{open('utils/banner.txt', 'r').read()}")
        
        # Record arguments for logs
        for key in vars(ARGS):
            LOGGER.info(f"{key:<15}{ARGS.__getattribute__(key):>15}")

        # INITIALIZATION ==========================================================================

        # Initialize output directory
        output_dir = f"{ARGS.output_path}/{ARGS.ticker}/{ARGS.variables}/{ARGS.epochs}_epochs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        # Initialize dataset & model
        match ARGS.variables:
            case 'univariate': 
                data =  StockUnivariate(ARGS.ticker, ARGS.start_date, ARGS.end_date)
                model = TraderLSTMUnivariate(input_size=1, hidden_size=ARGS.hidden_size, num_layers=ARGS.layers, output_dir=output_dir)
            case 'multivariate': 
                data =  StockMultivariate(ARGS.ticker, ARGS.start_date, ARGS.end_date)

        # EXECUTION ===============================================================================

        # Execute on data
        model.execute(data)

    except KeyboardInterrupt:
        LOGGER.critical("Keyboard interrupt detected...")

    except Exception as e:
        LOGGER.error(e)
        traceback.print_exc()

    finally:
        LOGGER.info("Exiting...")