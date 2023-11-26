"""Drive application."""

import datetime, numpy as np, os, pandas as pd, torch, torch.nn as nn, traceback
from matplotlib                 import pyplot as plt
from sklearn.metrics            import accuracy_score
from termcolor                  import colored
from tqdm                       import tqdm
from torchmetrics.functional    import r2_score

from data                       import StockUnivariate, StockMultivariate
from models                     import LSTMUnivariate, LSTMMultivariate
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

        data =  StockUnivariate(ARGS.variables, ARGS.ticker, ARGS.start_date, ARGS.end_date)
        model = LSTMUnivariate(input_size=1, hidden_size=ARGS.hidden_size, num_layers=ARGS.layers, output_dir=output_dir)

        # # Initialize dataset & model
        # match ARGS.variables:
        #     case 'univariate': 
        #         data =  StockUnivariate(ARGS.ticker, ARGS.start_date, ARGS.end_date)
        #         model = LSTMUnivariate(input_size=1, hidden_size=ARGS.hidden_size, num_layers=ARGS.layers, output_dir=output_dir)
        #     case 'multivariate': 
        #         data =  StockMultivariate(ARGS.ticker, ARGS.start_date, ARGS.end_date)
        #         model = LSTMMultivariate(num_classes=50, input_size=4, hidden_size=2, num_layers=1, output_dir=output_dir)

        LOGGER.debug(f"Dataset:\n{data}")
        LOGGER.debug(f"Model:\n{model}")

        # EXECUTION ===============================================================================

        # Execute on data
        loss, epoch = model.execute(data)

        # Record results
        LOGGER.info("Recording results")
        results_file = pd.read_csv('experiments/results.csv')
        results_file.loc[
            (results_file['VARIABILITY']==ARGS.variables) &
            (results_file['TICKER']==ARGS.ticker) &
            (results_file['EPOCHS']==ARGS.epochs),
            ['BEST ACCURACY', 'BEST EPOCH'] 
        ] = loss, epoch

        results_file.to_csv('experiments/results.csv', index=False)

    except KeyboardInterrupt:
        LOGGER.critical("Keyboard interrupt detected...")

    except Exception as e:
        LOGGER.error(e)
        traceback.print_exc()

    finally:
        LOGGER.info("Exiting...")