#!/bin/bash

#################################################
# Run univariate model on univariate stock data #
#################################################
# This will run the experiments with the
# following default parameters:
#   - Start date:       1900-01-01
#   - End date:         Day/Time of experiment
#   - Batch size:       32
#   - Learning rate:     0.001
#   - Look ahead:       30
#   - Hidden size:      64
#   - Layers:            2


for ticker in AAPL AMZN GIB MSFT TSLA
do
    for epochs in 100 1000
    do
        clear
        python main.py univariate --ticker $ticker --epochs $epochs
    done
done