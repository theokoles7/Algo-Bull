"""[SCRIPT] Initialize results.csv file."""

import csv

with open('experiments/results.csv', 'w') as file_out:

    writer = csv.writer(file_out)

    writer.writerow(['VARIABILITY', 'TICKER', 'EPOCHS', 'BEST ACCURACY', 'BEST EPOCH'])

    for variability in ['univariate', 'multivariate']:
        for ticker in ['AAPL', 'AMZN', 'GIB', 'MSFT', 'TSLA']:
            for epochs in ['100', '1000']:
                writer.writerow([variability, ticker, epochs, '--', '--'])