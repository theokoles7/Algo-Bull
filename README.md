# ALGO-BULL
Algorithmic Trading project.

## Dataset

The data set for this initial version of the project uses [yfinance](https://pypi.org/project/yfinance/) to pull data for a stock security, with full range of historical data available (i.e., from the time that the stock was introduced to the market to the current date).

Example set for AAPL:
```
                  Open        High         Low       Close   Adj Close     Volume
Date                                                                             
1980-12-12    0.128348    0.128906    0.128348    0.128348    0.099319  469033600
1980-12-15    0.122210    0.122210    0.121652    0.121652    0.094137  175884800
1980-12-16    0.113281    0.113281    0.112723    0.112723    0.087228  105728000
1980-12-17    0.115513    0.116071    0.115513    0.115513    0.089387   86441600
1980-12-18    0.118862    0.119420    0.118862    0.118862    0.091978   73449600
...                ...         ...         ...         ...         ...        ...
2023-11-06  176.380005  179.429993  176.210007  179.229996  178.994186   63841300
2023-11-07  179.179993  182.440002  178.970001  181.820007  181.580780   70530000
2023-11-08  182.350006  183.449997  181.589996  182.889999  182.649368   49340300
2023-11-09  182.960007  184.119995  181.809998  182.410004  182.169998   53763500
2023-11-10  183.970001  186.570007  183.529999  186.399994  186.399994   66133400

[10820 rows x 6 columns]
```

As the example shows, the set contains features for:
* Date: Dates, in chronological order
* Open: Opening price on corresponding trading date
* High: The highest price that the stock reached on the corresponding trading date
* Low: The lowest price that the stock reached on the corresponding trading date
* Close: Closing price on corresponding trading date
* Adj Close: Closing price, adjusted for corporate actions
* Volume: Traded volume during trading day

## Models

### Univariate LSTM

### Multivariate LSTM

---
## Implemented using:
* [yfinance](https://pypi.org/project/yfinance/)