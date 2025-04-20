## Abstract
- Summarize the project: motivation, methods, findings, etc. 

## Introduction
In recent years, the application of algorithmic trading strategies has become increasingly prominent in financial markets. This project aims to investigate the effectiveness of a rule-based technical trading strategy, leveraging well-known indicators such as Relative Strength Index (RSI), and Bollinger Bands (BB) to predict trend reversals and generate trading signals.

### Hypothesis
- Financial markets are often noisy and volatile, making it difficult to identify trend reversals with confidence. Traders frequently rely on technical indicators to assist in decision-making. However, the effectiveness of these indicators—particularly in combination—remains an open question.
- This project explores the hypothesis about golden crosses and death crosses, which are widely recognized in technical analysis. The project aims to determine whether these indicators can reliably predict trend reversals and generate profitable trading signals.
### Methodology for Testing the Hypothesis
To test this hypothesis, a backtesting framework is implemented using Python. The strategy involves:
1. Loading historical price data from a CSV file.
2. Calculating the RSI, and BB indicators.
3. Using RSI and BB conditions to confirm trade entries and exits.
4. Executing trades in a simulated environment and tracking performance metrics such as capital growth, win rate, and drawdown.
The backtesting is conducted over historical intraday market data, and results are analyzed to determine the effectiveness of the proposed strategy.
## Related Work (or Background)
### Relative Strength Index (RSI)
The Relative Strength Index (RSI) is a momentum oscillator developed by J. Welles Wilder. It measures the speed and change of price movements and is used to identify overbought or oversold conditions in a market. The RSI oscillates between 0 and 100, with the following thresholds commonly interpreted:
- An RSI above 70 indicates that a security is overbought and may be due for a price correction.
- An RSI below 30 indicates that a security is oversold and may be due for a price increase.
### Bollinger Bands (BB)
Bollinger Bands are a type of volatility indicator created by John Bollinger. They consist of three components:
1. A simple moving average (SMA) of the closing prices.
2. An upper band, which is the SMA plus a specified number of standard deviations.
3. A lower band, which is the SMA minus a specified number of standard deviations.
Bollinger Bands are used to identify overbought or oversold conditions, as well as potential trend reversals. When the price approaches the upper band, it may indicate that the security is overbought, while a price near the lower band may suggest that it is oversold.
### Integration of RSI and BB
-   The methodology intentionally combines momentum (RSI) and volatility (BB) indicators to improve the robustness of trade signals. While RSI helps identify potential reversal points based on market psychology, Bollinger Bands complement this by confirming whether the price is at a statistically significant extreme.
-   By using both indicators in conjunction, the strategy avoids relying solely on price movement or trend-following logic (e.g., moving averages), and instead seeks trades where both price behavior and volatility context align.

## Trading (Algorithm) Hypotheses
- When the RSI value is extremely low (indicating oversold conditions) and the price touches the lower Bollinger Band, an upward price reversal is likely. Conversely, when the RSI is extremely high (indicating overbought conditions), a downward reversal is expected. The strategy also employs stop-loss and take-profit levels to manage risk and lock in gains.

## Data

### Data Source

The historical market data used in this project is collected from the **VN30F1M** (Vietnam's Futures Index) dataset, which contains **minute-level intraday price data**. This dataset reflects real trading activity in a liquid futures market, making it suitable for testing short-term trading strategies.

The data is either fetched through an external data API or read from a local CSV file (`daily_data.csv`) for reproducibility and offline processing. The `DataFetcher` class in the code handles this logic.

---

### Data Type

The dataset is **time-series financial data** and includes the following fields for each minute interval:

- `date`: Timestamp of the price record  
- `open`: Opening price  
- `high`: Highest price during the interval  
- `low`: Lowest price during the interval  
- `close`: Closing price  
- `volume`: Trading volume  

After indicators are calculated, additional fields are appended, such as:
- `RSI`: Relative Strength Index values  
- `BB_Upper`, `BB_Lower`: Bollinger Bands (Upper and Lower)  
- `SMA50`, `SMA200`: Simple Moving Averages (even though not used in final strategy)

---

### Data Period

The exact time span of the dataset can vary based on the CSV file used. From the file structure and example plots in the code, the data spans several months of **minute-level trading data**. This granular data is suitable for high-frequency or short-horizon strategy evaluation.

---

### How to Get the Input Data?

The data is initialized in the script via:

```python
backtest.initiate_data(use_csv=True)
```

## Implementation
- Briefly describe the implemetation.
    - How to set up the enviroment to run the source code and required steps to replicate the results
    - Discuss the concrete implementation if there are any essential details
    - How to run each step from `In-sample Backtesting`, Step 4 to `Out-of-sample Backtesting`, Step 6 (or `Paper Trading`, Step 7).
    - How to change the algorithm configurations for different run.
- Most important section and need the most details to correctly replicate the results.

## In-sample Backtesting
- Describe the In-sample Backtesting step
    - Parameters
    - Data
- Step 4 of the Nine-Step
### In-sample Backtesting Result
- Brieftly shown the result: table, image, etc.
- Has link to the In-sample Backtesting Report

## Optimization
- Describe the Optimization step
    - Optimization process/methods/library
    - Parameters to optimize
    - Hyper-parameter of the optimize process
- Step 5 of the Nine-Step
### Optimization Result
- Brieftly shown the result: table, image, etc.
- Has link to the Optimization Report

## Out-of-sample Backtesting
- Describe the Out-of-sample Backtesting step
    - Parameter
    - Data
- Step 6 of th Nine-Step
### Out-of-sample Backtesting Reuslt
- Brieftly shown the result: table, image, etc.
- Has link to the Out-of-sample Backtesting Report

## Paper Trading
- Describe the Paper Trading step
- Step 7 of the Nine-Step
- Optional
### Optimization Result
- Brieftly shown the result: table, image, etc.
- Has link to the Paper Trading Report


## Conclusion
- What is the conclusion?
- Optional

## Reference
- All the reference goes here.

## Other information
- Link to the Final Report (Paper) should be somewhere in the `README.md` file.
- Please make sure this file is relatively easy to follow.
