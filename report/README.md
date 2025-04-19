## Abstract
- Summarize the project: motivation, methods, findings, etc. 

## Introduction
In recent years, the application of algorithmic trading strategies has become increasingly prominent in financial markets. This project aims to investigate the effectiveness of a rule-based technical trading strategy, leveraging well-known indicators such as Simple Moving Averages (SMA), Relative Strength Index (RSI), and Bollinger Bands (BB), to predict trend reversals and generate trading signals.

### Hypothesis
- Financial markets are often noisy and volatile, making it difficult to identify trend reversals with confidence. Traders frequently rely on technical indicators to assist in decision-making. However, the effectiveness of these indicators—particularly in combination—remains an open question.
- This project explores the hypothesis about golden crosses and death crosses, which are widely recognized in technical analysis. The project aims to determine whether these indicators can reliably predict trend reversals and generate profitable trading signals.
### Methodology for Testing the Hypothesis
To test this hypothesis, a backtesting framework is implemented using Python. The strategy involves:
1. Loading historical price data from a CSV file.
2. Calculating the SMA50, SMA200, RSI, and BB indicators.
3. Detecting SMA crossovers to determine trend reversals.
4. Using RSI and BB conditions to confirm trade entries and exits.
5. Executing trades in a simulated environment and tracking performance metrics such as capital growth, win rate, and drawdown.
The backtesting is conducted over historical intraday market data, and results are analyzed to determine the effectiveness of the proposed strategy.
## Related Work (or Background)
- Prerequisite reading if the audience needs knowledge before exploring the project.
- Optional

## Trading (Algorithm) Hypotheses
- Describe the Trading Hypotheses
- Step 1 of the Nine-Step

## Data
- Data source
- Data type
- Data period
- How to get the input data?
- How to store the output data?
### Data collection
- Step 2 of the Nine-Step
### Data Processing
- Step 3 of the Nine-Step

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
