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

---

### Data Period

The exact time span of the dataset can vary based on the CSV file used. From the file structure and example plots in the code, the data spans several months of **minute-level trading data**. This granular data is suitable for high-frequency or short-horizon strategy evaluation.
At the moment we write this report, the dataset contains data from **2021-01-15** to **2025-03-20**.

---

### How to Get the Input Data?

The data is initialized in the script via:

```python
backtest.initiate_data(use_csv=True)
```

## 🛠️ Implementation

This section provides an in-depth overview of the system implementation, including environment setup, key modules, execution steps, and configuration options. The project is implemented in Python and follows a modular design that separates data handling, strategy logic, backtesting, and optimization.

---

### Overview of Implementation

The core functionality is divided into the following components:

- **`main.py`**: The entry point to the system with a menu interface to run backtesting or optimization.
- **`backtesting.py`**: Contains the `Backtesting` class which implements the trading logic, indicator calculations, trade execution, and performance evaluation.
- **`optimize.py`**: Uses the Optuna library to find optimal hyperparameters for the trading strategy.
- **`README.md`**: Provides step-by-step instructions for setting up and executing the project.

The system is built to support reproducibility and configurability through a consistent interface.

---

### Environment Setup

To replicate the environment and run the code, follow these steps:

```bash
# Step 1: Clone the repository
git clone https://github.com/algotrade-course/Group6.git
cd Group6

# Step 2: Create and activate a virtual environment
python -m venv envgroup6
source envgroup6/bin/activate  # On Windows: envgroup6\Scripts\activate

# Step 3: Run the code (it checks and installs missing packages automatically)
python main.py
```
Upon the first run, the script will check for required packages. If any are missing, the user is prompted to install them. After installation, the script should be re-run.
### Code structure and key modules
- **`backtesting.py`**: Contains the `Backtesting` class which implements the trading logic, indicator calculations, trade execution, and performance evaluation - Implements the main trading logic. Key features include:
    - **Indicator Calculation**: Computes RSI and Bollinger Bands
    - Signal generation based on indicator thresholds.
    - Risk management via stop-loss and take-profit.
    - Logging and export of trade data.
    - Visualization of returns and price data.
- **`main.py`**: The entry point to the system with a menu interface to run backtesting or optimization. Offers a CLI with options to:
    - Run in-sample backtesting.
    - Run backtesting without or with trading fee.
    - Run hyperparameter optimization.
    - Load optimized parameters and rerun the best strategy.
- **`optimize.py`**: Uses the Optuna library to find optimal hyperparameters for the trading strategy - Uses the Optuna framework to run a parameter search over:
    - `period_bb`: Period for Bollinger Bands
    - `period_rsi`: Period for RSI
    - `risk_per_trade`: Risk percentage per trade - total capital used for each trade
    - `rsi_oversold`: RSI threshold for oversold condition
    - `rsi_overbought`: RSI threshold for overbought condition
    - `stop_loss`: Stop-loss percentage
    - `take_profit`: Take-profit percentage
- **`evaluator.py`** : calculate indicators such as RSI and Bollinger Bands, including additional metrics for performance evaluation like Sharpe ratio, maximum drawdown, and win rate.
### Execution Flow:
#### In-sample Backtesting:
To run the backtest using predefined parameters:
```bash
python main.py
# Then select option 1 or 2 from the menu (with or without trading fee)
```
This run backtesting on the in-sample portion (80% of the data) and generates a report.
#### Optimization:
To run the optimization by Optuna:
```bash
python main.py
# Then select option 3 from the menu
# Then enter the number of trials (e.g., 200)
```
This will run the optimization process and save the best parameters in optimization_result.json.
#### Out-of-sample Backtesting:
To run the backtest using the optimized parameters:
```bash
python main.py
# Select option 3 to optimize → after it finishes, choose to run the backtest
# Choose one of the following:
# 1 → In-sample
# 2 → Out-of-sample
# 3 → All data
```
The script loads parameters from optimization_results.json and runs backtest accordingly.

#### Configuration and Customization:
You can customize the strategy directly by editing the main.py or using the menu.
## In-sample Backtesting

### Overview

The **in-sample backtesting** phase evaluates the performance of the trading strategy using a subset of historical data that the model has been trained or optimized on. This step helps validate whether the strategy's logic and parameters can generate favorable results under known market conditions.

In this project, 80% of the entire dataset is designated as the **in-sample data**, with the remaining 20% reserved for out-of-sample evaluation. This proportion is controlled via the `in_sample_size` parameter.

---

### Parameters

For this backtest, the following fixed parameters were used:

| Parameter         | Value | Description |
|-------------------|-------|-------------|
| `in_sample_size`  | 0.8   | 80% of the dataset is used for in-sample backtesting |
| `period_bb`       | 24    | Window length for Bollinger Bands |
| `period_rsi`      | 16    | Window length for RSI |
| `risk_per_trade`  | 0.1   | 10% of available capital allocated to each trade |
| `rsi_oversold`    | 13    | RSI threshold to detect oversold conditions |
| `rsi_overbought`  | 90    | RSI threshold to detect overbought conditions |
| `stop_loss`       | 0.15  | Trades are closed if a 15% loss is reached |
| `take_profit`     | 0.1   | Trades are closed if a 10% gain is reached | 

---

### Data

The in-sample backtesting uses the first 80% of the full minute-level price data from the `daily_data.csv` file. This data includes:

- **Timestamped price series**: `open`, `high`, `low`, `close`, `volume`
- **Computed technical indicators**:
  - `RSI` (Relative Strength Index) with a period of 6
  - `Bollinger Bands` (upper and lower) with a period of 21 and 2 standard deviations

The data is preprocessed using the `Backtesting.apply_indicators()` function, and the in-sample subset is saved to `data_in_sample.csv` after splitting.

---

### Backtesting Execution

To run the in-sample backtest:

```bash
python main.py
# Choose option 1 from the menu: Run Backtest (with trading fee)
# or option 2: Run Backtest (without trading fee)
```
The backtest will execute the strategy on the in-sample data, applying the defined parameters and logging trades. This run the following methods:
- `backtest.run_backtest(print_result=True)`
    - Splits the dataset into 80% in-sample and 20% out-of-sample
    - Applies the trading strategy
    - Tracks capital over time, trade entries/exits, and overall performance metrics
#### Logic Recap: 
- The strategy enters a long position when:
    - RSI is below the oversold threshold (5)
    - Price touches the lower Bollinger Band
- The strategy enters a short position when:
    - RSI is above the overbought threshold (71)
    - Price touches the upper Bollinger Band
- Positions are closed based on:
    - Stop-loss: 30% loss
    - Take-profit: 25% gain
    - Opposing signal (e.g., if a long position is open and the RSI crosses above the overbought threshold)

### In-sample Backtesting Result
| Metric                  | Value                  |
|-------------------------|------------------------|
| Initial Capital         | 1,000,000,000          |
| Final Capital           | 975,665,122.25         |
| Total Return            | -2.43%                 |
| Win Rate                | 55.56%                 |
| Max Drawdown            | 2.77%                  |
| Sharpe Ratio            | -0.1387                |
| Number of Transactions  | 324                    |

These results indicate that while the strategy was moderately successful in identifying profitable trades (with a **win rate above 50%**), it still resulted in an overall **net loss** during the in-sample period. The **Sharpe Ratio being negative** suggests that returns did not compensate for the volatility, meaning the strategy carried risk without consistent reward.

Despite the negative return, the relatively low drawdown and decent trade count imply the system maintains **risk control**, though the entry and exit rules may require further tuning to improve profitability.

These findings provide a baseline for comparison when evaluating **optimized parameters** and testing on **out-of-sample data**.

## 🧪 Optimization

### Overview

The optimization step is designed to automatically search for the best set of strategy parameters that yield the highest performance in terms of return. This is crucial for identifying configurations that outperform the baseline, especially when the strategy has multiple tunable components affecting entry, exit, and risk behavior.

---

### ⚙️ Optimization Process

The strategy parameters are optimized using **Optuna**, an open-source hyperparameter optimization framework. Optuna employs an efficient sampling algorithm called **Tree-structured Parzen Estimator (TPE)**, which intelligently explores the parameter space by learning from previous trials. This method balances exploration and exploitation better than traditional grid or random search approaches.

The optimization process is implemented in both `main.py` (interactive) and `optimize.py` (standalone). It follows these key steps:

1. Define the objective function (`objectives`) that:
   - Instantiates the `Backtesting` class with trial-specific parameters.
   - Runs the backtest on the **in-sample dataset**.
   - Returns the **total return** as the objective to maximize.
2. Create an Optuna study with `direction="maximize"`.
3. Run multiple trials to search for the best-performing configuration.

---

### 🎯 Parameters Optimized

The following parameters are subject to optimization:

| Parameter         | Range                          | Description |
|-------------------|--------------------------------|-------------|
| `period_bb`       | 20 to 30 (int)                 | Period of Bollinger Bands |
| `period_rsi`      | 5 to 20 (int)                  | RSI window length |
| `risk_per_trade`  | 0.1 to 0.5 (float, step 0.1)   | Capital risked per trade |
| `rsi_oversold`    | 5 to 30 (float)                | RSI threshold to trigger long entry |
| `rsi_overbought`  | 70 to 90 (float)               | RSI threshold to trigger short entry |
| `stop_loss`       | 0.05 to 0.3 (float, step 0.05) | Max loss before closing a trade |
| `take_profit`     | 0.05 to 0.3 (float, step 0.05) | Max gain before closing a trade |

These values are sampled in each trial and passed into the `Backtesting` instance for evaluation.

---

### 🔧 Hyperparameters of the Optimization Process

| Setting            | Value         | Description |
|--------------------|---------------|-------------|
| `n_trials`         | User-defined (e.g., 100 or 200) | Number of parameter combinations to test |
| `direction`        | `"maximize"`  | Optimization goal (maximize return) |
| `sampler`          | TPE Sampler   | Optuna's default sampler for efficient searching |
| `return metric`    | Total Return  | Output used to evaluate each configuration |

Example optimization execution from `main.py`:

```bash
python main.py
# Select option 3 → Enter number of trials (e.g., 100 or 200)
```

At the end of the process, Users are prompted to immediately re-run the backtest using the optimal parameters on in-sample, out-of-sample, or full dataset.
### Optimization Result
- The optimization process yields the following best parameters - which we mentioned above:

| Parameter         | Value | Description |
|-------------------|-------|-------------|
| `in_sample_size`  | 0.8   | 80% of the dataset is used for in-sample backtesting |
| `period_bb`       | 24    | Window length for Bollinger Bands |
| `period_rsi`      | 16    | Window length for RSI |
| `risk_per_trade`  | 0.1   | 10% of available capital allocated to each trade |
| `rsi_oversold`    | 13    | RSI threshold to detect oversold conditions |
| `rsi_overbought`  | 90    | RSI threshold to detect overbought conditions |
| `stop_loss`       | 0.15  | Trades are closed if a 15% loss is reached |
| `take_profit`     | 0.1   | Trades are closed if a 10% gain is reached | 

---


## Out-of-Sample Backtesting

### Overview

After identifying the best-performing parameters through in-sample backtesting and optimization, the next critical step is **out-of-sample backtesting**. This process evaluates the strategy's generalization ability by testing it on a **previously unseen portion of the data**.

In this project, 20% of the dataset is reserved as out-of-sample data. This segment is not used in any part of the optimization or training process, ensuring that the results reflect real-world performance more reliably.

---

### How It Works

Once optimization is complete, the user is prompted with the option to run the best parameters on:
- **In-sample data**
- **Out-of-sample data**
- **All data**

This interaction is handled via:

```python
run_backtest_from_optimized_params()
```
Which internally loads optimization_results.json and passes the best configuration to the Backtesting class. Then, based on user selection, it runs:
```python
backtest.run_backtest(
    print_result=True,
    all_sample=all_sample_flag,
    out_sample=out_sample_flag
)
```
This function executes the backtest on the selected dataset, applying the optimized parameters and logging trades. The results are then displayed in a similar format to the in-sample backtest.
### Purpose and Value
The out-of-sample backtesting serves several purposes:
- **Detect overfitting**: Parameters that perform well in-sample may exploit noise in the data.
- **Assess robustness**: If the strategy maintains consistent or improved performance, it shows promise for real-time application.
- **Validate optimization**: Good out-of-sample performance indicates the optimizer found a genuinely effective parameter set.
### Execution Steps
To run the out-of-sample backtest using the optimized parameters:
1. Start the script:
```bash
python main.py
```
2. Select option 3 from the menu to optimize (or skip this step if already done).
3. After optimization, choose to run the backtest.
4. Select option 2 for out-of-sample backtesting.
5. The script will execute the backtest on the out-of-sample data and display the results.
### Results Handling
- All trade logs, performance metrics, and capital growth plots are generated as part of the backtest. If enabled, trade records can be exported as:
    - `data_out_sample.csv` for out-of-sample data
    - `trades_output.csv` for trade logs

### Out-of-sample Backtesting Result
| Metric                  | Value                |
|-------------------------|----------------------|
| Final Capital           | 986,925,228.18       |
| Total Return            | -1.31%               |
| Win Rate                | 50.55%               |
| Max Drawdown            | 1.39%                |
| Sharpe Ratio            | -0.3119              |
| Number of Transactions  | 91                   |

#### 🔍 Interpretation

The out-of-sample results demonstrate a **mild negative return** of -1.31%, which, although better than the in-sample performance (-2.43%), still indicates the strategy did not generate consistent profitability on unseen data.

Some key insights:

- The **win rate (~50.5%)** suggests that the strategy was correct slightly more than half of the time, showing that trade direction predictions were not purely random.
- **Low drawdown (1.39%)** indicates **strong risk control**, meaning losses were contained even when profits were limited.
- The **Sharpe ratio is negative**, reflecting that returns did not justify the volatility taken on during the period.
- A relatively low number of trades (91) implies **lower market activity** or **stricter entry conditions** in the out-of-sample dataset.

#### 📌 Conclusion

While the strategy showed some consistency between in-sample and out-of-sample behavior in terms of win rate and drawdown, it failed to generate a positive return in both cases. This suggests the need for:
- Further tuning of risk-reward thresholds.
- Refinement of signal conditions.
- Possibly introducing additional filters (e.g., trend confirmation, volume) or ensemble approaches to increase robustness.

The out-of-sample test confirms that the strategy is **not severely overfitted**, but also **not yet profitable**, making it a stable yet underperforming baseline for future iterations.

## Reference
- All the reference goes here.

## Other information
- Link to the Final Report (Paper) should be somewhere in the `README.md` file.
- Please make sure this file is relatively easy to follow.
