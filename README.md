# Group6

## How to Run the Code

### Step 1: Clone the Repository
```bash
git clone https://github.com/algotrade-course/Group6.git
```

### Step 2: Open the Folder
```bash
cd Group6
```

### Step 3: Create the new environment
```bash
python -m venv envgroup6
```

### Step 4: Execute the environment
```bash
source myenv/bin/activate
```

### Step 5: Execute the code
```
python main.py
```
After executing this command, if the environment lacks some packages for compiling, there would be an annoucement
```
The following required packages are missing: pandas, numpy, mplfinance, plotly, psycopg2-binary, optuna
Do you want to install them now? (y/n):
```
Click y to install. Then re-execute the code 

## Optimization
### Parameters
The following results were generated using Optuna libary after 100 trials with the following parameters: 
```
    period_bb = trial.suggest_int("period_bb", 20, 30, step=1)
    period_rsi = trial.suggest_int("period_rsi", 5, 20, step=1)
    risk_per_trade = trial.suggest_float("risk_per_trade", 0.1, 0.5, step=0.1)
    rsi_oversold = trial.suggest_float("rsi_oversold", 5, 30, step=1)
    rsi_overbought = trial.suggest_float("rsi_overbought", 70, 90, step=1)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.3, step=0.05)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.3, step=0.05)

```

The sample size for in sample data with 80% of the full data. The optimization was executed to receive the highest returns with the number of trades at least 300 trades.


### Result
Recently, the set of parameters which results in the highest returns in in-sample data set is 
```

```
