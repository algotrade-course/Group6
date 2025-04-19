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

### Step 3: Install the Required Packages
```bash
pip install pandas
pip install mplfinance
pip install plotly
pip install psycopg2-binary
pip install optuna
```

### Step 4: Execute the Code
```bash
python main.py
```

### Step 5: Choose an Option
```
1. Run Backtest  
2. Run Backtest without fee  
3. Optimize Strategy
```

## Optimization
### Parameters
The following results were generated using Optuna libary after 100 trials with the following parameters: 
```
    period_bb = trial.suggest_int("period_bb", 20, 40, step=1)
    period_rsi = trial.suggest_int("period_rsi", 5, 20, step=1)
    risk_per_trade = trial.suggest_float("risk_per_trade", 0.1, 0.5, step=0.1)
    rsi_oversold = trial.suggest_float("rsi_oversold", 5, 30, step=1)
    rsi_overbought = trial.suggest_float("rsi_overbought", 70, 95, step=1)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.3, step=0.05)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.3, step=0.05)

```

The sample size for in sample data with 80% of the full data. The optimization was executed to receive the highest sharpe ratio.
```
    study = optuna.create_study(direction="maximize")
    study.optimize(objectives, n_trials=100)

    print("\n--- Optimization Complete ---")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print(f"Best Sharpe Ratio: {study.best_value:.6f}")

```


### Result



