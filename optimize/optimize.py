import optuna
import warnings
import numpy as np
from backtesting.backtesting import *

# warnings.filterwarnings("ignore")

def objectives(trial):
    in_sample_size = 0.8
    period_bb = trial.suggest_int("period_bb", 20, 40, step = 1)
    period_rsi = trial.suggest_int("period_rsi", 5, 20, step = 1)
    risk_per_trade = trial.suggest_float("risk_per_trade", 0.1, 0.5, step = 0.1)
    rsi_oversold = trial.suggest_float("rsi_oversold", 5, 30, step = 1)
    rsi_overbought = trial.suggest_float("rsi_overbought", 70, 95, step = 1)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.3, step = 0.05)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.3, step = 0.05)

    backtest = Backtesting(
        period_rsi,
        period_bb,
        in_sample_size,
        risk_per_trade,
        rsi_oversold,
        rsi_overbought,
        stop_loss,
        take_profit,
    )

    backtest.initiate_data(True)
    backtest.apply_indicators()

    sharpe_ratio = backtest.run_backtest(returns_sharp=True, print_result=False)

    if sharpe_ratio is None or np.isnan(sharpe_ratio):
        return float("-inf")
    
    return sharpe_ratio

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objectives, n_trials=100)  

    print("Best hyperparameters:")
    print(study.best_params)
    print(f"Best Sharpe Ratio: {study.best_value}")