import optuna
import warnings
import numpy as np
from backtesting.backtesting import *

# warnings.filterwarnings("ignore")

def objectives(trial):
    in_sample_size = 0.8
    period_rsi = trial.suggest_int("period_rsi", 5, 20)
    period_bb = trial.suggest_int("period_bb", 10, 100)
    risk_per_trade = trial.suggest_float("risk_per_trade", 0.01, 0.5)
    rsi_oversold = trial.suggest_int("rsi_oversold", 5, 30)
    rsi_overbought = trial.suggest_int("rsi_overbought", 70, 95)

    backtest = Backtesting(
        period_rsi,
        period_bb,
        in_sample_size,
        risk_per_trade,
        rsi_oversold,
        rsi_overbought,
    )

    backtest.initiate_data(True)
    backtest.apply_indicators()

    sharpe_ratio = backtest.run_backtest(returns_sharpe=True, print_result=False)

    if sharpe_ratio is None or np.isnan(sharpe_ratio):
        return float("-inf")
    
    return sharpe_ratio

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objectives, n_trials=50)  

    print("Best hyperparameters:")
    print(study.best_params)
    print(f"Best Sharpe Ratio: {study.best_value}")