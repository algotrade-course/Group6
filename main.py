import optuna
import warnings
from backtesting.backtesting import Backtesting
import numpy as np
import json
import os

warnings.filterwarnings("ignore")

def objectives(trial):
    in_sample_size = 0.8
    period_bb = trial.suggest_int("period_bb", 20, 40, step=1)
    period_rsi = trial.suggest_int("period_rsi", 5, 20, step=1)
    risk_per_trade = trial.suggest_float("risk_per_trade", 0.1, 0.5, step=0.1)
    rsi_oversold = trial.suggest_float("rsi_oversold", 5, 30, step=1)
    rsi_overbought = trial.suggest_float("rsi_overbought", 70, 95, step=1)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.3, step=0.05)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.3, step=0.05)

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


def run_optimization():
    print("\nStarting Hyperparameter Optimization...\n")
    study = optuna.create_study(direction="maximize")
    study.optimize(objectives, n_trials=5)

    print("\n--- Optimization Complete ---")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print(f"Best Sharpe Ratio: {study.best_value:.6f}")

    # Prepare result dictionary
    result = {
        "best_params": study.best_params,
        "best_sharpe_ratio": study.best_value,
    }

    # Define file path
    result_path = os.path.join(os.getcwd(), "optimization_results.json")

    # Write to JSON file
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {result_path}")


def run_backtesting():
    print("\nRunning Backtest with Predefined Parameters...\n")
    
    in_sample_size = 0.8 # Percentage of data that used for the in sample test 
    period_bb = 21
    period_rsi = 6
    risk_per_trade = 0.1 # Percentage of total capital that used for each trade 
    rsi_oversold = 5
    rsi_overbought = 71
    stop_loss = 0.3
    take_profit = 0.25

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
    backtest.run_backtest(print_result=True)

def run_backtesting_no_fee():
    print("\nRunning Backtest with Predefined Parameters...\n")
    
    in_sample_size = 0.8 # Percentage of data that used for the in sample test 
    period_bb = 21
    period_rsi = 6
    risk_per_trade = 0.1 # Percentage of total capital that used for each trade 
    rsi_oversold = 5
    rsi_overbought = 71
    stop_loss = 0.3
    take_profit = 0.25

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
    backtest.run_backtest_no_fee(print_result=True)

def main_menu():
    while True:
        print("\n=== Trading Strategy Menu ===")
        print("1. Run Backtest")
        print("2. Run Backtest without fee")
        print("3. Optimize Strategy")
        print("4. Exit")
        choice = input("Choose an option (1-3): ")

        if choice == "1":
            run_backtesting()
        elif choice == "2":
            run_backtesting_no_fee()
        elif choice == "3":
            run_optimization()
        elif choice == "4":
            print ("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main_menu()
