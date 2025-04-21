import sys
import subprocess  # Add this import
import warnings
import os
import json

# Mapping of import names to pip install names
required_packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "mplfinance": "mplfinance",
    "plotly": "plotly",
    "psycopg2": "psycopg2-binary",  # correct import name -> pip install name
    "optuna": "optuna"
}

def check_and_install_packages():
    missing = []
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"\nThe following required packages are missing: {', '.join(missing)}")
        choice = input("Do you want to install them now? (y/n): ").strip().lower()
        if choice == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("\nAll missing packages installed successfully. Please rerun the script.")
            sys.exit(0)
        else:
            print("\nCannot proceed without installing the required packages. Exiting...")
            sys.exit(1)

check_and_install_packages()

import optuna
from backtesting.backtesting import Backtesting
import numpy as np


warnings.filterwarnings("ignore")

def objectives(trial):
    in_sample_size = 0.8
    period_bb = trial.suggest_int("period_bb", 20, 30, step=1)
    period_rsi = trial.suggest_int("period_rsi", 5, 20, step=1)
    risk_per_trade = trial.suggest_float("risk_per_trade", 0.1, 0.5, step=0.1)
    rsi_oversold = trial.suggest_float("rsi_oversold", 5, 30, step=1)
    rsi_overbought = trial.suggest_float("rsi_overbought", 70, 90, step=1)
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

    total_return = backtest.run_backtest(returns_total_return=True)
    if total_return is None or np.isnan(total_return):
        return float("-inf")
    return total_return

def run_optimization(n_trials):
    print("\nStarting Hyperparameter Optimization...\n")
    study = optuna.create_study(direction="maximize")
    study.optimize(objectives, n_trials)

    print("\n--- Optimization Complete ---")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print(f"Best Returns: {study.best_value:.6f}")

    result = {
        "best_params": study.best_params,
        "best_returns": study.best_value,
    }

    # Define file path
    result_path = os.path.join(os.getcwd(), "optimization_results.json")

    # Write to JSON file
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {result_path}")
    run_now = input("\nDo you want to run backtest with best parameters now? (y/n): ").lower()
    if run_now == 'y':
        run_backtest_from_optimized_params()

# Parameter for backtesting manually input
in_sample_size = 0.8 # Percentage of data that used for the in sample test 
period_bb = 21
period_rsi = 6
risk_per_trade = 0.1 # Percentage of total capital that used for each trade 
rsi_oversold = 5
rsi_overbought = 71
stop_loss = 0.3
take_profit = 0.25

def run_backtesting():
    print("\nRunning Backtest with Predefined Parameters...\n")

    # Ask user which dataset to use
    print("Which dataset do you want to run the backtest on?")
    print("1. In-sample data")
    print("2. Out-of-sample data")
    print("3. All data")
    dataset_choice = input("Enter choice (1/2/3): ").strip()

    in_sample_flag = False
    out_sample_flag = False
    all_sample_flag = False

    if dataset_choice == "1":
        in_sample_flag = True
    elif dataset_choice == "2":
        out_sample_flag = True
    elif dataset_choice == "3":
        all_sample_flag = True
    else:
        print("Invalid choice. Defaulting to in-sample data.")
        in_sample_flag = True

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
    backtest.run_backtest(
        print_result=True,
        all_sample=all_sample_flag,
        out_sample=out_sample_flag
    )

def run_backtesting_no_fee():
    print("\nRunning Backtest with Predefined Parameters...\n")

    # Ask user which dataset to use
    print("Which dataset do you want to run the backtest on?")
    print("1. In-sample data")
    print("2. Out-of-sample data")
    print("3. All data")
    dataset_choice = input("Enter choice (1/2/3): ").strip()

    in_sample_flag = False
    out_sample_flag = False
    all_sample_flag = False

    if dataset_choice == "1":
        in_sample_flag = True
    elif dataset_choice == "2":
        out_sample_flag = True
    elif dataset_choice == "3":
        all_sample_flag = True
    else:
        print("Invalid choice. Defaulting to in-sample data.")
        in_sample_flag = True

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
    backtest.run_backtest_no_fee(
        print_result=True,
        all_sample=all_sample_flag,
        out_sample=out_sample_flag
    )



def run_backtest_from_optimized_params():
    result_path = os.path.join(os.getcwd(), "optimization_results.json")
    
    if not os.path.exists(result_path):
        print("No optimization_results.json found. Please run optimization first.")
        return

    with open(result_path, "r") as f:
        result = json.load(f)

    params = result["best_params"]

    # Ask user which dataset to use
    print("\nWhich dataset do you want to run the backtest on?")
    print("1. In-sample data")
    print("2. Out-of-sample data")
    print("3. All data")
    dataset_choice = input("Enter choice (1/2/3): ").strip()

    in_sample_flag = False
    out_sample_flag = False
    all_sample_flag = False

    if dataset_choice == "1":
        in_sample_flag = True
    elif dataset_choice == "2":
        out_sample_flag = True
    elif dataset_choice == "3":
        all_sample_flag = True
    else:
        print("Invalid choice. Defaulting to in-sample data.")
        in_sample_flag = True

    backtest = Backtesting(
        params["period_rsi"],
        params["period_bb"],
        0.8,  # in_sample_size for splitting
        params["risk_per_trade"],
        params["rsi_oversold"],
        params["rsi_overbought"],
        params["stop_loss"],
        params["take_profit"],
    )

    backtest.initiate_data(True)
    backtest.apply_indicators()
    
    backtest.run_backtest(
        print_result=True,
        all_sample=all_sample_flag,
        out_sample=out_sample_flag
    )




def main_menu():
    while True:
        print("\n=== Trading Strategy Menu ===")
        print("1. Run Backtest")
        print("2. Run Backtest without fee")
        print("3. Optimize Strategy")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            run_backtesting()
        elif choice == "2":
            run_backtesting_no_fee()
        elif choice == "3":            
            try:
                n = int(input("Enter number of optimization trials: "))
                run_optimization(n)
            except ValueError:
                print("Please enter a valid number.")
        elif choice == "4":
            print ("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()
