import pandas as pd
import numpy as np
import vectorbt as vbt
import optuna
import optuna_dashboard
import warnings

# Suppress specific warnings if needed (e.g., from TA-Lib if used indirectly)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

''' ------------------------------
Part 1, import data
------------------------------ '''
try:
    # Using the 4H data as specified in the original request
    data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_4H_5y_cleaned.csv'
    price_df = pd.read_csv(data_path)
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'])
    price_df.set_index('Datetime', inplace=True)

    # Ensure standard column names (vectorbt prefers lowercase)
    price_df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume' # Keep volume if available/needed
    }, inplace=True)

    # Ensure data is sorted by Datetime index
    price_df.sort_index(inplace=True)

    # Select the 'close' price series for calculations
    # Keep the DataFrame for Portfolio simulation if using OHLC
    close_prices = price_df['close']
    print(f"1) Data loaded successfully. Shape: {price_df.shape}")

except FileNotFoundError:
    print(f"1.1) Error: Data file not found at {data_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

''' ------------------------------
Part 2, Define the Objective Function for Optuna
------------------------------ '''

# --- Constants ---
INITIAL_CAPITAL = 10000
COMMISSION_PCT = 0.0  # Example: 0.05% per trade (adjust as needed)
PIP_VALUE_EURUSD = 0.0001
DATA_FREQUENCY = '4H' # Important for performance calculations
MIN_TRADES = 10 # Minimum number of trades required for a valid trial

def run_backtest(close_prices, params):
    """
    Runs the VectorBT backtest for a given set of parameters.
    """
    ema_fast_period = params['ema_fast']
    ema_mid_period = params['ema_mid']
    ema_slow_period = params['ema_slow']
    fixed_sl_pips = params['fixed_sl']
    reward_ratio = params['reward_ratio']

    # --- Calculate Indicators ---
    ema_fast = vbt.MA.run(close_prices, ema_fast_period, short_name='fast', ewm=True).ma
    ema_mid = vbt.MA.run(close_prices, ema_mid_period, short_name='mid', ewm=True).ma
    ema_slow = vbt.MA.run(close_prices, ema_slow_period, short_name='slow', ewm=True).ma

    # --- Signal Generation ---
    # Market Conditions
    market_cond_buy = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    market_cond_sell = (ema_fast < ema_mid) & (ema_mid < ema_slow)

    # Crossover Signals
    fast_cross_above_mid = ema_fast.vbt.crossed_above(ema_mid)
    fast_cross_below_mid = ema_fast.vbt.crossed_below(ema_mid)

    # Entry Signals
    entries_long = fast_cross_above_mid & market_cond_buy
    entries_short = fast_cross_below_mid & market_cond_sell

    # --- SL/TP Calculation (as percentages for vectorbt) ---
    # Calculate SL amount in price terms
    sl_amount = fixed_sl_pips * PIP_VALUE_EURUSD
    # Ensure no division by zero if close_prices can be zero (unlikely for FX)
    close_prices_safe = close_prices.replace(0, np.nan) # Replace 0 with NaN temporarily
    sl_stop_pct = (sl_amount / close_prices_safe).fillna(method='ffill').fillna(0) # Forward fill to handle potential NaNs at start
    tp_stop_pct = sl_stop_pct * reward_ratio

    # Calculate SL percentage relative to the closing price *at each point*
    # Note: More accurately, SL/TP should be based on entry price, but that's
    # harder in pure vectorization. Using close price is a common approximation.
    sl_stop_pct = sl_amount / close_prices
    tp_stop_pct = sl_stop_pct * reward_ratio

    # --- Portfolio Simulation ---
    # Use entries_long and entries_short. Exits handled by SL/TP.
    # Set exits=None as we rely on sl_stop and tp_stop
    try:
        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=entries_long,
            exits=None,                # Use SL/TP for exits
            short_entries=entries_short,
            short_exits=None,          # Use SL/TP for exits
            sl_stop=sl_stop_pct,       # Pass the SL percentage Series
            tp_stop=tp_stop_pct,       # Pass the TP percentage Series
            init_cash=INITIAL_CAPITAL,
            fees=COMMISSION_PCT,       # Commission percentage
            freq=DATA_FREQUENCY        # Set data frequency
        )
        return portfolio
    except Exception as e:
        print(f"Warning: Error during portfolio simulation for params {params}: {e}")
        return None


def objective(trial):
    """
    Objective function for Optuna MULTI-OBJECTIVE optimization.
    Suggests parameters, runs backtest, and returns multiple performance metrics.
    """
    # --- Suggest Parameters ---
    ema_fast = trial.suggest_int('ema_fast', 5, 50)    # Wider range start
    ema_mid = trial.suggest_int('ema_mid', 20, 100)   # Wider range start
    ema_slow = trial.suggest_int('ema_slow', 100, 250) # Wider range start

    # Ensure fast < mid < slow logic more efficiently
    # Adjust mid based on fast, and slow based on mid
    ema_mid = trial.suggest_int('ema_mid', ema_fast + 10, 150) # Ensure mid > fast + reasonable gap
    ema_slow = trial.suggest_int('ema_slow', ema_mid + 50, 300) # Ensure slow > mid + reasonable gap

    # Prune explicitly if constraint is somehow violated (shouldn't be with new suggestions)
    if not (ema_fast < ema_mid < ema_slow):
         raise optuna.exceptions.TrialPruned("EMA order constraint violated.")

    fixed_sl = trial.suggest_int('fixed_sl', 5, 50) # Pips
    reward_ratio = trial.suggest_float('reward_ratio', 0.5, 5.0, step=0.1)

    params = {
        'ema_fast': ema_fast,
        'ema_mid': ema_mid,
        'ema_slow': ema_slow,
        'fixed_sl': fixed_sl,
        'reward_ratio': reward_ratio,
    }

    # --- Run Backtest ---
    portfolio = run_backtest(close_prices, params)

    # --- Calculate Metrics & Handle Invalid Trials ---
    if portfolio is None or portfolio.trades.count() < MIN_TRADES:
        # Prune trials with errors, or too few trades for metrics to be meaningful
        raise optuna.exceptions.TrialPruned(f"Error, or fewer than {MIN_TRADES} trades executed.")

    # Calculate Sharpe Ratio (maximize)
    sharpe = portfolio.sharpe_ratio()
    sharpe = sharpe if np.isfinite(sharpe) else -1.0 # Penalize non-finite Sharpe

    # Calculate Win Rate (maximize)
    win_rate = portfolio.trades.win_rate() # win_rate is typically between 0 and 100
    win_rate = win_rate if np.isfinite(win_rate) else 0.0 # Penalize non-finite Win Rate

    # Calculate Max Drawdown (maximize - since it's negative, maximizing brings it closer to 0)
    max_drawdown = portfolio.max_drawdown() # max_drawdown is typically negative (e.g., -0.1 for -10%)
    max_drawdown = max_drawdown if np.isfinite(max_drawdown) else -1.0 # Penalize non-finite Drawdown

    # --- Return Multiple Metrics ---
    # Ensure all returned values are floats
    return float(sharpe), float(win_rate), float(max_drawdown)


''' ------------------------------
Part 3, Run Optuna Optimization
------------------------------ '''
if __name__ == '__main__':
    if 'close_prices' in globals() and isinstance(close_prices, pd.Series) and not close_prices.empty:
        print("\n2) Setting up Optuna study...")

        # --- Setup Study ---
        # Use SQLite for persistence, enabling the dashboard
        study_name = 'Opt_Strategy_EMA_vbt_eurusd_4h' # Unique name for the study
        storage_name = f"sqlite:///{study_name}.db"

        # Define the directions for each objective (must match the order returned by the objective function)
        # Maximize Sharpe, Maximize Win Rate, Maximize Max Drawdown (closer to 0)
        direction = ['maximize', 'maximize', 'maximize']

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,  # Load previous results if study exists
            directions=direction # Specify multi-objective directions
        )

        # --- Run Optimization ---
        n_trials = 1000 # Adjust number of trials as needed
        print(f"🚀 Starting Multi-Objective Optimization for {n_trials} trials...")
        print(f"   Objectives: Sharpe Ratio (max), Win Rate (max), Max Drawdown (max -> min magnitude)")
        print(f"   Study Name: {study_name}")
        print(f"   Storage: {storage_name}")
        print(f"   Run 'optuna-dashboard {storage_name}' in your terminal to view progress.")

        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                # n_jobs=-1 # Consider uncommenting for parallel execution, test performance
                gc_after_trial=True # Helps manage memory with vectorbt/pandas
            )
        except KeyboardInterrupt:
            print("\n🛑 Optimization interrupted by user.")
        except Exception as e:
            print(f"\n❌ An error occurred during optimization: {e}")
            import traceback
            traceback.print_exc()


        # --- Analyze and Display Results ---
        print('\n🏁 Optimization Finished.')
        print(f"Number of finished trials: {len(study.trials)}")

        '''
        In multi-objective optimization, there isn't a single "best" trial,
        but rather a set of "Pareto optimal" solutions.
        A solution is Pareto optimal if no objective can be improved
        without worsening at least one other objective.
        '''
        pareto_trials = study.best_trials

        print(f"\n--- Pareto Optimal Trials ({len(pareto_trials)}) ---")
        if not pareto_trials:
             print("No optimal trials found (perhaps all were pruned or encountered errors).")
        else:
            print("Showing results for Pareto optimal trials (trade-offs between objectives):")
            print("-" * 80)
            print(f"{'Trial':>5} | {'Sharpe':>10} | {'Win Rate':>10} | {'Max DD':>10} | {'Params':<40}")
            print("-" * 80)
            for i, trial in enumerate(pareto_trials):
                params_str = ', '.join(f"{k}={v}" for k, v in trial.params.items())
                print(f"{trial.number:>5} | {trial.values[0]:>10.4f} | {trial.values[1]:>10.2f} | {trial.values[2]:>10.4f} | {params_str}")
            print("-" * 80)

            # --- Optional: Find and display the trial that was best for EACH objective individually ---
            print("\n--- Trials with highest individual metrics (among all completed trials) ---")

            completed_trials_df = study.trials_dataframe(multi_index=False) # Get results as DataFrame
            completed_trials_df = completed_trials_df[completed_trials_df['state'] == 'COMPLETE'] # Filter for completed trials

            if not completed_trials_df.empty:
                 # Need to rename columns as Optuna >= 3.0 names them values_0, values_1, etc.
                 completed_trials_df.rename(columns={'values_0': 'sharpe', 'values_1': 'win_rate', 'values_2': 'max_drawdown'}, inplace=True)

                 # Find best Sharpe
                 best_sharpe_trial_num = completed_trials_df.loc[completed_trials_df['sharpe'].idxmax()]['number']
                 best_sharpe_trial = study.trials[best_sharpe_trial_num]
                 print(f"\nBest Sharpe Ratio Trial (#{best_sharpe_trial.number}):")
                 print(f"  Metrics: Sharpe={best_sharpe_trial.values[0]:.4f}, WinRate={best_sharpe_trial.values[1]:.2f}, MaxDD={best_sharpe_trial.values[2]:.4f}")
                 print(f"  Params: {best_sharpe_trial.params}")

                 # Find best Win Rate
                 best_winrate_trial_num = completed_trials_df.loc[completed_trials_df['win_rate'].idxmax()]['number']
                 best_winrate_trial = study.trials[best_winrate_trial_num]
                 print(f"\nBest Win Rate Trial (#{best_winrate_trial.number}):")
                 print(f"  Metrics: Sharpe={best_winrate_trial.values[0]:.4f}, WinRate={best_winrate_trial.values[1]:.2f}, MaxDD={best_winrate_trial.values[2]:.4f}")
                 print(f"  Params: {best_winrate_trial.params}")

                 # Find best Max Drawdown (highest value, i.e., closest to zero)
                 best_drawdown_trial_num = completed_trials_df.loc[completed_trials_df['max_drawdown'].idxmax()]['number']
                 best_drawdown_trial = study.trials[best_drawdown_trial_num]
                 print(f"\nBest Max Drawdown Trial (#{best_drawdown_trial.number}):")
                 print(f"  Metrics: Sharpe={best_drawdown_trial.values[0]:.4f}, WinRate={best_drawdown_trial.values[1]:.2f}, MaxDD={best_drawdown_trial.values[2]:.4f}")
                 print(f"  Params: {best_drawdown_trial.params}")

                 # --- Optional: Run backtest with one of the best trials (e.g., best Sharpe) ---
                 print("\n--- Running backtest with the 'Best Sharpe Ratio' parameters ---")
                 final_portfolio = run_backtest(close_prices, best_sharpe_trial.params)
                 if final_portfolio is not None:
                    print("\n--- Performance Metrics (Best Sharpe Params) ---")
                    print(final_portfolio.stats())
                    # Plotting (optional)
                    try:
                        fig = final_portfolio.plot()
                        fig.show()
                    except Exception as plot_err:
                        print(f"Plotting failed: {plot_err}")
                 else:
                    print("Could not run final backtest with best Sharpe parameters.")

            else:
                 print("No completed trials found to determine individual bests.")


        # --- Optuna Dashboard Reminder ---
        print("\n--- Optuna Dashboard ---")
        print("To visualize the multi-objective optimization study (Pareto front), run:")
        print(f"optuna-dashboard {storage_name}")

    else:
        print("Data was not loaded properly. Exiting.")