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
    sl_stop_pct = (sl_amount / close_prices_safe).fillna(0) # Calculate pct, fill NaN results with 0
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
    Objective function for Optuna to optimize.
    Takes an Optuna trial object, suggests parameters, runs the backtest,
    and returns a performance metric.
    """
    # --- Suggest Parameters ---
    # Use integers for periods
    ema_fast = trial.suggest_int('ema_fast', 1, 15) # Example range
    ema_mid = trial.suggest_int('ema_mid', 40, 60) # Example range
    ema_slow = trial.suggest_int('ema_slow', 150, 200) # Example range

    # Ensure fast < mid < slow logic
    if not (ema_fast < ema_mid < ema_slow):
         # Prune this trial if condition not met
         raise optuna.exceptions.TrialPruned()

    # Use integers for pips
    fixed_sl = trial.suggest_int('fixed_sl', 5, 25) # Example range

    # Use float for ratio
    reward_ratio = trial.suggest_float('reward_ratio', 1.0, 4.0, step=0.1) # Example range

    params = {
        'ema_fast': ema_fast,
        'ema_mid': ema_mid,
        'ema_slow': ema_slow,
        'fixed_sl': fixed_sl,
        'reward_ratio': reward_ratio,
    }

    # --- Run Backtest ---
    portfolio = run_backtest(close_prices, params)

    # --- Return Metric ---
    if portfolio is None or portfolio.trades.count() == 0:
        # Penalize trials with errors or no trades (return a very low value)
        raise optuna.exceptions.TrialPruned("No trades executed or error in backtest.")
        return -1.0 # Return low value for maximization

    # --- Choose Metric to Optimize ---
    # Example: Total Return (can be negative)
    metric = portfolio.total_return()

    # Alternative: Sharpe Ratio (handle potential NaN/inf)
    # sharpe = portfolio.sharpe_ratio()
    # metric = sharpe if np.isfinite(sharpe) else -1.0

    # Alternative: Sortino Ratio
    # sortino = portfolio.sortino_ratio()
    # metric = sortino if np.isfinite(sortino) else -1.0

    # Alternative: SQN (System Quality Number) - May require portfolio.trades access
    # try:
    #     sqn = portfolio.sqn() # Check if this method exists or calculate manually
    #     metric = sqn if np.isfinite(sqn) else -1.0
    # except AttributeError:
    #     metric = portfolio.total_return() # Fallback

    # Ensure metric is float
    return float(metric)


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

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,  # Load previous results if study exists
            direction='maximize'  # Maximize the metric returned by objective
        )

        # --- Run Optimization ---
        n_trials = 100 # Number of optimization trials to run
        print(f"🚀 Starting Optimization for {n_trials} trials...")
        print(f"   Study Name: {study_name}")
        print(f"   Storage: {storage_name}")
        print("   Run 'optuna-dashboard sqlite:///Opt_Strategy_EMA_vbt_eurusd_4h' in your terminal to view progress.")

        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                # n_jobs=-1 # Use all available CPU cores for parallel execution
                # Note: Parallelization with vectorbt might not always yield linear speedups
                # depending on the strategy complexity and data size. Test with n_jobs=1 first.
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

        best_trial = study.best_trial
        print("\n--- Best Trial ---")
        print(f"  Value (Metric): {best_trial.value:.6f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # --- Run Backtest with Best Parameters ---
        print("\n--- Running backtest with best parameters ---")
        best_params = best_trial.params
        # Need to handle the pruning constraint again if loading best params
        if not (best_params['ema_fast'] < best_params['ema_mid'] < best_params['ema_slow']):
             print("Warning: Best parameters found violate fast < mid < slow constraint. This shouldn't happen if pruning worked.")
        else:
            final_portfolio = run_backtest(close_prices, best_params)

            if final_portfolio is not None:
                print("\n--- Performance Metrics (Best Parameters) ---")
                print(final_portfolio.stats())

                # Plotting (optional, requires plotly installed)
                try:
                    fig = final_portfolio.plot()
                    fig.show()
                except Exception as plot_err:
                    print(f"Plotting failed: {plot_err}")
            else:
                print("Could not run final backtest with best parameters.")

        # --- Optuna Dashboard Reminder ---
        print("\n--- Optuna Dashboard ---")
        print("To visualize the optimization study, run the following command in your terminal:")
        print(f"optuna-dashboard {storage_name}")

    else:
        print("Data was not loaded properly. Exiting.")