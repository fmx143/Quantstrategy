import pandas as pd
import numpy as np
import vectorbt as vbt
import optuna
import optuna_dashboard
import warnings
import random



''' ------------------------------
Part 1, Configuration & Constantes
------------------------------ '''
INITIAL_CAPITAL = 10000
COMMISSION_PCT = 0.0  # Example: 0.05% per trade (adjust as needed)
PIP_VALUE_EURUSD = 0.0001
DATA_FREQUENCY = '1min' # Must match CSV timeframe (e.g., '1h', '15min')
MIN_TRADES = 10 # Minimum number of trades required for a valid trial
MC_ITERATIONS = 100

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

''' ------------------------------
Part 1, import data
------------------------------ '''
try:
    # Using the data as specified in the original request
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_4H_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_1H_5y_clean.csv'
    data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_1min_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_15min_5y_clean.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_Daily_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_4H_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_1H_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_1min_5y_clean.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_15min_5y_clean.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_Daily_5y_cleaned.csv'


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
Part 2, Backtest Function
------------------------------ '''
# Add this function to calculate VWAP (Using your manual implementation)
def calculate_vwap(open_p, high_p, low_p, close_p, volume_v):
    """
    Calculate VWAP (Volume Weighted Average Price).
    VWAP = Cumulative (Price * Volume) / Cumulative Volume
    Note: VWAP is session-based. This cumulative calculation is for the entire dataset.
    For true session VWAP, you'd need to reset the cumulative sums at the start of each trading session (e.g., daily).
    Using typical price as (Open+High+Low+Close)/4 as is common.
     """
    typical_price = (open_p + high_p + low_p + close_p) / 4
    cumulative_price_volume = (typical_price * volume_v).cumsum()
    cumulative_volume = volume_v.cumsum()
    # Avoid division by zero where volume might be 0 or cumulative volume is 0
    vwap = cumulative_price_volume.div(cumulative_volume, fill_value=np.nan)
    return vwap

# Add this function to calculate VWMA (Using your manual implementation)
def calculate_vwma(close_p, volume_v, window):
    """
    Calculate VWMA (Volume Weighted Moving Average).
    VWMA = Sum(Price * Volume) / Sum(Volume) over a rolling window
    """
    price_volume = close_p * volume_v
    # Rolling sum, then divide
    vwma = price_volume.rolling(window=window).sum().div(volume_v.rolling(window=window).sum(), fill_value=np.nan)
    return vwma

def run_backtest(price_data, params):
    """
    Runs the VectorBT backtest for a given set of parameters
    using VWAP/VWMA Bands and ATR with specific entry/exit rules.
    """
    # Extract necessary price series from the DataFram
    open_prices = price_data['open']
    high_prices = price_data['high']
    low_prices = price_data['low']
    close_prices = price_data['close']
    volume = price_data['volume'] # Need volume for VWAP and VWMA
    fixed_sl_pips = params['fixed_sl']
    reward_ratio = params['reward_ratio']

    # --- Strategy Parameters ---
    vwma_window = params['vwma_window']
    atr_window = params['atr_window']
    # Sort band multipliers to ensure band ordering
    band_multipliers = sorted([params['band_multiplier_1'], params['band_multiplier_2'], params['band_multiplier_3']])
    band_multiplier_1 = band_multipliers[0]
    band_multiplier_2 = band_multipliers[1]
    band_multiplier_3 = band_multipliers[2]

    # Parameters for fixed SL/TP from Optuna
    fixed_sl_pips = params['fixed_sl'] # Note: Using 'fixed_sl' key from Optuna
    risk_reward_ratio = params['reward_ratio'] # Note: Using 'reward_ratio' key from Optuna

    # Calculate fixed pip values in price units
    fixed_sl_value = fixed_sl_pips * PIP_VALUE_EURUSD # Use the defined PIP_VALUE
    fixed_tp_value = fixed_sl_value * risk_reward_ratio

    # --- Calculate Indicators (using your manual functions and vectorbt for ATR) ---
    vwap = calculate_vwap(open_prices, high_prices, low_prices, close_prices, volume)
    vwma = calculate_vwma(close_prices, volume, vwma_window)
    atr = vbt.ATR.run(high_prices, low_prices, close_prices, window=atr_window).atr

    # Calculate VWAP Bands based on ATR (Keep band calculations as they are used in entry logic)
    lower_band_3 = vwap - atr * band_multiplier_3
    lower_band_2 = vwap - atr * band_multiplier_2
    lower_band_1 = vwap - atr * band_multiplier_1
    upper_band_1 = vwap + atr * band_multiplier_1
    upper_band_2 = vwap + atr * band_multiplier_2
    upper_band_3 = vwap + atr * band_multiplier_3

    # --- Signal Generation (High Win Rate, Small Gains - Mean Reversion) ---
    # Entry Logic: Price crosses below/above the outermost band and then crosses back inside
    # Look for a close back inside the outermost band after penetrating it
    crossed_below_band3 = (close_prices.shift(1) > lower_band_3.shift(1)) & (close_prices <= lower_band_3)
    crossed_above_band3 = (close_prices.shift(1) < upper_band_3.shift(1)) & (close_prices >= upper_band_3)
    crossed_back_above_band3 = (close_prices.shift(1) <= lower_band_3.shift(1)) & (close_prices > lower_band_3)
    crossed_back_below_band3 = (close_prices.shift(1) >= upper_band_3.shift(1)) & (close_prices < upper_band_3)

    # Long Entry: Close crosses back above the lowest band
    entries_long = crossed_below_band3 & crossed_back_above_band3
    # Short Entry: Close crosses back below the highest band
    entries_short = crossed_above_band3 & crossed_back_below_band3

# --- SL/TP Calculation (as percentages for vectorbt) ---
    # Calculate SL amount in price terms
    sl_amount = fixed_sl_pips * PIP_VALUE_EURUSD
    # Ensure no division by zero if close_prices can be zero (unlikely for FX)
    close_prices_safe = close_prices.replace(0, np.nan) # Replace 0 with NaN temporarily
    sl_stop_pct = (sl_amount / close_prices_safe).fillna(method='ffill').fillna(0) # Forward fill to handle potential NaNs at start
    tp_stop_pct = sl_stop_pct * reward_ratio
    # Calculate SL percentage relative to the closing price *at each point*
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
            sl_stop=sl_stop_pct,       # Pass the calculated SL percentage Series
            tp_stop=tp_stop_pct,       # Pass the calculated TP percentage Series
            init_cash=INITIAL_CAPITAL,
            fees=COMMISSION_PCT,       # Commission percentage
            freq=DATA_FREQUENCY
        )
        return portfolio
    except Exception as e:
        print(f"Warning: Error during portfolio simulation for params {params}: {e}")
        return None

''' ------------------------------
Part 3 Monte Carlo Simulation - KEPT AS IS
------------------------------ '''
def monte_carlo_trades(portfolio, n_iter=MC_ITERATIONS):
    pnls = portfolio.trades.pnl.values

    # Handle case with no trades
    if len(pnls) == 0:
        print("Warning: No trades available for Monte Carlo simulation.")
        # Return placeholder values indicating no results
        return {'sharpe': [0.0], 'max_dd': [0.0], 'total_return': [0.0]}


    results = {'sharpe': [], 'max_dd': [], 'total_return': []}

    # Calculate the annualization factor based on DATA_FREQUENCY
    # This logic tries to determine periods per year based on the frequency string
    try:
        annualization_factor = 252 # Default to daily periods if parsing fails or is not supported

        if 'min' in DATA_FREQUENCY.lower():
            freq_minutes = int(DATA_FREQUENCY.lower().replace('min', '').strip())
            periods_per_day = (24 * 60) / freq_minutes
            annualization_factor = 252 * periods_per_day # 252 trading days/year * periods/day
        elif 'h' in DATA_FREQUENCY.lower():
            freq_hours = int(DATA_FREQUENCY.lower().replace('h', '').strip())
            freq_minutes = freq_hours * 60
            periods_per_day = (24 * 60) / freq_minutes
            annualization_factor = 252 * periods_per_day
        elif 'd' in DATA_FREQUENCY.lower(): # Handle 'd' or 'D' for daily
             periods_per_day = 1 # 1 period per day
             annualization_factor = 252 * periods_per_day
        # Add other frequencies like 'w' (weekly), 'm' (monthly) if needed
        # e.g., elif 'w' in DATA_FREQUENCY.lower(): annualization_factor = 52
        # e.g., elif 'm' in DATA_FREQUENCY.lower(): annualization_factor = 12
        else:
             print(f"Warning: Unknown DATA_FREQUENCY format '{DATA_FREQUENCY}'. Cannot annualize Sharpe ratio correctly.")
             # annualization_factor remains default 252

        # Ensure positive annualization factor
        annualization_factor = max(annualization_factor, 1)

    except (ValueError, ZeroDivisionError) as e:
         print(f"Error parsing DATA_FREQUENCY '{DATA_FREQUENCY}' for annualization: {e}. Using daily annualization (252).")
         annualization_factor = 252 # Fallback to daily if parsing fails

    sqrt_annualization_factor = np.sqrt(annualization_factor)

    for _ in range(n_iter):
        # --- Monte Carlo Resampling with Replacement ---
        # Simulate potential trade sequences by sampling trades with replacement
        sample = np.random.choice(pnls, size=len(pnls), replace=True)
        # If you intended just shuffling (sequence risk), use:
        # sample = np.random.permutation(pnls) # or random.sample(list(pnls), len(pnls))

        # Calculate cumulative sum of PnL
        cum = np.cumsum(sample) + INITIAL_CAPITAL

        # Calculate Metrics from the simulated equity curve
        # Handle edge case where cum might have only one element (unlikely with replace=True and >0 trades)
        if len(cum) <= 1:
             total_ret = (cum[-1] / INITIAL_CAPITAL - 1) * 100 if len(cum) > 0 else 0.0
             dd = (np.maximum.accumulate(cum) - cum).max() / INITIAL_CAPITAL * 100 if len(cum) > 0 else 0.0
             sharpe = 0.0 # Cannot calculate meaningful Sharpe with 0 or 1 point
        else:
            # Total Return
            total_ret = (cum[-1] / INITIAL_CAPITAL - 1) * 100

            # Max Drawdown (as percentage of previous peak equity)
            peak = np.maximum.accumulate(cum)
            # Avoid division by zero if peak is 0 (e.g., if initial capital was 0, though not in this case)
            dd_values = (peak - cum) / np.where(peak == 0, 1, peak) # Calculate relative drawdown
            dd = dd_values.max() * 100 # Max drawdown percentage

            # Returns for Sharpe ratio (percentage change between periods)
            # Avoid division by zero if cum[:-1] contains zeros
            rets = np.diff(cum) / np.where(cum[:-1] == 0, 1, cum[:-1])

            # Calculate Sharpe Ratio
            # Handle case where returns have no variance
            std_dev_rets = np.std(rets, ddof=1)
            if std_dev_rets == 0:
                sharpe = np.mean(rets) * sqrt_annualization_factor if np.mean(rets) > 0 else 0.0 # Avoid division by zero
            else:
                sharpe = np.mean(rets) / std_dev_rets * sqrt_annualization_factor

        results['sharpe'].append(sharpe)
        results['max_dd'].append(dd)
        results['total_return'].append(total_ret)

    return results

''' ------------------------------
Part 4 Optuna Objective Function - MODIFIED for new parameters
------------------------------ '''
def objective(trial):
    """
    Objective function for Optuna MULTI-OBJECTIVE optimization.
    Suggests parameters, runs backtest, and returns multiple performance metrics.
    """
    vwma_window = trial.suggest_int('vwma_window', 10, 100)
    atr_window = trial.suggest_int('atr_window', 10, 100)
    band_multiplier_1 = trial.suggest_float('band_multiplier_1', 0.5, 2.0, step=0.1)
    band_multiplier_2 = trial.suggest_float('band_multiplier_2', 1.5, 3.0, step=0.1)
    band_multiplier_3 = trial.suggest_float('band_multiplier_3', 2.5, 5.0, step=0.1)
    atr_sl_multiplier = trial.suggest_float('atr_sl_multiplier', 1.0, 5.0, step=0.1)
    fixed_sl = trial.suggest_int('fixed_sl', 5, 50)           # SL in pips
    reward_ratio = trial.suggest_float('reward_ratio', 0.5, 5.0, step=0.1)

    params = {
        'vwma_window': vwma_window,
        'atr_window': atr_window,
        'band_multiplier_1': band_multiplier_1,
        'band_multiplier_2': band_multiplier_2,
        'band_multiplier_3': band_multiplier_3,
        'atr_sl_multiplier': atr_sl_multiplier,
        'fixed_sl': fixed_sl,
        'reward_ratio': reward_ratio,
    }

    # 2) Run your backtest
    # Pass the entire price_df DataFrame
    portfolio = run_backtest(price_df, params)

    # 3) Prune if too few trades or outright errors
    if portfolio is None or portfolio.trades.count() < MIN_TRADES:
        # Penalize invalid trials - Return negative infinity for maximization objectives
        # and positive infinity for minimization objectives (like drawdown magnitude if it were minimized)
        # Since max_drawdown is returned as a negative percentage, maximizing it means minimizing its magnitude towards 0.
        return -np.inf, 0.0, -np.inf, 0.0 # Sharpe, WinRate, MaxDD, FinalEquity

    # 4) Calculate and return your four metrics
    # Ensure calculated metrics are floats and handle potential None results
    sharpe       = float(portfolio.sharpe_ratio()) if portfolio.sharpe_ratio() is not None else -1.0
    win_rate     = float(portfolio.trades.win_rate()) if portfolio.trades.win_rate() is not None else 0.0
    max_drawdown = float(portfolio.max_drawdown()) * 100 if portfolio.max_drawdown() is not None else -100.0  # Convert to percentage, default to -100% if None
    final_equity = float(portfolio.final_value()) if portfolio.final_value() is not None else INITIAL_CAPITAL # Default to initial capital if None


    return sharpe, win_rate, max_drawdown, final_equity


''' ------------------------------
Part 5 Main Execution - MODIFIED to pass price_df
------------------------------ '''
if __name__ == '__main__':
    # Check if price_df exists and is valid DataFrame before proceeding
    if 'price_df' in globals() and isinstance(price_df, pd.DataFrame) and not price_df.empty:
        # Ensure required columns are present, including volume
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in price_df.columns for col in required_cols):
            print(f"Error: Dataframe does not contain all required columns for VWAP/VWMA: {required_cols}")
            exit()


        print("\n2) Setting up Optuna study...")

        # --- Setup Study ---
        # Use SQLite for persistence, enabling the dashboard
        # Updated study name for the new strategy
        study_name = 'Opt_Strategy_VWAP_VWMA_ATR_vbt_gbpusd_15min' # Unique name for the study
        storage_name = f"sqlite:///{study_name}.db"

        # Define the directions for each objective (must match the order returned by the objective function)
        # Maximize Sharpe, Maximize Win Rate, Maximize Max Drawdown (closer to 0), Maximize Final Equity
        direction = ['maximize', 'maximize', 'maximize', 'maximize']
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,  # Load previous results if study exists
                directions=direction # Specify multi-objective directions
            )
            print(f"Study '{study_name}' loaded or created.")

        except Exception as e:
            print(f"Error creating or loading Optuna study: {e}")
            # Fallback or exit if study cannot be created/loaded
            try: # Attempt to create in-memory study as fallback
                print("Attempting to create in-memory study as fallback...")
                study = optuna.create_study(study_name=study_name + "_in_memory", directions=direction)
                storage_name = "in-memory"
                print("In-memory study created. Results will not be persistent.")
            except Exception as inner_e:
                print(f"Failed to create in-memory study: {inner_e}")
                print("Exiting due to study creation failure.")
                exit()


        # --- Run Optimization ---
        n_trials = MC_ITERATIONS # Adjust number of trials as needed
        print(f"🚀 Starting Multi-Objective Optimization for {n_trials} trials...")
        print(f"   Objectives: Sharpe Ratio (max), Win Rate (max), Max Drawdown (max -> min magnitude), Final Equity (max)")
        print(f"   Study Name: {study_name}")
        print(f"   Storage: {storage_name}")
        if storage_name != "in-memory":
            print(f"   Run 'optuna-dashboard {storage_name}' in your terminal to view progress.")
        else:
            print("   Optuna Dashboard not available for in-memory study.")


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
            # Adjusted header to match the 4 metrics returned
            print(f"{'Trial':>5} | {'Sharpe':>10} | {'Win Rate':>10} | {'Max DD':>10} | {'Final Equity':>15} | {'Params':<40}")
            print("-" * 100)
            successful_pareto_count = 0
            for i, trial in enumerate(pareto_trials):
                # Check if the trial has the expected number of values before accessing them
                # Now expecting 4 values
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None and len(trial.values) == 4:
                    params_str = ', '.join(f"{k}={v}" for k, v in trial.params.items())
                    # Access values using indices 0, 1, 2, 3
                    print(f"{trial.number:>5} | {trial.values[0]:>10.4f} | {trial.values[1]:>10.2f} | {trial.values[2]:>10.4f} | {trial.values[3]:>15.2f} | {params_str}")
                    successful_pareto_count += 1
                else:
                    # Optionally print a warning for skipped trials
                    print(f"Skipping Trial {trial.number}: State={trial.state}, Values={trial.values} (Incomplete/Invalid)")

            print("-" * 80)
            if successful_pareto_count == 0:
                print("Warning: Although Pareto trials were identified, none had complete/valid objective values.")

        # --- Optional: Find and display the trial that was best for EACH objective individually ---
        print("\n--- Trials with highest individual metrics (among all completed trials) ---")

        try: # Add error handling around DataFrame creation/manipulation
            completed_trials_df = study.trials_dataframe(multi_index=False) # Get results as DataFrame
            # Filter ONLY for completed trials AFTER creating the dataframe
            completed_trials_df = completed_trials_df[completed_trials_df['state'] == 'COMPLETE'].copy() # Use .copy() to avoid SettingWithCopyWarning

            # Check if 'values_0', 'values_1', 'values_2', 'values_3' exist, otherwise skip this section
            required_value_cols = ['values_0', 'values_1', 'values_2', 'values_3']
            if not all(col in completed_trials_df.columns for col in required_value_cols):
                print("Could not find required 'values_x' columns in completed trials DataFrame. Skipping individual bests.")
            elif not completed_trials_df.empty:
                 # Rename columns (make sure these names exist first)
                 completed_trials_df.rename(columns={
                    'values_0': 'sharpe',
                    'values_1': 'win_rate',
                    'values_2': 'max_drawdown', # Note: Max DD is negative, so 'best' means largest (closest to 0)
                    'values_3': 'final_equity'
                 }, inplace=True)
                 # Check if columns were successfully renamed and drop rows with NaN in metric columns before finding idxmax
                 metric_cols = ['sharpe', 'win_rate', 'max_drawdown', 'final_equity']
                 if not all(col in completed_trials_df.columns for col in metric_cols):
                    print("Failed to rename metric columns. Skipping individual bests.")
                 else:
                    completed_trials_df.dropna(subset=metric_cols, inplace=True) # Drop rows where metrics couldn't be calculated

                    if not completed_trials_df.empty:
                        # Find best Sharpe
                        best_sharpe_idx = completed_trials_df['sharpe'].idxmax()
                        best_sharpe_trial_num = int(completed_trials_df.loc[best_sharpe_idx]['number']) # Ensure it's int
                        best_sharpe_trial = study.trials[best_sharpe_trial_num]
                        print(f"\nBest Sharpe Ratio Trial (#{best_sharpe_trial.number}):")
                        # Check values before printing (now expecting 4 values)
                        if best_sharpe_trial.values and len(best_sharpe_trial.values) == 4:
                            print(f"  Metrics: Sharpe={best_sharpe_trial.values[0]:.4f}, WinRate={best_sharpe_trial.values[1]:.2f}, MaxDD={best_sharpe_trial.values[2]:.4f}%, FinalEquity={best_sharpe_trial.values[3]:.2f}")
                            print(f"  Params: {best_sharpe_trial.params}")
                        else:
                            print(f"  Metrics: Incomplete data - {best_sharpe_trial.values}")
                            print(f"  Params: {best_sharpe_trial.params}")

                        # Find best Win Rate
                        best_winrate_idx = completed_trials_df['win_rate'].idxmax()
                        best_winrate_trial_num = int(completed_trials_df.loc[best_winrate_idx]['number'])
                        best_winrate_trial = study.trials[best_winrate_trial_num]
                        print(f"\nBest Win Rate Trial (#{best_winrate_trial.number}):")
                        if best_winrate_trial.values and len(best_winrate_trial.values) == 4:
                            print(f"  Metrics: Sharpe={best_winrate_trial.values[0]:.4f}, WinRate={best_winrate_trial.values[1]:.2f}, MaxDD={best_winrate_trial.values[2]:.4f}%, FinalEquity={best_winrate_trial.values[3]:.2f}")
                            print(f"  Params: {best_winrate_trial.params}")
                        else:
                            print(f"  Metrics: Incomplete data - {best_winrate_trial.values}")
                            print(f"  Params: {best_winrate_trial.params}")

                        # Find best Max Drawdown (closest to 0)
                        best_drawdown_idx = completed_trials_df['max_drawdown'].idxmax() # Since it's negative, max finds closest to 0
                        best_drawdown_trial_num = int(completed_trials_df.loc[best_drawdown_idx]['number'])
                        best_drawdown_trial = study.trials[best_drawdown_trial_num]
                        print(f"\nBest Max Drawdown Trial (#{best_drawdown_trial.number}):")
                        if best_drawdown_trial.values and len(best_drawdown_trial.values) == 4:
                            print(f"  Metrics: Sharpe={best_drawdown_trial.values[0]:.4f}, WinRate={best_drawdown_trial.values[1]:.2f}, MaxDD={best_drawdown_trial.values[2]:.4f}%, FinalEquity={best_drawdown_trial.values[3]:.2f}")
                            print(f"  Params: {best_drawdown_trial.params}")
                        else:
                            print(f"  Metrics: Incomplete data - {best_drawdown_trial.values}")
                            print(f"  Params: {best_drawdown_trial.params}")

                        # Find best Final Equity
                        best_equity_idx = completed_trials_df['final_equity'].idxmax()
                        best_equity_trial_num = int(completed_trials_df.loc[best_equity_idx]['number'])
                        best_equity_trial = study.trials[best_equity_trial_num]
                        print(f"\nBest Final Equity Trial (#{best_equity_trial.number}):")
                        if best_equity_trial.values and len(best_equity_trial.values) == 4:
                            print(f"  Metrics: Sharpe={best_equity_trial.values[0]:.4f}, WinRate={best_equity_trial.values[1]:.2f}, MaxDD={best_equity_trial.values[2]:.4f}%, FinalEquity={best_equity_trial.values[3]:.2f}")
                            print(f"  Params: {best_equity_trial.params}")
                        else:
                            print(f"  Metrics: Incomplete data - {best_equity_trial.values}")
                            print(f"  Params: {best_equity_trial.params}")


                        # --- Run backtest with best Sharpe parameters for final stats and MC ---
                        print("\n--- Running backtest with the 'Best Sharpe Ratio' parameters ---")
                        # Ensure the best_sharpe_trial has valid parameters before running
                        if best_sharpe_trial.params:
                            # Pass price_df to the final run_backtest call
                            final_portfolio = run_backtest(price_df, best_sharpe_trial.params)
                            if final_portfolio is not None:
                                print("\n--- Performance Metrics (Best Sharpe Params) ---")
                                print(final_portfolio.stats())

                                # <<< START: Add Monte Carlo Simulation here >>> - KEPT AS IS
                                if final_portfolio.trades.count() >= MIN_TRADES:
                                    print(f"\n--- Running Monte Carlo Simulation ({MC_ITERATIONS} iterations) ---")
                                    mc_results = monte_carlo_trades(final_portfolio, n_iter=MC_ITERATIONS)

                                    # Analyze MC results (e.g., median and percentiles)
                                    mc_sharpe_median = np.median(mc_results['sharpe'])
                                    mc_sharpe_10th = np.percentile(mc_results['sharpe'], 10)
                                    mc_sharpe_90th = np.percentile(mc_results['sharpe'], 90)

                                    # Max Drawdown is negative, so 90th percentile is a 'worse' case (more negative)
                                    mc_max_dd_median = np.median(mc_results['max_dd'])
                                    mc_max_dd_10th = np.percentile(mc_results['max_dd'], 10) # Less negative/better
                                    mc_max_dd_90th = np.percentile(mc_results['max_dd'], 90) # More negative/worse

                                    mc_total_return_median = np.median(mc_results['total_return'])
                                    mc_total_return_10th = np.percentile(mc_results['total_return'], 10) # Lower bound
                                    mc_total_return_90th = np.percentile(mc_results['total_return'], 90) # Upper bound


                                    print(f"MC Sharpe Ratio (Median): {mc_sharpe_median:.4f}")
                                    print(f"MC Sharpe Ratio (10th-90th Pctl Range): ({mc_sharpe_10th:.4f}, {mc_sharpe_90th:.4f})")
                                    print(f"MC Max Drawdown % (Median): {mc_max_dd_median:.4f}")
                                    print(f"MC Max Drawdown % (10th-90th Pctl Range): ({mc_max_dd_10th:.4f}, {mc_max_dd_90th:.4f})")
                                    print(f"MC Total Return % (Median): {mc_total_return_median:.2f}")
                                    print(f"MC Total Return % (10th-90th Pctl Range): ({mc_total_return_10th:.2f}, {mc_total_return_90th:.2f})")

                                    # Plotting Optional
                                    
                                else:
                                    print("Not enough trades for Monte Carlo simulation with best Sharpe parameters.")
                                # <<< END: Add Monte Carlo Simulation here >>>

                            else:
                                print("Could not run final backtest with best Sharpe parameters.")
                        else:
                            print("Best Sharpe trial had no parameters to run backtest.")
 
                    else:
                        print("No completed trials with valid metrics found after filtering NaNs.")
            else:
                print("No completed trials found to determine individual bests.")

        except Exception as e:
            print(f"\nError processing trials dataframe or finding individual bests: {e}")
            import traceback
            traceback.print_exc()

# --- [Optuna Dashboard Reminder and else block for data loading remain the same] ---
# This message should ideally be printed after successful study creation, not always.
if 'study' in globals() and storage_name != "in-memory":
    print(f"\nTo view the Optuna Dashboard, run the following command in your terminal:")
    print(f"optuna-dashboard {storage_name}")

elif 'price_df' not in globals() or price_df.empty:
    print("\nScript finished without running optimization due to data loading issues.")