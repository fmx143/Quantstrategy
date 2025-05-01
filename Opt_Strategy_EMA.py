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
DATA_FREQUENCY = '15min' # Must match CSV timeframe (e.g., '1h', '15min')
MIN_TRADES = 10 # Minimum number of trades required for a valid trial
MC_ITERATIONS = 1000

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
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_1min_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_15min_5y_clean.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_Daily_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_4H_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_1H_5y_cleaned.csv'
    #data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_1min_5y_clean.csv'
    data_path = r'C:\Users\loick\VS Code\Forex Historical Data\GBPUSD_Tickstory_15min_5y_clean.csv'
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

''' ------------------------------
Part 3 Monte Carlo Simulation
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
Part 4 Optuna Objective Function
------------------------------ '''
def objective(trial):
    """
    Objective function for Optuna MULTI-OBJECTIVE optimization.
    Suggests parameters, runs backtest, and returns multiple performance metrics.
    """
    ema_fast = trial.suggest_int('ema_fast', 5, 50)
    ema_mid = trial.suggest_int('ema_mid', ema_fast + 10, 150)
    ema_slow = trial.suggest_int('ema_slow', ema_mid + 50, 300)
    fixed_sl = trial.suggest_int('fixed_sl', 5, 50)           # SL in pips
    reward_ratio = trial.suggest_float('reward_ratio', 0.5, 5.0, step=0.1)

    params = {
        'ema_fast': ema_fast,
        'ema_mid': ema_mid,
        'ema_slow': ema_slow,
        'fixed_sl': fixed_sl,
        'reward_ratio': reward_ratio,
    }

    # 2) Run your backtest
    portfolio = run_backtest(close_prices, params)

    # 3) Prune if too few trades or outright errors
    if portfolio is None or portfolio.trades.count() < MIN_TRADES:
        # Penalize invalid trials
        return -np.inf, 0.0, -np.inf, 0.0

    # 4) Calculate and return your three metrics
    sharpe       = portfolio.sharpe_ratio() or -1.0
    win_rate     = portfolio.trades.win_rate() or 0.0
    max_drawdown = (portfolio.max_drawdown() or -1.0) * 100  # Convert to percentage
    final_equity = portfolio.final_value() or 0.0  # Get the final equity value


    return float(sharpe), float(win_rate), float(max_drawdown), float(final_equity)


''' ------------------------------
Part 5 Main Execution
------------------------------ '''
if __name__ == '__main__':
    if 'close_prices' in globals() and isinstance(close_prices, pd.Series) and not close_prices.empty:
        print("\n2) Setting up Optuna study...")

        # --- Setup Study ---
        # Use SQLite for persistence, enabling the dashboard
        study_name = 'Opt_Strategy_EMA_vbt_eurusd_4h' # Unique name for the study
        storage_name = f"sqlite:///{study_name}.db"

        # Define the directions for each objective (must match the order returned by the objective function)
        # Maximize Sharpe, Maximize Win Rate, Maximize Max Drawdown (closer to 0), Maximize Final Equity
        direction = ['maximize', 'maximize', 'maximize', 'maximize']
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,  # Load previous results if study exists
            directions=direction # Specify multi-objective directions
        )

        # --- Run Optimization ---
        n_trials = MC_ITERATIONS # Adjust number of trials as needed
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
            print(f"{'Trial':>5} | {'Sharpe':>10} | {'Win Rate':>10} | {'Max DD':>10} | {'Final Equity':>15} | {'Params':<40}")
            print("-" * 100)
            successful_pareto_count = 0
            for i, trial in enumerate(pareto_trials):
                # <<< START MODIFICATION >>>
                # Check if the trial has the expected number of values before accessing them
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None and len(trial.values) == 4:
                    params_str = ', '.join(f"{k}={v}" for k, v in trial.params.items())
                    # Access values using indices 0, 1, 2
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

            # Check if 'values_0', 'values_1', 'values_2' exist, otherwise skip this section
            required_value_cols = ['values_0', 'values_1', 'values_2']
            if not all(col in completed_trials_df.columns for col in required_value_cols):
                print("Could not find required 'values_x' columns in completed trials DataFrame. Skipping individual bests.")
            elif not completed_trials_df.empty:
                 # Rename columns (make sure these names exist first)
                 completed_trials_df.rename(columns={
                    'values_0': 'sharpe',
                    'values_1': 'win_rate',
                    'values_2': 'max_drawdown',
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
                        # Check values before printing
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

                        # Find best Max Drawdown
                        best_drawdown_idx = completed_trials_df['max_drawdown'].idxmax()
                        best_drawdown_trial_num = int(completed_trials_df.loc[best_drawdown_idx]['number'])
                        best_drawdown_trial = study.trials[best_drawdown_trial_num]
                        print(f"\nBest Max Drawdown Trial (#{best_drawdown_trial.number}):")
                        if best_drawdown_trial.values and len(best_drawdown_trial.values) == 4:
                            print(f"  Metrics: Sharpe={best_drawdown_trial.values[0]:.4f}, WinRate={best_drawdown_trial.values[1]:.2f}, MaxDD={best_drawdown_trial.values[2]:.4f}%, FinalEquity={best_drawdown_trial.values[3]:.2f}")
                            print(f"  Params: {best_drawdown_trial.params}")
                        else:
                            print(f"  Metrics: Incomplete data - {best_drawdown_trial.values}")
                            print(f"  Params: {best_drawdown_trial.params}")


                        # --- Run backtest with best Sharpe parameters ---
                        print("\n--- Running backtest with the 'Best Sharpe Ratio' parameters ---")
                        # Ensure the best_sharpe_trial has valid parameters before running
                        if best_sharpe_trial.params:
                            final_portfolio = run_backtest(close_prices, best_sharpe_trial.params)
                            if final_portfolio is not None:
                                print("\n--- Performance Metrics (Best Sharpe Params) ---")
                                print(final_portfolio.stats())

                                # <<< START: Add Monte Carlo Simulation here >>>
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
print(f"   Run 'optuna-dashboard {storage_name}' in your terminal to view progress.")


''' ------------------------------
Terminal last message 


🏁 Optimization Finished. (1H data)
Number of finished trials: 10000

--- Pareto Optimal Trials (63) ---
Showing results for Pareto optimal trials (trade-offs between objectives):
--------------------------------------------------------------------------------
Trial |     Sharpe |   Win Rate |     Max DD |    Final Equity | Params
----------------------------------------------------------------------------------------------------
  307 |     1.4209 |       0.38 |    -2.8493 |        12300.47 | ema_fast=34, ema_mid=77, ema_slow=254, fixed_sl=35, reward_ratio=3.3
  357 |     1.7922 |       0.45 |    -3.0031 |        12641.22 | ema_fast=33, ema_mid=99, ema_slow=168, fixed_sl=36, reward_ratio=3.1
 1068 |     1.8925 |       0.35 |    -3.0477 |        12274.92 | ema_fast=23, ema_mid=116, ema_slow=206, fixed_sl=18, reward_ratio=5.0
 1198 |     1.6535 |       0.55 |    -1.2820 |        10724.62 | ema_fast=43, ema_mid=93, ema_slow=299, fixed_sl=11, reward_ratio=1.7
 1312 |     1.6271 |       0.39 |    -3.7853 |        12915.06 | ema_fast=29, ema_mid=106, ema_slow=192, fixed_sl=36, reward_ratio=4.4
 1450 |     1.6628 |       0.46 |    -3.5693 |        12267.82 | ema_fast=42, ema_mid=92, ema_slow=242, fixed_sl=38, reward_ratio=2.7
 1490 |     1.6221 |       0.38 |    -1.2661 |        11010.73 | ema_fast=29, ema_mid=77, ema_slow=264, fixed_sl=8, reward_ratio=4.1
 1550 |     1.6023 |       0.51 |    -2.9185 |        12275.29 | ema_fast=32, ema_mid=103, ema_slow=208, fixed_sl=44, reward_ratio=2.0
 1827 |     1.4999 |       0.73 |    -4.5968 |        11339.35 | ema_fast=16, ema_mid=137, ema_slow=299, fixed_sl=46, reward_ratio=0.6
 2405 |     1.7363 |       0.39 |    -1.0716 |        10986.73 | ema_fast=25, ema_mid=79, ema_slow=173, fixed_sl=8, reward_ratio=3.9
 2550 |     1.4913 |       0.65 |    -0.5095 |        10452.99 | ema_fast=35, ema_mid=102, ema_slow=200, fixed_sl=8, reward_ratio=0.9
 2718 |     1.6267 |       0.49 |    -3.6149 |        12386.45 | ema_fast=40, ema_mid=94, ema_slow=159, fixed_sl=44, reward_ratio=2.5
 2954 |     1.3225 |       0.64 |    -1.3420 |        11135.94 | ema_fast=30, ema_mid=66, ema_slow=196, fixed_sl=36, reward_ratio=0.8
 3301 |     1.5742 |       0.55 |    -1.1692 |        10758.79 | ema_fast=33, ema_mid=43, ema_slow=237, fixed_sl=11, reward_ratio=1.3
 3753 |     2.0450 |       0.57 |    -1.4867 |        11702.41 | ema_fast=26, ema_mid=47, ema_slow=135, fixed_sl=20, reward_ratio=1.5
 3862 |     1.4204 |       0.73 |    -2.9858 |        11311.97 | ema_fast=39, ema_mid=69, ema_slow=226, fixed_sl=43, reward_ratio=0.6
 3943 |     1.7160 |       0.53 |    -1.3704 |        11028.10 | ema_fast=39, ema_mid=100, ema_slow=233, fixed_sl=17, reward_ratio=1.8
 3958 |     1.6102 |       0.43 |    -0.7209 |        10765.20 | ema_fast=42, ema_mid=91, ema_slow=238, fixed_sl=10, reward_ratio=2.6
 4108 |     1.5946 |       0.67 |    -0.8060 |        10690.84 | ema_fast=45, ema_mid=88, ema_slow=177, fixed_sl=19, reward_ratio=0.8
 4271 |     1.5020 |       0.75 |    -3.1624 |        11325.22 | ema_fast=34, ema_mid=74, ema_slow=223, fixed_sl=46, reward_ratio=0.5
 4395 |     1.0537 |       0.71 |    -1.6705 |        10829.29 | ema_fast=32, ema_mid=88, ema_slow=138, fixed_sl=43, reward_ratio=0.5
 4473 |     1.5411 |       0.70 |    -0.4938 |        10425.24 | ema_fast=32, ema_mid=97, ema_slow=246, fixed_sl=9, reward_ratio=0.5
 4541 |     1.6129 |       0.52 |    -2.3993 |        11908.61 | ema_fast=39, ema_mid=100, ema_slow=174, fixed_sl=36, reward_ratio=2.2
 4623 |     1.2931 |       0.66 |    -1.4653 |        11209.04 | ema_fast=24, ema_mid=82, ema_slow=180, fixed_sl=44, reward_ratio=0.7
 5388 |     1.2574 |       0.64 |    -0.4758 |        10301.12 | ema_fast=45, ema_mid=113, ema_slow=221, fixed_sl=7, reward_ratio=0.5
 5451 |     1.3014 |       0.59 |    -1.2889 |        10792.04 | ema_fast=27, ema_mid=50, ema_slow=111, fixed_sl=19, reward_ratio=1.0
 5490 |     1.3344 |       0.74 |    -1.3057 |        10807.34 | ema_fast=34, ema_mid=70, ema_slow=135, fixed_sl=33, reward_ratio=0.5
 5492 |     1.6172 |       0.80 |    -2.1056 |        11204.24 | ema_fast=45, ema_mid=88, ema_slow=197, fixed_sl=49, reward_ratio=0.5
 5779 |     1.0779 |       0.71 |    -1.0601 |        10465.38 | ema_fast=26, ema_mid=74, ema_slow=245, fixed_sl=16, reward_ratio=0.5
 5825 |     1.4543 |       0.66 |    -3.5351 |        11558.00 | ema_fast=38, ema_mid=68, ema_slow=204, fixed_sl=44, reward_ratio=0.8
 5973 |     2.0863 |       0.53 |    -1.6133 |        11763.01 | ema_fast=27, ema_mid=53, ema_slow=118, fixed_sl=19, reward_ratio=1.8
 6054 |     1.2747 |       0.64 |    -0.4899 |        10356.95 | ema_fast=29, ema_mid=47, ema_slow=151, fixed_sl=6, reward_ratio=0.5
 6444 |     1.4329 |       0.39 |    -3.9000 |        12928.33 | ema_fast=31, ema_mid=93, ema_slow=146, fixed_sl=44, reward_ratio=4.1
 6526 |     1.5497 |       0.47 |    -3.3686 |        12562.90 | ema_fast=32, ema_mid=92, ema_slow=242, fixed_sl=48, reward_ratio=2.4
 6820 |     1.7211 |       0.72 |    -1.8129 |        11413.39 | ema_fast=44, ema_mid=90, ema_slow=158, fixed_sl=43, reward_ratio=0.8
 6949 |     1.5847 |       0.47 |    -1.1207 |        10887.82 | ema_fast=27, ema_mid=76, ema_slow=233, fixed_sl=10, reward_ratio=2.4
 7011 |     1.6771 |       0.43 |    -3.3116 |        12824.16 | ema_fast=47, ema_mid=92, ema_slow=187, fixed_sl=38, reward_ratio=4.4
 7040 |     1.6931 |       0.46 |    -4.6100 |        12301.29 | ema_fast=20, ema_mid=132, ema_slow=185, fixed_sl=34, reward_ratio=2.6
 7373 |     1.6156 |       0.76 |    -2.0021 |        11132.49 | ema_fast=48, ema_mid=90, ema_slow=149, fixed_sl=43, reward_ratio=0.6
 7404 |     1.7922 |       0.54 |    -0.6099 |        10726.99 | ema_fast=47, ema_mid=83, ema_slow=186, fixed_sl=10, reward_ratio=1.8
 7442 |     1.5729 |       0.45 |    -1.3152 |        10916.79 | ema_fast=39, ema_mid=99, ema_slow=204, fixed_sl=13, reward_ratio=2.7
 7458 |     1.9773 |       0.64 |    -1.8434 |        11654.95 | ema_fast=13, ema_mid=146, ema_slow=279, fixed_sl=27, reward_ratio=1.1
 7461 |     1.9237 |       0.48 |    -1.4664 |        11157.16 | ema_fast=47, ema_mid=92, ema_slow=297, fixed_sl=14, reward_ratio=2.7
 7492 |     1.8580 |       0.51 |    -1.4520 |        11643.04 | ema_fast=25, ema_mid=48, ema_slow=137, fixed_sl=19, reward_ratio=1.8
 7596 |     1.6048 |       0.41 |    -3.5913 |        12902.34 | ema_fast=45, ema_mid=75, ema_slow=127, fixed_sl=40, reward_ratio=4.2
 7783 |     1.5708 |       0.63 |    -0.5194 |        10538.43 | ema_fast=47, ema_mid=80, ema_slow=179, fixed_sl=13, reward_ratio=0.9
 8075 |     1.5466 |       0.58 |    -2.4872 |        11794.96 | ema_fast=42, ema_mid=92, ema_slow=182, fixed_sl=43, reward_ratio=1.5
 8084 |     1.5015 |       0.72 |    -0.7174 |        10365.13 | ema_fast=39, ema_mid=126, ema_slow=254, fixed_sl=8, reward_ratio=0.7
 8209 |     1.6998 |       0.57 |    -3.3349 |        12038.73 | ema_fast=49, ema_mid=83, ema_slow=184, fixed_sl=39, reward_ratio=1.9
 8262 |     1.7567 |       0.51 |    -2.4365 |        12244.81 | ema_fast=38, ema_mid=103, ema_slow=183, fixed_sl=36, reward_ratio=2.4
 8272 |     1.5024 |       0.52 |    -1.2861 |        10813.82 | ema_fast=24, ema_mid=81, ema_slow=295, fixed_sl=13, reward_ratio=1.6
 8383 |     1.5138 |       0.47 |    -1.1508 |        10955.87 | ema_fast=24, ema_mid=74, ema_slow=212, fixed_sl=12, reward_ratio=2.2
 8411 |     1.4357 |       0.46 |    -1.1284 |        10898.07 | ema_fast=25, ema_mid=73, ema_slow=229, fixed_sl=12, reward_ratio=2.2
 8559 |     1.7484 |       0.51 |    -0.9339 |        10871.94 | ema_fast=46, ema_mid=88, ema_slow=148, fixed_sl=13, reward_ratio=2.2
 8710 |     1.6050 |       0.64 |    -0.5508 |        10562.65 | ema_fast=44, ema_mid=90, ema_slow=175, fixed_sl=13, reward_ratio=0.9
 8977 |     1.7340 |       0.52 |    -1.3603 |        11051.98 | ema_fast=36, ema_mid=46, ema_slow=120, fixed_sl=14, reward_ratio=1.7
 9085 |     1.6819 |       0.77 |    -2.0031 |        11297.18 | ema_fast=46, ema_mid=87, ema_slow=198, fixed_sl=45, reward_ratio=0.6
 9089 |     1.5127 |       0.53 |    -1.3224 |        10816.98 | ema_fast=36, ema_mid=101, ema_slow=189, fixed_sl=14, reward_ratio=1.7
 9125 |     1.4288 |       0.57 |    -1.4134 |        10774.37 | ema_fast=48, ema_mid=85, ema_slow=217, fixed_sl=19, reward_ratio=1.3
 9140 |     1.6740 |       0.57 |    -1.4181 |        11183.13 | ema_fast=25, ema_mid=54, ema_slow=110, fixed_sl=19, reward_ratio=1.3
 9289 |     1.4471 |       0.66 |    -2.8822 |        11463.94 | ema_fast=41, ema_mid=74, ema_slow=202, fixed_sl=43, reward_ratio=0.8
 9421 |     1.5816 |       0.62 |    -1.4910 |        10814.66 | ema_fast=38, ema_mid=101, ema_slow=257, fixed_sl=18, reward_ratio=1.1
 9654 |     1.6123 |       0.57 |    -2.3004 |        11967.61 | ema_fast=38, ema_mid=100, ema_slow=151, fixed_sl=41, reward_ratio=1.8
--------------------------------------------------------------------------------

--- Trials with highest individual metrics (among all completed trials) ---

Best Sharpe Ratio Trial (#5973):
  Metrics: Sharpe=2.0863, WinRate=0.53, MaxDD=-1.6133%, FinalEquity=11763.01
  Params: {'ema_fast': 27, 'ema_mid': 53, 'ema_slow': 118, 'fixed_sl': 19, 'reward_ratio': 1.8}

Best Win Rate Trial (#5492):
  Metrics: Sharpe=1.6172, WinRate=0.80, MaxDD=-2.1056%, FinalEquity=11204.24
  Params: {'ema_fast': 45, 'ema_mid': 88, 'ema_slow': 197, 'fixed_sl': 49, 'reward_ratio': 0.5}

Best Max Drawdown Trial (#5388):
  Metrics: Sharpe=1.2574, WinRate=0.64, MaxDD=-0.4758%, FinalEquity=10301.12
  Params: {'ema_fast': 45, 'ema_mid': 113, 'ema_slow': 221, 'fixed_sl': 7, 'reward_ratio': 0.5}

--- Running backtest with the 'Best Sharpe Ratio' parameters ---

--- Performance Metrics (Best Sharpe Params) ---
Start                               2020-04-23 00:00:00
End                                 2025-04-22 23:00:00
Period                               1299 days 05:00:00
Start Value                                     10000.0
End Value                                   11763.01109
Total Return [%]                              17.630111
Benchmark Return [%]                            4.98742
Max Gross Exposure [%]                            100.0
Total Fees Paid                                     0.0
Max Drawdown [%]                               1.613257
Max Drawdown Duration                 102 days 15:00:00
Total Trades                                        156
Total Closed Trades                                 156
Total Open Trades                                     0
Open Trade PnL                                      0.0
Win Rate [%]                                  53.205128
Best Trade [%]                                 1.105457
Worst Trade [%]                               -0.429839
Avg Winning Trade [%]                          0.397684
Avg Losing Trade [%]                          -0.228453
Avg Winning Trade Duration    0 days 17:09:23.855421686
Avg Losing Trade Duration     0 days 12:21:22.191780821
Profit Factor                                  1.972652
Expectancy                                    11.301353
Sharpe Ratio                                   2.086255
Calmar Ratio                                   2.893162
Omega Ratio                                    1.288872
Sortino Ratio                                  3.580107
dtype: object
   Run 'optuna-dashboard sqlite:///Opt_Strategy_EMA_vbt_eurusd_4h.db' in your terminal to view progress.

   
---------------------------------------------------'''