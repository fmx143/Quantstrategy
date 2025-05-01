import vectorbt as vbt
import pandas as pd
import numpy as np

''' ------------------------------
Part 1, import data (Keep as is)
------------------------------ '''
try:
    # Using the 4H data as specified in the original request
    data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_1H_5y_clean.csv'
    price_df = pd.read_csv(data_path)
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'])
    price_df.set_index('Datetime', inplace=True)

    # Ensure standard column names (vectorbt prefers lowercase)
    price_df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'  # Keep volume if available/needed
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

''' ------------------------
Part 2, define the strategy
-------------------------'''
def run_vectorbt_test(data, params):
    # Extract parameters
    ema_fast = params['ema_fast']
    ema_mid = params['ema_mid']
    ema_slow = params['ema_slow']
    fixed_sl = params['fixed_sl']
    reward_ratio = params['reward_ratio']
    pip_value = 0.0001
    init_cash = 10000
    fees = 0.0
    freq = '1h'

    print(f"\n🚀 Starting VectorBT Run with Params: {params}")

    # Dynamically calculate bars per year based on the data frequency
    def get_bars_per_year(freq):
        if freq == '1d':  # Daily data
            return 252
        elif freq == '1h':  # Hourly data
            return 252 * 24
        elif freq == '4h':  # 4-hour data
            return 252 * 6
        elif freq == '15min':  # 15-minute data
            return 252 * 24 * 4
        elif freq == '5min':  # 5-minute data
            return 252 * 24 * 12
        elif freq == '1min':  # 1-minute data
            return 252 * 24 * 60
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

    # Calculate true EMAs
    fast_ema = vbt.MA.run(data['close'], window=ema_fast, ewm=True).ma
    mid_ema  = vbt.MA.run(data['close'], window=ema_mid,  ewm=True).ma
    slow_ema = vbt.MA.run(data['close'], window=ema_slow, ewm=True).ma

    # Market condition filters
    cond_long = (fast_ema > mid_ema) & (mid_ema > slow_ema)
    cond_short= (fast_ema < mid_ema) & (mid_ema < slow_ema)

    # Crossover signals
    entries_long  = fast_ema.vbt.crossed_above(mid_ema)  & cond_long
    entries_short = fast_ema.vbt.crossed_below(mid_ema) & cond_short

    # SL/TP percentages
    sl_pct = (fixed_sl * pip_value) / data['close']
    tp_pct = sl_pct * reward_ratio

    # Run portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=entries_long,
        exits=None,
        short_entries=entries_short,
        short_exits=None,
        sl_stop=sl_pct,
        tp_stop=tp_pct,
        init_cash=init_cash,
        fees=fees,
        freq=freq
    )

    # --- Display results ---
    print("\n--- VectorBT Results ---")
    final_value_series = portfolio.value()  # Call the method to get the value series
    final_value = final_value_series.iloc[-1]  # Get the final portfolio value
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Net Profit: {final_value - init_cash:.2f}")
    print(f"Total Return: {((final_value / init_cash) - 1) * 100:.2f}%")

    # Get Sharpe ratio
    returns = portfolio.returns()
    bars_per_year = get_bars_per_year(freq)
    sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
    print(f"Sharpe Ratio: {sharpe:.3f}")

    # Get drawdown
    drawdown = portfolio.drawdown()
    max_dd = drawdown.min() * 100
    print(f"Max Drawdown: {max_dd:.2f}%")

    # Calculate SQN manually
    trades = portfolio.trades
    if len(trades) > 0:
        pnl = trades.pnl
        sqn = pnl.mean() / pnl.std() * np.sqrt(len(pnl))
        print(f"SQN: {sqn:.2f}")
    else:
        print("SQN: N/A (no trades)")

    # Trade analysis details
    total_trades = len(trades)
    if total_trades > 0:
        # Use the `pnl` attribute to get the PnL of each trade
        pnl_values = trades.pnl.values
        won_trades = pnl_values[pnl_values > 0]
        lost_trades = pnl_values[pnl_values <= 0]
        win_rate = (len(won_trades) / total_trades * 100)

        avg_win = won_trades.mean() if len(won_trades) > 0 else 0
        avg_loss = lost_trades.mean() if len(lost_trades) > 0 else 0

        print("\n--- Trade Analysis ---")
        print(f"Total Closed Trades: {total_trades}")
        print(f"Winning Trades: {len(won_trades)}")
        print(f"Losing Trades: {len(lost_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win PnL: {avg_win:.2f}")
        print(f"Average Loss PnL: {avg_loss:.2f}")
    else:
        print("\n--- Trade Analysis ---")
        print("No trades executed")

    # Generate and show plots
    print("\nGenerating plot...")
    fig = portfolio.plot().show()

    return portfolio, fig

if __name__ == '__main__':
    if 'price_df' in globals() and isinstance(price_df, pd.DataFrame) and not price_df.empty:
        # --- Replace these values with the parameters from your best Optuna trial (e.g., best Sharpe)
        best_params_from_optuna = {
            'ema_fast': 27,      # Example: Replace with actual Optuna result
            'ema_mid': 53,       # Example: Replace with actual Optuna result
            'ema_slow': 118,     # Example: Replace with actual Optuna result
            'fixed_sl': 19,      # Example: Replace with actual Optuna result (in pips)
            'reward_ratio': 1.8 # Example: Replace with actual Optuna result
        }

        # Run the backtest using the loaded data and the best parameters
        portfolio, fig = run_vectorbt_test(price_df, best_params_from_optuna)
    else:
        print("Data was not loaded properly. Exiting.")

'''-------------------------
Last terminal output (1H data)

1) Data loaded successfully. Shape: (31181, 5)

🚀 Starting VectorBT Run with Params: {'ema_fast': 27, 'ema_mid': 53, 'ema_slow': 118, 'fixed_sl': 19, 'reward_ratio': 1.8}

--- VectorBT Results ---
Final Portfolio Value: 11763.01
Net Profit: 1763.01
Total Return: 17.63%
Sharpe Ratio: 1.733
Max Drawdown: -1.61%
SQN: 3.90

--- Trade Analysis ---
Total Closed Trades: 156
Winning Trades: 83
Losing Trades: 73
Win Rate: 53.21%
Average Win PnL: 43.08
Average Loss PnL: -24.83

Generating plot...

------------------------'''