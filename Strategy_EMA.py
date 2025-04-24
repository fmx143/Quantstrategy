import backtrader as bt
import pandas as pd


''' ------------------------------
Part 1, import data
'''
# Data usage (the data should be a Pandas DataFrame with OHLC data):
try:
    data = pd.read_csv(r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_4H_5y_cleaned.csv')
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)
    print("1) Data loaded successfully.")

except FileNotFoundError:
    print("1.1) Error: Data file not found. Please check the path.")
    exit() # Exit if data can't be loaded
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()


''' ------------------------------
Part 2, define the strategy, indicators, and entry/exit conditions
'''
class EmaCrossoverStrategy(bt.Strategy):
    params = (
        ('ema_fast', 10),       # Fast EMA period (will be optimized)
        ('ema_mid', 50),        # Mid EMA period (will be optimized)
        ('ema_slow', 200),      # Slow EMA period (will be optimized)
        ('reward_ratio', 3.0),  # Reward-to-risk ratio (will be optimized)
        ('fixed_sl', 20),       # Stop-loss in pips (will be optimized)
    )

    def __init__(self):
        # Define EMAs
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_mid = bt.ind.EMA(self.data.close, period=self.p.ema_mid)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)

        # Crossovers conditions
        self.cross_fast_mid = bt.ind.CrossOver(self.ema_fast, self.ema_mid)

        # Track orders (stats will be collected by analyzers)
        self.order = None

    def market_condition_buy(self):
        """Check if the market condition is suitable for a buy."""
        return self.ema_fast[0] > self.ema_mid[0] and self.ema_mid[0] > self.ema_slow[0]

    def market_condition_sell(self):
        """Check if the market condition is suitable for a sell."""
        return self.ema_fast[0] < self.ema_mid[0] and self.ema_mid[0] < self.ema_slow[0]

    def next(self):
        # If an order is pending, skip
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Calculate stop loss distance in price terms
            sl_pips = self.p.fixed_sl
            # Assuming EURUSD or similar pair where pip value is 0.0001
            # considers the quote currency and pair type (e.g., USDJPY).
            pip_value = 0.0001
            sl_amount = sl_pips * pip_value

            # --- Entry conditions: ---
            # 1) Fast EMA crosses above Mid EMA and both are above Slow EMA
            if self.cross_fast_mid[0] > 0 and self.market_condition_buy():
                buy_price = self.data.close[0]
                buy_stop_loss_price = buy_price - sl_amount
                buy_take_profit_price = buy_price + (buy_price - buy_stop_loss_price) * self.p.reward_ratio # Needs reward_ratio param

                # Place buy order (stop loss handled by exit logic below for simplicity now)
                # Using buy_bracket directly might be complex if TP is also dynamic or based on exits.
                self.order = self.buy()
                self.sl_price = buy_stop_loss_price
                self.tp_price = buy_take_profit_price


            # 2) Fast EMA crosses below Mid EMA and both are below Slow EMA
            elif self.cross_fast_mid[0] < 0 and self.market_condition_sell():
                sell_price = self.data.close[0]
                sell_stop_loss_price = sell_price + sl_amount
                sell_take_profit_price = sell_price - (sell_stop_loss_price - sell_price) * self.p.reward_ratio # Needs reward_ratio param

                # Place sell order
                self.order = self.sell()
                self.sl_price = sell_stop_loss_price
                self.tp_price = sell_take_profit_price

        else: # We are in the market, check exit conditions
            current_price = self.data.close[0]

            # --- Exit conditions: ---
            if self.position.size > 0: # Long position
                # 1. Fixed Stop Loss Hit
                if current_price <= self.sl_price:
                    self.order = self.close()
                elif current_price >= self.tp_price:
                    self.order = self.close()
                # 3. Exit on crossover back
                elif self.cross_fast_mid[0] < 0:
                    self.order = self.close()

            elif self.position.size < 0: # Short position
                # 1. Fixed Stop Loss Hit
                if current_price >= self.sl_price:
                    self.order = self.close()
                # 2. Optional: Take Profit Hit (if defined)
                elif current_price <= self.tp_price:
                    self.order = self.close()
                # 3. Exit on crossover back
                elif self.cross_fast_mid[0] > 0:
                    self.order = self.close()


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # An order has been submitted/accepted - Nothing to do
            return
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # print(f'Order Canceled/Margin/Rejected: Status {order.getstatusname()}')
            pass # Reduce noise

        # Reset order variable irrespective of status after completion/rejection etc.
        self.order = None

''' ------------------------------
Part 3, OPTIMIZE the strategy
'''
def optimize_strategy(data):
    cerebro = bt.Cerebro(optreturn=False) # optreturn=False faster for large optimizations

    # --- Define Parameter Ranges ---
    # IMPORTANT: Keep the number of combinations reasonable, otherwise it takes very long!
    # Total combinations = len(range1) * len(range2) * ...
    cerebro.optstrategy(
        EmaCrossoverStrategy,
        ema_fast=range(1, 15, 1),       # Example: 1, 2, 3, 4, 5
        ema_mid=range(40, 60, 1),       # Example: 40, 41, 42, 43
        ema_slow=range(100, 200, 10),    # Example: 100, 110, 120
        # Add other parameters here if needed, e.g. reward_ratio
        fixed_sl=range(5, 30, 5),     # Example: 15, 20, 25, 30, 35
        reward_ratio=(1, 5, 0.1)
    )

    # Add a PercentSizer
    cerebro.addsizer(bt.sizers.PercentSizer, percents=2) # Using 2% size as example

    # Add the data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set the initial cash
    cerebro.broker.setcash(10000)
    # Optional: Set commission (e.g., 0.1% per trade)
    # cerebro.broker.setcommission(commission=0.001)

    # Add analyzers (These will be run for each parameter combination)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer,   _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,     _name='sharpe', timeframe=bt.TimeFrame.Months) # Adjust timeframe if needed
    cerebro.addanalyzer(bt.analyzers.DrawDown,        _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns,         _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN,             _name='sqn') # System Quality Number

    # --- Run the Optimization ---
    print('🚀 Starting Optimization... This may take a while.')
    opt_results = cerebro.run(maxcpus=None) # Use maxcpus=None to use all available cores, or set specific number
    print('🏁 Optimization Finished.')

    # --- Process Optimization Results ---
    final_results_list = []
    for run in opt_results:
        for strategy in run: # opt_results is a list of lists, outer list per param set
            params = strategy.params
            analyzers = strategy.analyzers
            trade_analysis = analyzers.trade_analyzer.get_analysis()
            sharpe_analysis = analyzers.sharpe.get_analysis()
            drawdown_analysis = analyzers.drawdown.get_analysis()
            returns_analysis = analyzers.returns.get_analysis()
            sqn_analysis = analyzers.sqn.get_analysis()

            # Basic check if trades were made
            total_trades = trade_analysis.total.closed if trade_analysis.total is not None else 0
            win_rate = (trade_analysis.won.total / total_trades * 100) if total_trades > 0 else 0
            avg_win = (trade_analysis.won.pnl.total / trade_analysis.won.total) if trade_analysis.won.total > 0 else 0
            avg_loss = (trade_analysis.lost.pnl.total / trade_analysis.lost.total) if trade_analysis.lost.total > 0 else 0

            final_results_list.append({
                'ema_fast': params.ema_fast,
                'ema_mid': params.ema_mid,
                'ema_slow': params.ema_slow,
                'fixed_sl': params.fixed_sl,
                'reward_ratio': params.reward_ratio,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_analysis['sharperatio'] if sharpe_analysis else None,
                'max_drawdown': drawdown_analysis.max.drawdown if drawdown_analysis.max else None,
                'total_return': returns_analysis['rtot'] if returns_analysis else None,
                'sqn': sqn_analysis['sqn'] if sqn_analysis else None,
                'final_value': strategy.broker.getvalue() # Get final portfolio value
            })

    # --- Analyze and Display Best Results ---
    if not final_results_list:
        print("No results generated from optimization.")
        return

    # Create a Pandas DataFrame for easier analysis
    results_df = pd.DataFrame(final_results_list)
    # Adjust Pandas display options
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Adjust the width to fit the terminal
    pd.set_option('display.max_rows', None)     # Show all rows (optional, if needed)

    # Sort results by a chosen metric (e.g., Sharpe Ratio, descending)
    # results_df_sorted = results_df.sort_values(by='sharpe_ratio', ascending=False, na_position='last')
    # Alternative sorting: by final value or SQN
    results_df_sorted = results_df.sort_values(by='final_value', ascending=False)
    # results_df_sorted = results_df.sort_values(by='sqn', ascending=False, na_position='last')


    print("\n--- Optimization Results Summary (Sorted by final value) ---")
    # Display top N results
    print(results_df_sorted.head(5)) # Show top 5 results

    # You can also filter results, e.g., minimum number of trades
    min_trades = 5
    filtered_results = results_df[results_df['total_trades'] >= min_trades]
    if not filtered_results.empty:
        filtered_sorted = filtered_results.sort_values(by='sharpe_ratio', ascending=False, na_position='last')
        print(f"\n--- Top Results with at least {min_trades} trades (Sorted by final valu) ---")
        print(filtered_sorted.head(10))
    else:
        print(f"\nNo parameter combinations resulted in at least {min_trades} trades.")

    # Find the single best result based on Sharpe
    best_result = results_df_sorted.iloc[0]
    print("\n--- Best Result (based on Sharpe Ratio) ---")
    print(f"Parameters: ema_fast={best_result['ema_fast']}, ema_mid={best_result['ema_mid']}, ema_slow={best_result['ema_slow']}, fixed_sl={best_result['fixed_sl']}")
    print(f"Final Portfolio Value: {best_result['final_value']:.2f}")
    print(f"Total Trades: {best_result['total_trades']}")
    print(f"Win Rate: {best_result['win_rate']:.2f}%")
    print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.2f}%")
    print(f"Total Return: {best_result['total_return']*100:.2f}%")
    print(f"SQN: {best_result['sqn']:.2f}")


if __name__ == '__main__':
    # Ensure 'data' is loaded correctly before calling
    if 'data' in globals() and isinstance(data, pd.DataFrame) and not data.empty:
        optimize_strategy(data)
    else:
        print("Data was not loaded properly. Exiting.")