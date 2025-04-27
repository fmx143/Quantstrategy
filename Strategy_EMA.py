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
Part 3, Backtest the strategy
'''
def strategy(data):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EmaCrossoverStrategy, ema_fast=10, ema_mid=50, ema_slow=200, fixed_sl=20, reward_ratio=3.0)
    # Add a PercentSizer
    cerebro.addsizer(bt.sizers.PercentSizer, percents=2) # Using 2% size as example

    # Add the data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set the initial cash
    cerebro.broker.setcash(10000)
    # Optional: Set commission (e.g., 0.1% per trade)
    # cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Months)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    # Run the backtest
    print("\n🚀 Starting Backtest...")
    results = cerebro.run()
    strategy = results[0]

    # Retrieve analyzer results
    trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    returns_analysis = strategy.analyzers.returns.get_analysis()
    sqn_analysis = strategy.analyzers.sqn.get_analysis()

    # Display results
    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    print(f"Net Profit: {cerebro.broker.getvalue() - 10000:.2f}")
    print(f"Sharpe Ratio: {sharpe_analysis['sharperatio']:.2f}" if 'sharperatio' in sharpe_analysis else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {drawdown_analysis.max.drawdown:.2f}%")
    print(f"Total Return: {returns_analysis['rtot'] * 100:.2f}%" if 'rtot' in returns_analysis else "Total Return: N/A")
    print(f"SQN: {sqn_analysis['sqn']:.2f}" if 'sqn' in sqn_analysis else "SQN: N/A")

    # Trade analysis
    print("\n--- Trade Analysis ---")
    print(f"Total Trades: {trade_analysis.total.closed if 'total' in trade_analysis and 'closed' in trade_analysis.total else 'N/A'}")
    print(f"Winning Trades: {trade_analysis.won.total if 'won' in trade_analysis and 'total' in trade_analysis.won else 'N/A'}")
    print(f"Losing Trades: {trade_analysis.lost.total if 'lost' in trade_analysis and 'total' in trade_analysis.lost else 'N/A'}")
    print(f"Win Rate: {trade_analysis.won.total / trade_analysis.total.closed * 100:.2f}%" if 'won' in trade_analysis and 'total' in trade_analysis.won and 'closed' in trade_analysis.total else "Win Rate: N/A")
    print(f"Average Win: {trade_analysis.won.pnl.average:.2f}" if 'won' in trade_analysis and 'pnl' in trade_analysis.won and 'average' in trade_analysis.won.pnl else "Average Win: N/A")
    print(f"Average Loss: {trade_analysis.lost.pnl.average:.2f}" if 'lost' in trade_analysis and 'pnl' in trade_analysis.lost and 'average' in trade_analysis.lost.pnl else "Average Loss: N/A")

    # Plot the results (optional)
    cerebro.plot()

if __name__ == '__main__':
    # Ensure 'data' is loaded correctly before calling
    if 'data' in globals() and isinstance(data, pd.DataFrame) and not data.empty:
        strategy(data)
    else:
        print("Data was not loaded properly. Exiting.")