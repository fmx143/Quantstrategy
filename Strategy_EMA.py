import backtrader as bt
import pandas as pd
import numpy as np # Import numpy if not already

''' ------------------------------
Part 1, import data (Keep as is)
------------------------------ '''
try:
    # Using the 4H data as specified in the original request
    data_path = r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_4H_5y_cleaned.csv'
    price_df = pd.read_csv(data_path)
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'])
    price_df.set_index('Datetime', inplace=True)
    price_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    price_df.sort_index(inplace=True)
    print(f"1) Data loaded successfully. Shape: {price_df.shape}")
except FileNotFoundError:
    print(f"1.1) Error: Data file not found at {data_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()


''' ------------------------------
Part 2, define the strategy (MODIFIED)
------------------------------ '''
class EmaCrossoverStrategy(bt.Strategy):
    # Parameters will be passed during instantiation
    params = (
        ('ema_fast', None),      # Required: Period for fast EMA
        ('ema_mid', None),       # Required: Period for mid EMA
        ('ema_slow', None),      # Required: Period for slow EMA
        ('reward_ratio', None),  # Required: Reward:Risk ratio for TP
        ('fixed_sl', None),      # Required: Stop-loss in pips
        ('pip_value', 0.0001),   # Pip size for EUR/USD
    )

    def __init__(self):
        # Validate parameters
        if None in [self.p.ema_fast, self.p.ema_mid, self.p.ema_slow, self.p.reward_ratio, self.p.fixed_sl]:
            raise ValueError("Strategy parameters (ema_fast, ema_mid, ema_slow, reward_ratio, fixed_sl) must be provided.")

        # Define indicators
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_mid = bt.ind.EMA(self.data.close, period=self.p.ema_mid)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.cross_fast_mid = bt.ind.CrossOver(self.ema_fast, self.ema_mid)

        # Track orders to prevent multiple orders before fill
        self.order = None

    def market_condition_buy(self):
        # Market Trend Filter: Fast > Mid > Slow EMA
        return self.ema_fast[0] > self.ema_mid[0] and self.ema_mid[0] > self.ema_slow[0]

    def market_condition_sell(self):
        # Market Trend Filter: Fast < Mid < Slow EMA
        return self.ema_fast[0] < self.ema_mid[0] and self.ema_mid[0] < self.ema_slow[0]

    def next(self):
        # If an order is pending or we are already in the market, do nothing
        if self.order or self.position:
            return

        # Calculate SL and TP amounts in price terms
        sl_pips = self.p.fixed_sl
        pip_value = self.p.pip_value
        sl_amount = sl_pips * pip_value

        # Ensure reward_ratio is positive before calculating TP
        if self.p.reward_ratio <= 0:
             tp_amount = float('inf') # Effectively disable TP if ratio is non-positive
        else:
             tp_amount = sl_amount * self.p.reward_ratio

        current_close = self.data.close[0] # Use current close as reference for SL/TP calculation

        # --- Entry conditions with Bracket Orders ---
        # Buy Signal: Fast crosses above Mid AND trend filter is bullish
        if self.cross_fast_mid[0] > 0 and self.market_condition_buy():
            sl_price = current_close - sl_amount
            tp_price = current_close + tp_amount
            # Place Market Order with attached SL and TP
            self.order = self.buy_bracket(
                stopprice=sl_price,
                limitprice=tp_price,
                exectype=bt.Order.Market # Execute at market on next bar open
            )
            # print(f"[{self.data.datetime.date(0)}] BUY Bracket Sent: Entry Ref={current_close:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}")


        # Sell Signal: Fast crosses below Mid AND trend filter is bearish
        elif self.cross_fast_mid[0] < 0 and self.market_condition_sell():
            sl_price = current_close + sl_amount
            tp_price = current_close - tp_amount
            # Place Market Order with attached SL and TP
            self.order = self.sell_bracket(
                stopprice=sl_price,
                limitprice=tp_price,
                exectype=bt.Order.Market # Execute at market on next bar open
            )
            # print(f"[{self.data.datetime.date(0)}] SELL Bracket Sent: Entry Ref={current_close:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}")


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Active order - Do nothing further unless it's the main order confirmation
            if order.isbuy() or order.issell(): # Was it the main buy/sell order?
                # print(f"Order {order.getstatusname()}: Type {'Buy' if order.isbuy() else 'Sell'}, Ref: {order.ref}")
                pass
            return # Don't reset self.order until completion/failure

        if order.status == order.Completed:
            # Check if it was the main entry order or one of the SL/TP children
            price = order.executed.price
            size = order.executed.size
            comm = order.executed.comm
            # if order.isbuy() or order.issell(): # Main entry order executed
            #     print(f"ORDER EXECUTED: {'BUY' if order.isbuy() else 'SELL'}, Price: {price:.5f}, Size: {size}, Comm: {comm:.2f}")
            # elif order.isstop(): # Stop loss executed
            #     print(f"STOP LOSS HIT: Price: {price:.5f}, Size: {size}, Comm: {comm:.2f}")
            # elif order.islimit(): # Take profit executed
            #     print(f"TAKE PROFIT HIT: Price: {price:.5f}, Size: {size}, Comm: {comm:.2f}")

            # Reset order tracking ONLY when the order chain completes (entry or exit)
            # Since we prevent new orders while self.position exists, we can reset here
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # print(f'Order Failed/Cancelled: {order.getstatusname()}')
            self.order = None # Allow new orders


''' ------------------------------
Part 3, Backtest the strategy (MODIFIED)
------------------------------ '''
def run_backtrader_test(data, params):
    cerebro = bt.Cerebro(stdstats=True) # Disable standard observers if plotting manually or just want analyzers

    # Add strategy with the provided parameters
    cerebro.addstrategy(EmaCrossoverStrategy, **params)

    # Add the data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set the initial cash (must match Script 1)
    initial_capital = 10000
    cerebro.broker.setcash(initial_capital)

    # Set commission to zero (must match Script 1)
    cerebro.broker.setcommission(commission=0.0)

    # Set sizer to approximate full investment (must match Script 1 intent)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=99)

    # Add analyzers (use names consistent with Script 1 where possible)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    # Ensure Sharpe uses 4H timeframe and zero risk-free rate for closer comparison
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Minutes, compression=240, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn') # System Quality Number, useful metric

    # Run the backtest
    print(f"\n🚀 Starting Backtrader Run with Params: {params}")
    results = cerebro.run()
    strategy_instance = results[0] # Get the first strategy instance

    # Retrieve analyzer results
    final_value = cerebro.broker.getvalue()
    trade_analysis = strategy_instance.analyzers.trade_analyzer.get_analysis()
    sharpe_analysis = strategy_instance.analyzers.sharpe.get_analysis()
    drawdown_analysis = strategy_instance.analyzers.drawdown.get_analysis()
    returns_analysis = strategy_instance.analyzers.returns.get_analysis()
    sqn_analysis = strategy_instance.analyzers.sqn.get_analysis()

    # --- Display results ---
    print("\n--- Backtrader Results ---")
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Net Profit: {final_value - initial_capital:.2f}")
    print(f"Total Return: {returns_analysis.get('rtot', 'N/A') * 100:.2f}%" if isinstance(returns_analysis.get('rtot'), (int, float)) else "Total Return: N/A")
    print(f"Sharpe Ratio: {sharpe_analysis.get('sharperatio', 'N/A'):.3f}" if isinstance(sharpe_analysis.get('sharperatio'), (int, float)) else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {drawdown_analysis.max.get('drawdown', 'N/A'):.2f}%" if isinstance(drawdown_analysis.max.get('drawdown'), (int, float)) else "Max Drawdown: N/A")
    print(f"SQN: {sqn_analysis.get('sqn', 'N/A'):.2f}" if isinstance(sqn_analysis.get('sqn'), (int, float)) else "SQN: N/A")

    # Trade analysis details
    total_trades = trade_analysis.total.get('closed', 0)
    won_trades = trade_analysis.won.get('total', 0)
    lost_trades = trade_analysis.lost.get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    avg_win = trade_analysis.won.pnl.get('average', 0)
    avg_loss = trade_analysis.lost.pnl.get('average', 0)

    print("\n--- Trade Analysis ---")
    print(f"Total Closed Trades: {total_trades}")
    print(f"Winning Trades: {won_trades}")
    print(f"Losing Trades: {lost_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win PnL: {avg_win:.2f}")
    print(f"Average Loss PnL: {avg_loss:.2f}")

    # Plot the results - this will show the equity curve in the top panel
    print("\nGenerating plot...")
    cerebro.plot(style='candlestick', barup='green', bardown='red') # You can adjust style if needed
    print("Plot window opened (or saved if running non-interactively).")


if __name__ == '__main__':
    if 'price_df' in globals() and isinstance(price_df, pd.DataFrame) and not price_df.empty:

        # --- !!! IMPORTANT: PASTE OPTUNA RESULTS HERE !!! ---
        # Replace these values with the parameters from your best Optuna trial (e.g., best Sharpe)
        best_params_from_optuna = {
            'ema_fast': 35,      # Example: Replace with actual Optuna result
            'ema_mid': 85,       # Example: Replace with actual Optuna result
            'ema_slow': 247,     # Example: Replace with actual Optuna result
            'fixed_sl': 11,      # Example: Replace with actual Optuna result (in pips)
            'reward_ratio': 0.6  # Example: Replace with actual Optuna result
        }
        # --- !!! IMPORTANT !!! ---

        # Run the backtest using the loaded data and the best parameters
        run_backtrader_test(price_df, best_params_from_optuna)

    else:
        print("Data was not loaded properly. Exiting.")