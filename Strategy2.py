'''
The most profitable Daily Bias Strategy [EBP- 88% Proven Win Rate]
https://www.youtube.com/watch?v=XiU9JHUM5NE
'''

import backtrader as bt
import matplotlib as plt
import pandas as pd
from apy import *

# Usage (the data should be a Pandas DataFrame with OHLC data):
data = pd.read_csv(Backtrader_gold_daily)  # Replace with your data source
data['datetime'] = pd.to_datetime(data['datetime'])  # Ensure 'datetime' column is in datetime format
data.set_index('datetime', inplace=True)  # Set the 'datetime' column as the index

# Data for cerebro engine
data_feed = bt.feeds.PandasData(dataname=data)


class EngulfingBarStrategy(bt.Strategy):
    def __init__(self):
        # Keep references to the data feeds
        self.data_low = self.data.low
        self.data_high = self.data.high
        self.data_open = self.data.open
        self.data_close = self.data.close
        self.dol = None  # Draw on Liquidity level

    def next(self):
        # Previous bar values
        prev_bar_low = self.data_low[-2]
        prev_bar_high = self.data_high[-2]
        prev_bar_open = self.data_open[-2]
        prev_bar_close = self.data_close[-2]

        # Current bar values
        cur_bar_low = self.data_low[-1]
        cur_bar_high = self.data_high[-1]
        cur_bar_open = self.data_open[-1]
        cur_bar_close = self.data_close[-1]

        # Check for Bullish Engulfing Bar
        is_bullish_engulfing = cur_bar_low < prev_bar_low and cur_bar_close > prev_bar_open

        # Check for Bearish Engulfing Bar
        is_bearish_engulfing = cur_bar_high > prev_bar_high and cur_bar_close < prev_bar_open

        # Trade Execution
        # Bullish setup
        if is_bullish_engulfing and not self.position:
            # Define DOL (Draw on Liquidity)
            self.dol = cur_bar_high
            # Define the lower 50% range of the bullish engulfing bar
            bullish_trigger = cur_bar_low + (cur_bar_high - cur_bar_low) * 0.25
            # Check for Power of Three concept
            if prev_bar_close >= bullish_trigger:
                self.buy(size=1)  # Buy 1 unit
                self.sell_stop = cur_bar_low  # Set stop-loss

        # Bearish setup
        elif is_bearish_engulfing and not self.position:
            # Define DOL (Draw on Liquidity)
            self.dol = cur_bar_low
            # Define the upper 50% range of the bearish engulfing bar
            bearish_trigger = cur_bar_high - (cur_bar_high - cur_bar_low) * 0.25
            # Check for Power of Three concept
            if prev_bar_close <= bearish_trigger:
                self.sell(size=1)  # Sell 1 unit
                self.buy_stop = cur_bar_high  # Set stop-loss

        # Check to exit positions
        if self.position:
            if self.position.size > 0:  # Long position
                if cur_bar_high > self.dol:
                    self.close()  # Exit long position
            elif self.position.size < 0:  # Short position
                if cur_bar_low < self.dol:
                    self.close()  # Exit short position

        print(f"Portfolio Value: {self.broker.getvalue()}")

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f"Trade closed: Profit: {trade.pnl}, Net Profit: {trade.pnlcomm}")


def backtesting_strategy(mydata):
    # Activate the backtrader engine
    cerebro = bt.Cerebro()
    print("Activated the backtrader engine (Cerebro)...")

    cerebro.addstrategy(EngulfingBarStrategy)
    cerebro.adddata(mydata)

    cerebro.broker.set_cash(10000)
    # Set the commission to 0.1% (divide by 100 to remove the %)
    cerebro.broker.setcommission(commission=0.001)
    # Add a PercentSizer to use 2% of account balance per trade
    cerebro.addsizer(bt.sizers.PercentSizer, percents=2)

    # Add analyzers to the backtest
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = 'sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name = 'drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name = 'returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name = 'system_quality_number')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = 'trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name = 'transactions')
    print("Added analyzer to the backtrader engine (Cerebro)...")

    # Run the backtest engine
    results = cerebro.run()

    # Extract and print the results
    sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis() # create errors sometimes!
    print(f"Sharpe Ratio: {sharpe_ratio.get('sharperatio', 'N/A')}")
    drawdown = results[0].analyzers.drawdown.get_analysis()
    print(f"Max Drawdown: {drawdown['max']['drawdown']}%")
    returns = results[0].analyzers.returns.get_analysis()
    print(f"Total Returns: {returns['rtot']*100}%")
    system_quality_number = results[0].analyzers.system_quality_number.get_analysis()
    print(f"System Quality Number (SQN): {system_quality_number['system_quality_number']}")
    trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    print(f"Total Trades: {trade_analyzer['total']['total']}")
    print(f"Winning Trades: {trade_analyzer['won']['total']}")
    print(f"Losing Trades: {trade_analyzer['lost']['total']}")
    transactions = results[0].analyzers.transactions.get_analysis()
    print(f"Total Number of Transactions: {len(transactions)}")

    # Plot the results
    print('Plotting the results...')
    cerebro.plot(style='candlestick', volume=False) # Set volume to False to avoid plotting volume data and errors

backtesting_strategy(data_feed)