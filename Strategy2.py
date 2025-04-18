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
        '''
        Strictly on engulfing patterne, candle 1 is the one that has to be engulfed by candle 2.
        Candle 3 would be the candle that opens after the engulfing pattern, after candle 2 close. 
        '''
        candle2_open = data.open[0]
        candle2_close = data.close[0]
        candle2_high = data.high[0]
        candle2_low = data.low[0]
        candle1_open = data.open[-1]
        candle1_close = data.close[-1]
        candle1_high = data.open[-1]
        candle1_low = data.low[-1]

        first_candle_bullish = candle1_close > candle1_open
        first_candle_bearish = candle1_close < candle1_open
        candle_engulfing_bullish = first_candle_bearish and candle2_close > candle1_open and candle2_low < candle1_low
        candle_engulfing_bearish = first_candle_bullish and candle2_close < candle1_open and candle2_high > candle1_high

        # Trade Execution
        # Bullish setup
        if bullish_engulfing and not self.position:
            # Define DOL (Draw on Liquidity)
            self.dol = cur_bar_high
            # Define the lower 50% range of the bullish engulfing bar
            bullish_trigger = cur_bar_low + (cur_bar_high - cur_bar_low) * 0.25
            # Check for Power of Three concept
            if prev_bar_close >= bullish_trigger:
                self.buy(size=1)  # Buy 1 unit
                self.sell_stop = cur_bar_low  # Set stop-loss

        # Bearish setup
        elif bearish_engulfing and not self.position:
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