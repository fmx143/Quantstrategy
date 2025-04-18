'''
I made $387,620 with this one profitable Day Trading Strategy- [EBP Method]
https://www.youtube.com/watch?v=hCiGC-Yhv4k&start=12
'''

import backtrader as bt
import pandas as pd
from apy import *

# Usage (the data should be a Pandas DataFrame with OHLC data):
data = pd.read_csv(eu_daily_clean)  # Replace with your data source


class EngulfingCandleStrategy(bt.Strategy):
    def __init__(self):
        self.order_block_price = None  # Placeholder for support/resistance price

    def engulfing(self, data):

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

        if candle_engulfing_bullish:
            return 'bullish'
        elif candle_engulfing_bearish:
            return 'bearish'
        else:
            return None

    def next(self):
        current_low = self.data.low[0]
        current_high = self.data.high[0]

        # Check engulfing pattern
        engulfing = self.engulfing(self.data)

        if engulfing == 'bullish' and self.position.size < 0:  # If short, close and go long
            self.close()
            self.buy(size=1, exectype=bt.Order.Market, price=current_low)
        elif engulfing == 'bearish' and self.position.size > 0:  # If long, close and go short
            self.close()
            self.sell(size=1, exectype=bt.Order.Market, price=current_high)

    def stop(self):
        # Implement any additional stop logic here if needed
        pass


def backtest_strategy(data, cash=10000, commission=0.002):
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(EngulfingCandleStrategy)
    cerebro.broker.set_cash(10000)
    # Set the commission to 0.1% (divide by 100 to remove the %)
    cerebro.broker.setcommission(commission=0.001)
    # Add a PercentSizer to use 2% of account balance per trade
    cerebro.addsizer(bt.sizers.PercentSizer, percents=2)

    # Add the data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set the initial cash
    cerebro.broker.setcash(cash)

    # Set the commission
    cerebro.broker.setcommission(commission=commission)

    # Run the backtest
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the results
    cerebro.plot()

backtest_strategy(data)