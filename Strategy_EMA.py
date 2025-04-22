'''
I made $387,620 with this one profitable Day Trading Strategy- [EBP Method]
https://www.youtube.com/watch?v=hCiGC-Yhv4k&start=12
'''

import backtrader as bt
import pandas as pd
from apy import *

# Usage (the data should be a Pandas DataFrame with OHLC data):
data = pd.read_csv(r'C:\Users\loick\VS Code\Forex Historical Data\EURUSD_Tickstory_Daily_5y_cleaned.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
# Replace with your data source


class EmaCrossoverStrategy(bt.Strategy):
    params = (
        ('ema_fast', 10),     # Fast EMA period
        ('ema_mid', 50),      # Mid EMA period
        ('ema_slow', 100),    # Slow EMA period
        ('atr_period', 14),    # ATR period for stop-loss
        ('risk_atr', 1.5),     # ATR multiplier for stop-loss
        ('reward_ratio', 3.0), # Reward-to-risk ratio
        ('order_percentage', 0.01),  # 1% of equity per trade
    )

    def __init__(self):
        # Define EMAs
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_mid = bt.ind.EMA(self.data.close, period=self.p.ema_mid)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)

        # ATR for dynamic stop-loss
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        # Crossovers
        self.cross_fast_mid = bt.ind.CrossOver(self.ema_fast, self.ema_mid)
        self.cross_mid_slow = bt.ind.CrossOver(self.ema_mid, self.ema_slow)

        self.order = None

    def next(self):
        # If an order is pending, skip
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Entry conditions:
            # 1) Fast EMA crosses above Mid EMA
            # 2) Mid EMA above Slow EMA (trend confirmation)
            if self.cross_fast_mid > 0 and self.ema_mid > self.ema_slow:
                # Calculate size: 1% of account equity
                cash = self.broker.get_cash()
                size = (cash * self.p.order_percentage) / self.data.close[0]

                # Define stop-loss and take-profit prices
                stop_loss_price = self.data.close[0] - self.atr[0] * self.p.risk_atr
                take_profit_price = self.data.close[0] + (self.data.close[0] - stop_loss_price) * self.p.reward_ratio

                # Place bracket order
                self.order = self.buy_bracket(
                    size=size,
                    price=self.data.close[0],
                    stopprice=stop_loss_price,
                    limitprice=take_profit_price
                )

            # Short entry: Fast EMA crosses below Mid EMA and Mid below Slow
            elif self.cross_fast_mid < 0 and self.ema_mid < self.ema_slow:
                cash = self.broker.get_cash()
                size = (cash * self.p.order_percentage) / self.data.close[0]

                stop_loss_price = self.data.close[0] + self.atr[0] * self.p.risk_atr
                take_profit_price = self.data.close[0] - (stop_loss_price - self.data.close[0]) * self.p.reward_ratio

                self.order = self.sell_bracket(
                    size=size,
                    price=self.data.close[0],
                    stopprice=stop_loss_price,
                    limitprice=take_profit_price
                )
        else:
            # Position exists: optional trailing or exit on crossover back
            # Exit long when fast EMA crosses below mid EMA
            if self.position.size > 0 and self.cross_fast_mid < 0:
                self.close()
            # Exit short when fast EMA crosses above mid EMA
            elif self.position.size < 0 and self.cross_fast_mid > 0:
                self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def stop(self):
        pnl = round(self.broker.getvalue() - self.broker.startingcash, 2)
        print(f"{self.params.ema_fast}/{self.params.ema_mid}/{self.params.ema_slow} EmaCrossoverStrategy PnL: {pnl}")


def backtest_strategy(data, cash=10000, commission=0.002):
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(EmaCrossoverStrategy)
    cerebro.broker.set_cash(10000)
    # Set the commission to 0.1% (divide by 100 to remove the %)
    cerebro.broker.setcommission(commission=0.001)
    # Add a PercentSizer to use 1% of account balance per trade
    cerebro.addsizer(bt.sizers.PercentSizer, percents=1)

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
    cerebro.plot(style='candlestick', barup='green', bardown='red', barupcolor='green', bardowncolor='red')

if __name__ == '__main__':
    # Example usage - replace path with your data file
    backtest_strategy(data)