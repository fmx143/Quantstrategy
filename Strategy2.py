'''
The most profitable Daily Bias Strategy [EBP- 88% Proven Win Rate]
https://www.youtube.com/watch?v=XiU9JHUM5NE
'''

import backtrader as bt

class EngulfingBarStrategy(bt.Strategy):
    params = (
        ('cash', 10000),
        ('commission', 0.002),  # 0.2% commission
    )

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


# Backtest the strategy
if __name__ == '__main__':
    # Create a Backtrader cerebro engine
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(EngulfingBarStrategy)

    # Load the data
    data = bt.feeds.PandasData(dataname=bt.test.EURUSD)  # Replace with your own data if needed
    cerebro.adddata(data)

    # Set the initial cash
    cerebro.broker.setcash(10000)

    # Set the commission
    cerebro.broker.setcommission(commission=0.002)

    # Run the backtest
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the results
    cerebro.plot()