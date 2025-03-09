import backtrader as bt

# Custom Simple Moving Average (SMA) indicator
class CustomSMA(bt.Indicator):
    lines = ('sma',)
    params = (('period', 20),)

    # Initialize the indicator
    def __init__(self):
        self.addminperiod(self.params.period)
    # Define the logic for calculating the SMA. The next method is called for each new data point. 
    # It calculates the SMA by summing the last period data points and dividing by the period.
    def next(self):
        self.lines.sma[0] = sum(self.data.get(size=self.params.period)) / self.params.period

