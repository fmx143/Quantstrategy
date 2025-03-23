# Import relevant libraries and modules
import pandas as pd
import backtrader as btr
from clean_data import *
import matplotlib as plt

import quantstats as qs
print(qs.__version__)


# Initialize QuantStats
qs.extend_pandas()

'''
# Data gathering with yahoo finance, unlimited lookback for daily or higher timeframe.
# For lower timeframe you need to find other data sources!
import yfinance as yf
data = yf.Ticker('AAPL').history(period='1y')
print(data.head())
'''

# Convert the cleaned_df DataFrame to a Backtrader data feed (mandatory for cerebro engine)
data_feed = btr.feeds.PandasData(
    dataname=cleaned_csv_file, 
    datetime=0,     
    open=1, 
    high=2, 
    low=3, 
    close=4, 
    volume=5, 
    openinterest=-1
    )

# Define the strategy
class MyStrategy(btr.SignalStrategy):
    params = (
        ('ema1_period', 5), # EMA
        ('ema2_period', 15),
        ('ema3_period', 30),
        ('ema4_period', 50),
        ('bb_period', 20), # Bollinger Bands
        ('bb_deviation', 2),
        ('adx_period', 14), # Average Directional Index
        ('risk_to_reward', 2.0), # Risk to Reward ratio
        ('stop_loss', 10.0), # Stop loss in pips --> Need to adapt depending on the asset!!!!
    )

    # Define and initalize the indicators
    def __init__(self): 
        self.ema1 = btr.ind.EMA(self.data.close, period=self.params.ema1_period)
        self.ema2 = btr.ind.EMA(self.data.close, period=self.params.ema2_period)
        self.ema3 = btr.ind.EMA(self.data.close, period=self.params.ema3_period)
        self.ema4 = btr.ind.EMA(self.data.close, period=self.params.ema4_period)
        self.bb = btr.ind.BollingerBands(self.data.close, period=self.params.bb_period, devfactor=self.params.bb_deviation)
        self.adx = btr.ind.AverageDirectionalMovementIndex(self.data, period=self.params.adx_period)

        self.portfolio_values = []  # List to store portfolio values

    def next(self):
        # 1. condition
        # check if price has closed below the lower Bollinger Band or high above the upper Bollinger Band
        bb_overbought = self.data.close[0] > self.bb.lines.top[0]
        bb_oversold = self.data.close[0] < self.bb.lines.bot[0]
        
        # 2. condition
        # A bearish signal occurs when the 5 EMA is higher than all the other
        # A bullish signal occurs when the 5 EMA is lower than all the other
        ema_sell = self.ema1 > self.ema2 and self.ema1 > self.ema3 and self.ema1 > self.ema4
        ema_buy = self.ema1 < self.ema2 and self.ema1 < self.ema3 and self.ema1 < self.ema4

        # 3. condition
        # Only take trades if ADX > 30 (indicates a strong trend)
        adx_filter = self.adx.adx[0] > 30

        # Calculate stop loss and take profit levels
        entry_price = self.data.close[0]
        stop_loss = self.params.stop_loss * 0.0001 # Adaptation needed depending on the asset!!!
        take_profit = stop_loss * self.params.risk_to_reward

        # Buy signal: EMA, Bollinger Bands, Awesome Oscillator and ADX conditions are met
        if bb_overbought and ema_sell and adx_filter:
            self.sell_bracket(
                limitprice= entry_price - take_profit,
                stopprice= entry_price + stop_loss
            )
        elif bb_oversold and ema_buy and adx_filter:
            self.buy_bracket(
                limitprice= entry_price + take_profit,
                stopprice= entry_price - stop_loss
            )
        else:
            pass

    def notify_order(self, order):
        # Record portfolio value after every completed order
        if order.status in [order.Completed]:
            self.portfolio_values.append(self.broker.getvalue())


# Activate the backtrader engine
cerebro = btr.Cerebro()
cerebro.addstrategy(MyStrategy)
cerebro.adddata(data_feed)
cerebro.broker.set_cash(10000)
# Set the commission to 0.1% (divide by 100 to remove the %)
cerebro.broker.setcommission(commission=0.001)
# Add a PercentSizer to use 2% of account balance per trade
cerebro.addsizer(btr.sizers.PercentSizer, percents=2)

# Add analyzers to the backtest
#cerebro.addanalyzer(btr.analyzers.SharpeRatio, _name = 'sharpe_ratio')
cerebro.addanalyzer(btr.analyzers.DrawDown, _name = 'drawdown')
cerebro.addanalyzer(btr.analyzers.Returns, _name = 'returns')
cerebro.addanalyzer(btr.analyzers.SQN, _name = 'system_quality_number')
cerebro.addanalyzer(btr.analyzers.TradeAnalyzer, _name = 'trade_analyzer')
cerebro.addanalyzer(btr.analyzers.Transactions, _name = 'transactions')

# Run the backtest engine
results = cerebro.run()
strategy = results[0]

# Convert Portfolio Values to a Pandas Series with a Valid Datetime Index
portfolio_series = pd.Series(strategy.portfolio_values, index=pd.to_datetime(cleaned_df.index[-len(strategy.portfolio_values):]))
# Ensure the index is properly formatted as a DatetimeIndex
portfolio_series.index = pd.to_datetime(portfolio_series.index)

# Convert Portfolio Values to Returns
returns_series = portfolio_series.pct_change().dropna()

# Verify Index Type (For Debugging)
print(type(returns_series.index))  # Should print: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

# Use QuantStats for Performance Analysis
qs.reports.html(returns_series, output='quantstats-report.html', title="Backtest Performance Report")

# Display Key Performance Metrics
print(qs.stats.sharpe(returns_series))
print(qs.stats.drawdown(returns_series))
print(qs.stats.volatility(returns_series))

# Visualizations
qs.plots.returns(returns_series)
qs.plots.drawdowns(returns_series)
qs.plots.monthly_heatmap(returns_series)


'''
# Extract and print the results
#sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis() # create errors sometimes!
#print(f"Sharpe Ratio: {sharpe_ratio['sharperatio']}")
drawdown = results[0].analyzers.drawdown.get_analysis()
print(f"Max Drawdown: {drawdown['max']['drawdown']}%")
returns = results[0].analyzers.returns.get_analysis()
print(f"Total Returns: {returns['rtot']*100}%")
system_quality_number = results[0].analyzers.system_quality_number.get_analysis()
print(f"System Quality Number (SQN): {system_quality_number['system_quality_number']}")
trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
print(f"Total Number of Trades: {trade_analyzer['total']['total']}")
transactions = results[0].analyzers.transactions.get_analysis()
print(f"Total Number of Transactions: {len(transactions)}")

# Plot the results
print('Plotting the results...')
cerebro.plot()
'''