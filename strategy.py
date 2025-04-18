# Import relevant libraries and modules
import pandas as pd
import backtrader as btr
#from clean_data import cleaned_csv_file
import matplotlib as plt

'''
# Data gathering with yahoo finance, unlimited lookback for daily or higher timeframe.
# For lower timeframe you need to find other data sources!
import yfinance as yf
data = yf.Ticker('AAPL').history(period='1y')
print(data.head())
'''

cleaned_csv_file = r"C:\Users\loick\Visual Studio Code\My Coding\Quantstrategy\cleaned_USDJPY-2000-2020-15m.csv"

# Check if cleaned_csv_file is set
if cleaned_csv_file is None:
    raise ValueError("cleaned_csv_file is not set. Please check the clean_data.py script.")

# Load the cleaned CSV file into a DataFrame, limiting the number of rows
max_rows = 5000  # Set the maximum number of rows to read
cleaned_df = pd.read_csv(cleaned_csv_file, parse_dates=['datetime'], nrows=max_rows)

# Set the datetime column as the index
cleaned_df.set_index('datetime', inplace=True)

# Convert the cleaned DataFrame to a Backtrader data feed (mandatory for cerebro engine)
data_feed = btr.feeds.PandasData(
    dataname=cleaned_df, 
    datetime=None,  # Use the index as the datetime
    open=0, 
    high=1, 
    low=2, 
    close=3, 
    volume=-1,  # Set volume to -1 to indicate no volume data
    openinterest=-1,
    plot=True  # Enable plotting
)

# Management of the strategy
stop_loss = 10  # Stop loss in pips
risk_to_reward = 2  # Risk to reward ratio


# Define the strategy
class MyStrategy(btr.SignalStrategy):
    params = (
        ('bb_period', 20), # Bollinger Bands
        ('bb_deviation', 2),
        ('adx_period', 14), # Average Directional Index
        ('risk_to_reward', 2.0), # Risk to Reward ratio
        ('stop_loss', 10.0), # Stop loss in pips
        ('pairs', "jpy") # Type of pairs (jpy, usd, xau)
    )

    # Define and initalize the indicators
    def __init__(self): 
        self.bb = btr.ind.BollingerBands(self.data.close, period=self.params.bb_period, devfactor=self.params.bb_deviation)
        self.adx = btr.ind.AverageDirectionalMovementIndex(self.data, period=self.params.adx_period)

        self.breakout_flag = None  # will be set to 'upper' or 'lower'

    def next(self):
        # Price
        current_close = self.data.close[0]
        prev_close = self.data.close[-1]
        
        # Bolllinger Bands conditions
        upper_band = self.bb.lines.top[0]
        lower_band = self.bb.lines.bot[0]
        # Only take trades if ADX > 30 (indicates a strong trend)
        adx_filter = self.adx.adx[0] > 30


        # If no breakout flagged, check if previous candle broke out of the bands
        if self.data.close[-1] > self.bb.lines.top[-1]:
            self.breakout_flag = 'upper'
        elif self.data.close[-1] < self.bb.lines.bot[-1]:
            self.breakout_flag = 'lower'
            
        else:
            # A breakout has occurred, now wait for re-entry inside the bands
            if self.breakout_flag == 'upper':
                # Price must close below the upper band for a reversal confirmation (sell signal)
                if current_close < upper_band and adx_filter:
                    # Calculate stop loss and take profit based on asset pair
                    if self.params.pairs == "yen":
                        stop_loss = self.params.stop_loss * 0.01
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.01
                    elif self.params.pairs == "usd":
                        stop_loss = self.params.stop_loss * 0.0001
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.0001
                    elif self.params.pairs == "xau":
                        stop_loss = self.params.stop_loss * 0.1
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.1
                    else:
                        stop_loss = self.params.stop_loss * 0.0001  # default for USD pairs
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.0001

                    self.sell_bracket(
                        limitprice=current_close - take_profit,
                        stopprice=current_close + stop_loss
                    )
                    self.breakout_flag = None  # reset flag after trade

                # If the candle hasn't re-entered, do nothing.
            
            elif self.breakout_flag == 'lower':
                # Price must close above the lower band for a reversal confirmation (buy signal)
                if current_close > lower_band and adx_filter:
                    if self.params.pairs == "yen":
                        stop_loss = self.params.stop_loss * 0.01
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.01
                    elif self.params.pairs == "usd":
                        stop_loss = self.params.stop_loss * 0.0001
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.0001
                    elif self.params.pairs == "xau":
                        stop_loss = self.params.stop_loss * 0.1
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.1
                    else:
                        stop_loss = self.params.stop_loss * 0.0001  # default for USD pairs
                        take_profit = self.params.stop_loss * self.params.risk_to_reward * 0.0001

                    self.buy_bracket(
                        limitprice=current_close + take_profit,
                        stopprice=current_close - stop_loss
                    )
                    self.breakout_flag = None  # reset flag after trade
                # Else, if still outside, remain in breakout state without trading



# Activate the backtrader engine
cerebro = btr.Cerebro()
print("Activated the backtrader engine (Cerebro)...")
cerebro.addstrategy(MyStrategy)
cerebro.adddata(data_feed)
cerebro.broker.set_cash(10000)
# Set the commission to 0.1% (divide by 100 to remove the %)
cerebro.broker.setcommission(commission=0.001)
# Add a PercentSizer to use 2% of account balance per trade
cerebro.addsizer(btr.sizers.PercentSizer, percents=2)

# Add analyzers to the backtest
cerebro.addanalyzer(btr.analyzers.SharpeRatio, _name = 'sharpe_ratio')
cerebro.addanalyzer(btr.analyzers.DrawDown, _name = 'drawdown')
cerebro.addanalyzer(btr.analyzers.Returns, _name = 'returns')
cerebro.addanalyzer(btr.analyzers.SQN, _name = 'system_quality_number')
cerebro.addanalyzer(btr.analyzers.TradeAnalyzer, _name = 'trade_analyzer')
cerebro.addanalyzer(btr.analyzers.Transactions, _name = 'transactions')
print("Added analyzer to the backtrader engine (Cerebro)...")

# Run the backtest engine
results = cerebro.run()

# Extract and print the results
sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis() # create errors sometimes!
print(f"Sharpe Ratio: {sharpe_ratio['sharperatio']}")
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
cerebro.plot(volume=False)  # Set volume to False to avoid plotting volume data and errors