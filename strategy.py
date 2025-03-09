import pandas as pd
import backtesting as bt
import backtrader as btr
import yfinance as yf
import datetime
from indicators import *

# data gatheringwith yahoo finance
# data = yf.Ticker('AAPL').history(period='1y')
# print(data.head())

# Function to check for required columns in a CSV file
def check_required_columns(data):
    required_columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    columns_present = all(column in data.columns for column in required_columns)
    return columns_present

# Function to clean the CSV data
def clean_csv_data(file_path, cleaned_file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    if check_required_columns(df):
        # Separate the combined datetime column into separate date and time columns
        df[['Date', 'Time']] = df['DateTime'].str.split(' ', expand=True)
        df.drop(columns=['DateTime'], inplace=True)
       # Reorder the columns to have Date and Time in the first place
        df = df[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        # Save the cleaned data to a new CSV file
        df.to_csv(cleaned_file_path, index=False)
        print("Cleaned data saved to:", cleaned_file_path)
        return cleaned_file_path
    else:
        print("The CSV file does not contain all required columns.")
        return None

# Path to the original CSV file
data_csv_path = r"C:\Users\loick\Downloads\Forex Historical Data\EURUSD\ForexSB_EURUSD_M15_06.02.2017-14.02.2025.csv"
# Path to the cleaned CSV file
cleaned_csv_path = r"C:\Users\loick\Downloads\Forex Historical Data\EURUSD\ForexSB_EURUSD_M15_06.02.2017-14.02.2025_cleaned.csv"

# Clean the CSV data and print the head of the cleaned data
cleaned_file_path = clean_csv_data(data_csv_path, cleaned_csv_path)
print(pd.read_csv(cleaned_file_path).head())


# Backtrader function to read the csv file
data_feed = btr.feeds.GenericCSVData(
    dataname=cleaned_csv_path, 
    dtformat=('%Y-%m-%d'),  # datetime format
    tmformat=('%H:%M'),  # time format
    datetime=0,
    time=1, 
    open=2, 
    high=3, 
    low=4, 
    close=5, 
    volume=6    , 
    openinterest=-1
)

print("Running the strategy...")

# Define your strategy
class SmaCross(btr.SignalStrategy):
    params = (
        ('sma1_period', 8),
        ('sma2_period', 21),
        ('risk_to_reward', 2.0),
        ('stop_loss', 10.0),
    )

    def __init__(self): 
        # Use the custom SMA indicator with a tunable period, from indicators.py file
        self.sma_fast = btr.ind.SMA(self.data.close, period=self.params.sma1_period)
        self.sma_slow = btr.ind.SMA(self.data.close, period=self.params.sma2_period)
        # Define a crossover indicator between price and custom SMA
        self.crossover = btr.ind.CrossOver(self.sma_fast, self.sma_slow) 


    def next(self):
        # Check for crossover signals and place orders accordingly
        if self.crossover > 0 and self.data.close[0] > self.sma_fast[0] and self.data.close[0] > self.sma_slow[0]:  # Golden cross (buy signal)
            self.buy()
            # Calculate stop loss and take profit for long position
            entry_price = self.data.close[0]
            stop_loss = entry_price - self.params.stop_loss * 0.0001
            take_profit = entry_price + self.params.stop_loss * self.params.risk_to_reward * 0.0001
            # Place stop loss and take profit orders
            self.buy_bracket(limitprice=take_profit, stopprice=stop_loss)

        elif self.crossover < 0 and self.data.close[0] < self.sma_fast[0] and self.data.close[0] < self.sma_slow[0]:  # Death cross (sell signal)
            self.sell()
            # Calculate stop loss and take profit for short position
            entry_price = self.data.close[0]
            stop_loss = entry_price + self.params.stop_loss * 0.0001
            take_profit = entry_price - self.params.stop_loss * self.params.risk_to_reward * 0.0001
            # Place stop loss and take profit orders
            self.sell_bracket(limitprice=take_profit, stopprice=stop_loss)

# Activate the backtrader engine
cerebro = btr.Cerebro()
# Add the strategy to the engine
cerebro.addstrategy(SmaCross)  
# Add the data feed to the engine
cerebro.adddata(data_feed)
# Set the initial cash amount for the backtest
cerebro.broker.set_cash(1000)
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

# Run the backtest
results = cerebro.run()

# Extract and print the results
#sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis()
drawdown = results[0].analyzers.drawdown.get_analysis()
returns = results[0].analyzers.returns.get_analysis()
system_quality_number = results[0].analyzers.system_quality_number.get_analysis()
trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
transactions = results[0].analyzers.transactions.get_analysis()

#print(f"Sharpe Ratio: {sharpe_ratio['sharperatio']}")
print(f"Max Drawdown: {drawdown['max']['drawdown']}%")
print(f"Total Returns: {returns['rtot']*100}%")
print(f"System Quality Number (SQN): {system_quality_number['system_quality_number']}")
print(f"Total Number of Trades: {trade_analyzer['total']['total']}")
print(f"Total Number of Transactions: {len(transactions)}")

# Plot the results
cerebro.plot()
