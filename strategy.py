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

# Define your strategy
class SmaCross(btr.SignalStrategy):
    def __init__(self): 
        # Use the custom SMA indicator with a tunable period, from indicators.py file
        custom_sma = CustomSMA(self.data, period=20)  # You can change the period here
        # Define a crossover indicator between price and custom SMA
        crossover = btr.ind.CrossOver(self.data.close, custom_sma)

          # Add a LONG signal when the price crosses above the SMA
        self.signal_add(btr.SIGNAL_LONG, crossover > 0)
        # Add a SHORT signal when the price crosses below the SMA
        self.signal_add(btr.SIGNAL_SHORT, crossover < 0)

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
# Run the backtest
cerebro.run()
# Plot the results
cerebro.plot()
