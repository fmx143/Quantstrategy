import MetaTrader5 as mt5
import talib
import numpy as np
import pandas as pd
import datetime
import math
import time
from apy import *

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)

# Establish connection to the MetaTrader 5 terminal
if not mt5.initialize(login=ic_login , password=ic_pwd , server=ic_server):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# display data on MetaTrader 5 version and account informations
mt5_version=mt5.version()
mt5_account=mt5.account_info()
print("MetaTrader 5 version: ",mt5_version)
print("Account number: ",mt5_account.login)
print("Account balance: ",mt5_account.balance)
print("Account equity: ",mt5_account.equity)


'''Asset to trade'''
symbol="EURUSD"
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(symbol, "not found, can not call order_check()")
    mt5.shutdown()
    quit()
point = mt5.symbol_info(symbol).point  # Get the point size for the symbol
ask_price = mt5.symbol_info_tick(symbol).ask  # Get the current ask price
bid_price = mt5.symbol_info_tick(symbol).bid  # Get the current bid price
print("Ask price: ", ask_price)
print("Point size: ", point)

'''Modify or add all parameters here for the script'''
number_of_data=1000
timeframe=mt5.TIMEFRAME_M1
lots=0.01
deviation=10 # Time you can wait for the order to be executed(pips)
stop_loss=0.0010 # Stop loss in pips for USD
take_profit=.0020 # Take profit in pips for USD
fixed_buy_sl = ask_price - stop_loss # Stop loss price
fixed_buy_tp = ask_price + take_profit  # Take profit price
fixed_sell_sl = bid_price + stop_loss # Stop loss price
fixed_sell_tp = bid_price - take_profit  # Take profit price

ema_type="EMA" # Type of moving average to use for the strategy
ema_fast=8 # Fast moving average period
ema_slow=21 # Slow moving average period
ema_200=200 # 200 period moving average for trend direction


# Get the rates from the broker of the mt5 account
def get_rates(symbol,number_of_data,timeframe):
    # Extract data from MetaTrader 5
    from_date=datetime.datetime.now()
    rates=mt5.copy_rates_from(symbol,timeframe,from_date,number_of_data) # tuple format, not really readable
    if rates is None:
        print("No data received, error code =", mt5.last_error())
        quit()
    # Convert to pandas DataFrame
    df_rates=pd.DataFrame(rates)
    df_rates['time']=pd.to_datetime(df_rates['time'], unit='s')
    df_rates['time']=pd.to_datetime(df_rates['time'], format='%Y-%m-%d %H:%M:%S')
    df_rates.set_index('time')
    return df_rates
#print(get_rates(symbol,number_of_data,timeframe).head())


'''Buy position'''
#def buy(symbol,lots,deviation,fixed_stop_loss,fixed_take_profit):
    # Create dictionary for the request
request_buy = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lots,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick(symbol).ask,
    "sl": fixed_buy_sl,
    "tp": fixed_buy_tp,
    "deviation": deviation,
    "magic": 1001,
    "comment": "python script",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
#buy_order=mt5.order_send(request_buy)


'''Sell position'''
request_sell = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lots,
    "type": mt5.ORDER_TYPE_SELL,
    "price": mt5.symbol_info_tick(symbol).bid,
    "sl": fixed_sell_sl,
    "tp": fixed_sell_tp,
    "deviation": deviation,
    "magic": 1002,
    "comment": "python script",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
#sell_order=mt5.order_send(request_sell)

def strategy(symbol):
    global df
    df=get_rates(symbol,number_of_data,timeframe)
    