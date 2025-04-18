#!/usr/bin/env python3

"""
This script is a conversion of the MQL4 "Scalper-GG-14" expert advisor into Python for MetaTrader 5.
It uses the MetaTrader5 package to interface with MT5 and TA-Lib (plus numpy/pandas) for technical indicators.

Before running, install the required packages, for example:
    pip install MetaTrader5 numpy pandas TA-Lib

Also, make sure that MetaTrader5 is installed and running with the Python API enabled.
"""

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



''' ***** Global Parameters (converted from extern/input parameters)
These parameters define the behavior of the trading strategy and can be adjusted as needed.'''
mn = "Scalper-GG-14"  # Name of the expert advisor
lot = None  # Lot size (calculated dynamically based on account equity)
shifthighrecent = 5  # Shift for recent high calculation
shiftlowrecent = 2  # Shift for recent low calculation
shifthigh = 5  # Shift for longer-period high calculation
shiftlow = 7  # Shift for longer-period low calculation
trail_enabled = True  # Enable/disable trailing stop functionality
TrailingStop = 2  # Trailing stop distance in points
barnumber = 24  # Number of bars for longer-period high/low calculation
barnumberrecent = 33  # Number of bars for recent high/low calculation
MagicNumber = 14072020  # Unique identifier for orders placed by this EA
atrMultiple = 3  # Multiplier for ATR-based stop loss
TrailingStart = 21  # Minimum price movement (in points) before trailing stop activates
volume1 = 1  # Volume index for recent bar
volume0 = 0  # Volume index for older bar
adxperiod = 31  # Period for ADX indicator
adxthreshold = 23  # Threshold for ADX indicator to confirm trend strength
rsiperiod = 21  # Period for RSI indicator
rsilower = 48  # Lower threshold for RSI (oversold level)
rsiupper = 80  # Upper threshold for RSI (overbought level)
Slippage = 3  # Maximum allowed slippage in points
Indicatorperiod = 14  # Period for moving average and other indicators
MaxSpreadAllow = 10  # Maximum allowed spread (in points) for placing orders
LotFactor = 60  # Factor for calculating lot size based on account equity
ecartask = 2  # Offset (in points) for placing buy stop orders
ecartbid = 10  # Offset (in points) for placing sell stop orders
Start_Time = 0  # Trading start hour (0 disables time filtering)
Finish_Time = 0  # Trading finish hour (0 disables time filtering)
bidbid = 600  # Offset for sell order conditions
askask = 300  # Offset for buy order conditions
AddPriceGap = 1  # Additional price gap for stop loss/take profit
MinPipsToModify = 1  # Minimum pips to modify stop loss/take profit

# ***** Connection and symbol initialization *****
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

# get all symbols
symbols=mt5.symbols_get()
print('Symbols: ', len(symbols))
count=0
# display the first five ones
for s in symbols:
    count+=1
    print("{}. {}".format(count,s.name))
    if count==5: break
print()

symbol = "EURUSD"  # Trading symbol (e.g., EURUSD). Change as necessary.

# Make sure that the symbol is available
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"Symbol {symbol} not found, please check symbol name")
    mt5.shutdown()
    quit()
else:
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed")
            mt5.shutdown()
            quit()
    print(f"Symbol {symbol} is available and ready for trading.")


# shut down connection to the MetaTrader 5 terminal
# mt5.shutdown()


''' ***** Trading Strategy Overview *****
# The strategy uses technical indicators (ADX, RSI, ATR, and moving averages) to identify trading opportunities.
# It places pending buy or sell stop orders based on price levels calculated from recent highs/lows and moving averages.
# The strategy includes:
# - Dynamic lot size calculation based on account equity.
# - ATR-based stop loss and take profit levels.
# - Trailing stop functionality to lock in profits as the trade moves in favor.
# - Time filtering to restrict trading to specific hours (if enabled).
# - Spread filtering to avoid trading during high-spread conditions.
# - Booster logic to confirm trade signals using volume and indicator thresholds.

# The rest of the code implements helper functions, indicator calculations, and the main trading logic. '''

symbol_info = mt5.symbol_info(symbol)
print(f"📄Symbol info for {symbol}: {symbol_info}")
symbol_price = mt5.symbol_info_tick(symbol)
print(f"💸Symbol price for {symbol}: {symbol_price}")



# ***** Helper functions for indicator calculations *****
def get_rates(timeframe=mt5.TIMEFRAME_M1, count=500):
    """
    Retrieves historical price data as a DataFrame.
    """
    rates = mt5.copy_rates_from(symbol, timeframe, datetime.datetime.now(), count)
    if rates is None or len(rates) == 0:
        print("Failed to get rates data")
        return None
    
    # Convert the rates data into a Pandas DataFrame for easier manipulatio
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def iADX(df, period, price='close'):
    # talib.ADX requires high, low and close arrays
    if len(df) < period + 1:
        print(f"Not enough data points for ADX calculation. Need at least {period + 1}, got {len(df)}")
        return np.zeros(len(df)) # Return an array of zeros if not enough data
    
    try:
        return talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        return np.zeros(len(df)) # Return an array of zeros in case of an error


def iRSI(df, period, price='close'):
    if len(df) < period + 1:
        print(f"Not enough data points for RSI calculation. Need at least {period + 1}, got {len(df)}")
        return np.zeros(len(df))
    
    try:
        return talib.RSI(df[price].values, timeperiod=period)
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return np.zeros(len(df))


def iMA(df, period, matype=0, price='close'):
    # Using TA-Lib's MA. matype: 0 = SMA, 1 = EMA, 2 = WMA, etc.
    if len(df) < period:
        print(f"Not enough data points for MA calculation. Need at least {period}, got {len(df)}")
        return np.zeros(len(df))
    
    try:
        # Calculate the moving average using the specified type and price
        return talib.MA(df[price].values, timeperiod=period, matype=matype)
    except Exception as e:
        print(f"Error calculating MA: {e}")
        return np.zeros(len(df))


def iATR(df, period):
    if len(df) < period + 1:
        print(f"Not enough data points for ATR calculation. Need at least {period + 1}, got {len(df)}")
        return np.zeros(len(df))
    
    try:
        # Calculate ATR using high, low, and close prices
        return talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return np.zeros(len(df))


def iHigh(df, shift):
    # Highest high at the given shift (0 is the current bar)
    if len(df) <= shift:
        print(f"Not enough data points for High at shift {shift}. Need at least {shift + 1}, got {len(df)}")
        return None
    
    try:
        return df['high'].iloc[-1-shift]
    except Exception as e:
        print(f"Error getting High: {e}")
        return None


def iLow(df, shift):
    if len(df) <= shift:
        print(f"Not enough data points for Low at shift {shift}. Need at least {shift + 1}, got {len(df)}")
        return None
    
    try:
        return df['low'].iloc[-1-shift]
    except Exception as e:
        print(f"Error getting Low: {e}")
        return None


def iHighest(df, lookback, shift):
    # Get the index of the highest high from the last "lookback" bars ending at shift
    if len(df) < lookback + shift:
        print(f"Not enough data points for Highest calculation. Need at least {lookback + shift}, got {len(df)}")
        return None
    
    try:
        end_idx = len(df) - shift if shift > 0 else len(df)
        start_idx = end_idx - lookback
        if start_idx < 0:
            start_idx = 0
        
        segment = df['high'].iloc[start_idx:end_idx]
        return segment.idxmax()
    except Exception as e:
        print(f"Error calculating Highest: {e}")
        return None


def iLowest(df, lookback, shift):
    if len(df) < lookback + shift:
        print(f"Not enough data points for Lowest calculation. Need at least {lookback + shift}, got {len(df)}")
        return None
    
    try:
        end_idx = len(df) - shift if shift > 0 else len(df)
        start_idx = end_idx - lookback
        if start_idx < 0:
            start_idx = 0
        
        segment = df['low'].iloc[start_idx:end_idx]
        return segment.idxmin()
    except Exception as e:
        print(f"Error calculating Lowest: {e}")
        return None


# ***** Order and trade management helper functions *****
def TotalOrdersCount():
    """Return the number of orders with our MagicNumber that are open."""
    try:
        orders = mt5.orders_get(symbol=symbol)
        positions = mt5.positions_get(symbol=symbol)
        
        count = 0
        
        if orders is not None:
            for order in orders:
                if order.magic == MagicNumber:
                    count += 1
        
        if positions is not None:
            for position in positions:
                if position.magic == MagicNumber:
                    count += 1
                    
        return count
    except Exception as e:
        print(f"Error counting orders: {e}")
        return 0


def Lot_Volume():
    """Calculate lot volume based on account equity and the LotFactor parameter."""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return 0.01  # Default minimal lot
        
        global lot
        lot = account_info.equity * 0.01 / LotFactor
        return lot
    except Exception as e:
        print(f"Error calculating lot volume: {e}")
        return 0.01  # Default minimal lot


def NR(thelot):
    """Normalizes lot size based on symbol parameters."""
    try:
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            print("Failed to get symbol info")
            return 0.01  # Default minimal lot
            
        maxlots = sym_info.volume_max
        minilot = sym_info.volume_min
        lstep = sym_info.volume_step
        
        # Normalize the lot size to the symbol's step
        lots = lstep * round(thelot / lstep)
        return max(min(lots, maxlots), minilot)
    except Exception as e:
        print(f"Error normalizing lot size: {e}")
        return 0.01  # Default minimal lot


def new_del():
    """Delete pending orders (BUY_STOP and SELL_STOP) for our symbol."""
    try:
        orders = mt5.orders_get(symbol=symbol)
        if orders is None:
            print("No orders to delete or error getting orders")
            return 0
            
        for order in orders:
            if order.magic == MagicNumber and order.type in [mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP]:
                # Delete order (cancellation)
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "symbol": symbol,
                    "magic": MagicNumber,
                }
                result = mt5.order_send(request)
                print(f"Deleting pending order {order.ticket}, type: {order.type} result: {result}")
                
        return 0
    except Exception as e:
        print(f"Error deleting orders: {e}")
        return 0


def trail():
    """Trailing stop logic – adjust stop loss after breakeven."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return
            
        # Calculate minimal pips modification factor (in points)
        MPTM = MinPipsToModify * 10 * symbol_info.point
        
        tick = mt5.symbol_info(symbol)
        if tick is None:
            print("Failed to get current price tick")
            return
            
        for position in positions:
            if position.magic != MagicNumber:
                continue
                
            if position.type == mt5.POSITION_TYPE_BUY:
                # For a buy position trailing stop: if current bid - open price > threshold, then modify
                current_bid = tick.bid
                
                # Only trail if price moved enough in our favor (TrailingStart points)
                if current_bid - position.price_open < TrailingStart * 10 * symbol_info.point:
                    continue
                    
                if current_bid - position.price_open > TrailingStop * 10 * symbol_info.point + MPTM:
                    # Calculate new stop loss
                    new_sl = current_bid - TrailingStop * 10 * symbol_info.point
                    
                    # Only modify if the new SL is better than the current one
                    if position.sl == 0 or new_sl > position.sl:
                        # Prepare order modification request
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "sl": new_sl,
                            "tp": position.tp,
                            "symbol": symbol,
                        }
                        result = mt5.order_send(request)
                        print(f"Trailing adjustment (Buy), ticket: {position.ticket}, new SL: {new_sl}, result: {result}")
                        
            elif position.type == mt5.POSITION_TYPE_SELL:
                current_ask = tick.ask
                
                # Only trail if price moved enough in our favor
                if position.price_open - current_ask < TrailingStart * 10 * symbol_info.point:
                    continue
                    
                if position.price_open - current_ask > TrailingStop * 10 * symbol_info.point + MPTM:
                    new_sl = current_ask + TrailingStop * 10 * symbol_info.point
                    
                    # Only modify if the new SL is better than the current one
                    if position.sl == 0 or new_sl < position.sl:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "sl": new_sl,
                            "tp": position.tp,
                            "symbol": symbol,
                        }
                        result = mt5.order_send(request)
                        print(f"Trailing adjustment (Sell), ticket: {position.ticket}, new SL: {new_sl}, result: {result}")
    except Exception as e:
        print(f"Error in trailing stop function: {e}")


# ***** Main trading function (analogous to start() in MQL4) *****
def start():
    try:
        # Check trading times (if Start_Time and Finish_Time are set to nonzero values)
        current_hour = datetime.datetime.now().hour
        if (Start_Time != 0 and current_hour < Start_Time) or (Finish_Time != 0 and current_hour > Finish_Time):
            return

        # Adjust point for 3/5 digit brokers:
        MyPoint = symbol_info.point
        if symbol_info.digits in [3, 5]:
            MyPoint *= 10

        # Get the current spread, ask, bid
        tick = mt5.symbol_info(symbol)
        if tick is None:
            print("Failed to get current price tick")
            return
            
        ask = tick.ask
        bid = tick.bid
        Spread = symbol_info.spread  # Note: spread may be provided in points already

        if Spread <= 0 or Spread > MaxSpreadAllow:
            return

        # Check orders (only if no orders are open)
        if TotalOrdersCount() != 0:
            return

        # Check account margin condition
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return
            
        # Sample condition: AccountFreeMargin > (1000*(AccountEquity()*0.01 / LotFactor))
        if account_info.margin_free <= (1000 * (account_info.equity * 0.01 / LotFactor)):
            return

        # Get market history
        df = get_rates(timeframe=mt5.TIMEFRAME_M1, count=500)
        if df is None or len(df) < max(volume0, volume1, adxperiod, rsiperiod, barnumber, barnumberrecent):
            print("Not enough historical data")
            return

        # Calculate booster condition using volume and indicators.
        booster = False
        
        # Check array lengths before using indexed access
        adx_values = iADX(df, adxperiod)
        rsi_values = iRSI(df, rsiperiod)
        
        # Make sure both arrays have values and are long enough
        if (len(df) > max(volume0, volume1) and 
            len(adx_values) > 0 and 
            len(rsi_values) > 0):
            
            if (df['tick_volume'].iloc[-volume1-1] > df['tick_volume'].iloc[-volume0-1] and
                adx_values[-1] > adxthreshold and
                rsilower < rsi_values[-1] < rsiupper):
                booster = True

        # Check for a new bar – this simplistic implementation assumes the last bar time changes
        # (In production use, keep a global variable for last bar time)
        newBar = True   # For simplicity, we assume every execution corresponds to a new bar

        # Calculate dynamic stoploss / takeprofit values based on EMA and ATR
        ema_period = 300
        if len(df) >= ema_period:
            ema_array = talib.EMA(df['close'].values, timeperiod=ema_period)
            ema_val = ema_array[-1]
        else:
            # Not enough data, use a simpler method
            ema_val = df['close'].mean()
            print(f"Warning: Not enough data for EMA({ema_period}), using mean price instead")
            
        atr_array = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=3)
        if len(atr_array) > 0 and not np.isnan(atr_array[-1]):
            atr_val = atr_array[-1]
        else:
            # Not enough data or calculation failed, use a default value
            atr_val = (df['high'].max() - df['low'].min()) / len(df)
            print("Warning: ATR calculation failed, using simple range/bars instead")
            
        StopLoss = (atr_val * atrMultiple) / MyPoint
        TakeProfit = StopLoss

        # --- Calculation for Keltner Channel parts and price levels ---
        # Calculate recent high/low using "barnumberrecent" and shift values:
        idx_high_recent = iHighest(df, barnumberrecent, shifthighrecent)
        idx_low_recent = iLowest(df, barnumberrecent, shiftlowrecent)
        
        if idx_high_recent is None or idx_low_recent is None:
            print("Failed to calculate high/low indices")
            return
            
        HighOfLastBarsrecent = df.loc[idx_high_recent, 'high']
        LowOfLastBarsrecent = df.loc[idx_low_recent, 'low']

        # Calculate high/low for longer period:
        idx_high = iHighest(df, barnumber, shifthigh)
        idx_low = iLowest(df, barnumber, shiftlow)
        
        if idx_high is None or idx_low is None:
            print("Failed to calculate high/low indices for longer period")
            return
            
        HighOfLastBars = df.loc[idx_high, 'high']
        LowOfLastBars = df.loc[idx_low, 'low']
        AverageBars = (HighOfLastBars + LowOfLastBars) / 2.0
        PartialLow = (LowOfLastBars + AverageBars) / 2.0
        PartialHigh = (HighOfLastBars + AverageBars) / 2.0

        # Calculate MA channel using Weighted Moving Average (WMA) approximation
        if len(df) >= Indicatorperiod:
            iMaLow_array = talib.WMA(df['low'].values, timeperiod=Indicatorperiod)
            iMaHigh_array = talib.WMA(df['high'].values, timeperiod=Indicatorperiod)
            
            if len(iMaLow_array) > 0 and len(iMaHigh_array) > 0:
                iMaLow = iMaLow_array[-1]
                iMaHigh = iMaHigh_array[-1]
                iMaAverage = (iMaHigh + iMaLow) / 2.0
                PartialiMalow = (iMaLow + iMaAverage) / 2.0
                PartialiMahigh = (iMaHigh + iMaAverage) / 2.0
                
                UpperPart = (PartialiMahigh + PartialHigh) / 2.0
                LowerPart = (PartialiMalow + PartialLow) / 2.0
            else:
                print("WMA calculation failed")
                return
        else:
            print(f"Not enough data for WMA({Indicatorperiod})")
            return

        # ----- Open Buy Order Condition -----
        if (bid >= PartialiMahigh and bid >= PartialHigh and booster and
            bid >= HighOfLastBars and HighOfLastBarsrecent >= HighOfLastBars):
            
            # Additional check: Ask + askask * point < ema_val
            if ask + askask * MyPoint < ema_val:
                # Prepare a BUY STOP order at ask + ecartask*point
                price = ask + ecartask * MyPoint
                sl_price = PartialLow - StopLoss * MyPoint - AddPriceGap * MyPoint
                tp_price = PartialHigh + TakeProfit * MyPoint + AddPriceGap * MyPoint
                volume = NR(Lot_Volume())
                
                request = {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY_STOP,
                    "price": price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": Slippage,
                    "magic": MagicNumber,
                    "comment": mn,  # Use the same comment as defined in global parameters
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                result = mt5.order_send(request)
                print(f"Buy pending order result: {result}")
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Order failed with error code: {result.retcode}")
                    
                return

        # ----- Open Sell Order Condition -----
        if (PartialiMalow <= bid and PartialLow <= bid and booster and
            LowOfLastBars <= bid and LowOfLastBarsrecent <= LowOfLastBars):
            
            if bid - bidbid * MyPoint > ema_val:
                price = bid - ecartbid * MyPoint
                sl_price = PartialHigh + StopLoss * MyPoint + AddPriceGap * MyPoint
                tp_price = PartialLow - TakeProfit * MyPoint - AddPriceGap * MyPoint
                volume = NR(Lot_Volume())
                
                request = {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_SELL_STOP,
                    "price": price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": Slippage,
                    "magic": MagicNumber,
                    "comment": mn,  # Use the same comment as defined in global parameters
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                result = mt5.order_send(request)
                print(f"Sell pending order result: {result}")
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Order failed with error code: {result.retcode}")
                    
                return

        # ----- Trailing Stop (and check auto-trading enabled) -----
        if trail_enabled:
            trail()
            
    except Exception as e:
        print(f"Error in start function: {e}")


# ***** Main loop *****
def main():
    print(f"Expert advisor {mn} started for {symbol}")
    
    # This sample loop polls for new ticks every few seconds.
    try:
        while True:
            start()
            # Clean up pending orders if needed:
            new_del()
            time.sleep(5)  # adjust the sleep interval as needed
            
    except KeyboardInterrupt:
        print("Terminating EA script.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


if __name__ == "__main__":
    main()