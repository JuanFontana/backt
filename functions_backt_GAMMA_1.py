import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import skew, kurtosis, chi2
from datetime import datetime as dt
import datetime
import importlib
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA
import pandas_ta as at

import classes_backt_GAMMA_1
importlib.reload(classes_backt_GAMMA_1)

import tickers_backt_GAMMA_1
importlib.reload(tickers_backt_GAMMA_1)

#Strategies avilable
#cruce_de_medias // cruce_de_medias_short // rsi_sobrecompra_sobreventa // etc.

def cruce_de_medias(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #Param_1 = mm_ lenta
    #Param_2 = mm_rapida
    
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker, benchmark, length)
    data = backtesting.load_time_series(data_cut, data_type)
    
    
    #Indicadores
    ta = pd.DataFrame()
    
    ta['Mm_lenta'] = data['Close'].rolling(param_1).mean()
    ta['Mm_rapida'] = data['Close'].rolling(param_2).mean()
    
    ta = pd.concat([data, ta], axis = 1)
    
    
    #Logica del trade
    #Loop for backtest
    size = ta.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = ta['Close'][n]
        data_prev = ta['Close_prev'][n]
        mm_lenta = ta['Mm_lenta'][n]
        mm_rapida = ta['Mm_rapida'][n]
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if mm_rapida > mm_lenta:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            elif mm_rapida < mm_lenta:
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or mm_rapida < mm_lenta:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if mm_rapida > mm_lenta:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
        
    
    return ta, mtx_backtest, columns



def cruce_de_medias_short(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #Param_1 = mm_ lenta
    #Param_2 = mm_rapida
    
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker, benchmark, length)
    data = backtesting.load_time_series(data_cut, data_type)
    
    
    #Indicadores
    ta = pd.DataFrame()
    
    ta['Mm_lenta'] = data['Close'].rolling(param_1).mean()
    ta['Mm_rapida'] = data['Close'].rolling(param_2).mean()
    
    ta = pd.concat([data, ta], axis = 1)
    
    
    #Logica del trade
    #Loop for backtest
    size = ta.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = ta['Close'][n]
        data_prev = ta['Close_prev'][n]
        mm_lenta = ta['Mm_lenta'][n]
        mm_rapida = ta['Mm_rapida'][n]
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if mm_rapida > 10000:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            elif mm_rapida < mm_lenta:
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or mm_rapida < mm_lenta:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if mm_rapida > mm_lenta:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
        
    
    return ta, mtx_backtest, columns


###----------------------------------------------------------------------------###


def rsi_sobrecompra_sobreventa(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #param_1 = level of overbought
    #param_2 = level of oversell
    #param_3 = length of rsi
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker = ticker, benchmark = benchmark, length = length)
    load_time_series = backtesting.load_time_series(data_cut = data_cut, data_type = data_type)
    
    
    #Compute indicator
    delta = load_time_series['Close'].diff()
    
    sube, baja = delta.copy(), delta.copy()
    sube[sube < 0] = 0
    baja[baja > 0] = 0
    
    media_sube = sube.rolling(param_3).mean()
    media_baja = baja.abs().rolling(param_3).mean()
    
    rs = media_sube / media_baja
    
    load_time_series['Rsi'] = 100 - (100/(1 + rs))
    
    
    #Logica del trade & Loop for backtest
    size = load_time_series.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = load_time_series['Close'][n]
        data_prev = load_time_series['Close_prev'][n]
        rsi = load_time_series['Rsi'][n]
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if rsi < param_2:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            elif rsi > param_1:
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or rsi > param_1:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if rsi < param_2:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
    

        
    
    return load_time_series, mtx_backtest, columns 


###----------------------------------------------------------------------------###


def suba_x_percent(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #param_1 = X%
    #param_2 = N dias
    #param_3 = False
    
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker = ticker, benchmark = benchmark, length = length)
    load_time_series = backtesting.load_time_series(data_cut = data_cut, data_type = data_type)
    
    
    #Compute indicator
    load_time_series['Subida'] = load_time_series['Ret'].rolling(param_1).sum()
    
    
    #Logica del trade & Loop for backtest
    size = load_time_series.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = load_time_series['Close'][n]
        data_prev = load_time_series['Close_prev'][n]
        x = load_time_series['Subida'][n]
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if x > param_2:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            elif x < -10:
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or x < 0:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if x > 0:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
    

        
    
    return load_time_series, mtx_backtest, columns 


###----------------------------------------------------------------------------###


def adx_dmi(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #param_1 = Length of dmix and adx
    #param_2 = None
    #param_3 = None
    
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker = ticker, benchmark = benchmark, length = length)
    load_time_series = backtesting.load_time_series(data_cut = data_cut, data_type = data_type)
    
    
    #Compute indicator
    adx_indicator = at.adx(high = load_time_series['High'], low = load_time_series['Low'], close = load_time_series['Close'], length = param_1)
    
    #Logica del trade & Loop for backtest
    size = load_time_series.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = load_time_series['Close'][n]
        data_prev = load_time_series['Close_prev'][n]
        
        dmip = adx_indicator['DMP'][n]
        dmin = adx_indicator['DMN'][n]
        adx = adx_indicator['ADX'][n]
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if dmip > dmin:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            elif dmin > dmip:
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or dmin > dmip:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if dmip > dmin:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
    

        
    
    return load_time_series, mtx_backtest, columns 


###----------------------------------------------------------------------------###


def adx_dmi_cruce_de_medias(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #param_1 = Length of dmix and adx
    #param_2 = None
    #param_3 = None
    
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker = ticker, benchmark = benchmark, length = length)
    load_time_series = backtesting.load_time_series(data_cut = data_cut, data_type = data_type)
    
    
    #Compute indicator
    adx_indicator = at.adx(high = load_time_series['High'], low = load_time_series['Low'], close = load_time_series['Close'], length = param_3)
    
    load_time_series['Mm_lenta'] = load_time_series['Close'].rolling(param_1).mean()
    load_time_series['Mm_rapida'] = load_time_series['Close'].rolling(param_2).mean()
    
    
    
    #Logica del trade & Loop for backtest
    size = load_time_series.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = load_time_series['Close'][n]
        data_prev = load_time_series['Close_prev'][n]
        
        dmip = adx_indicator['DMP'][n]
        dmin = adx_indicator['DMN'][n]
        adx = adx_indicator['ADX'][n]
        
        mm_l = load_time_series['Mm_lenta'][n]
        mm_r = load_time_series['Mm_rapida'][n]
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if dmip > dmin and mm_r > mm_l:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            elif dmin > dmip and mm_r < mm_l:
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or dmin > dmip:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if dmip > dmin:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
    

        
    
    return load_time_series, mtx_backtest, columns 


###----------------------------------------------------------------------------###


def mm_precio(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #param_1 = length mm
    #param_2 = non asigned
    #param_3 = non asigned
    
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker = ticker, benchmark = benchmark, length = length)
    load_time_series = backtesting.load_time_series(data_cut = data_cut, data_type = data_type)
    
    
    #Compute indicator
    load_time_series['Mm'] = load_time_series['Close'].rolling(param_1).mean()
    
    
    #Logica del trade & Loop for backtest
    size = load_time_series.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = load_time_series['Close'][n]
        data_prev = load_time_series['Close_prev'][n]
        mm = load_time_series['Mm'][n]
        
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if data > mm:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            
            elif data < mm :
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or data < mm:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if data > mm:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
    

        
    
    return load_time_series, mtx_backtest, columns

###----------------------------------------------------------------------------###


def supertrend(param_1, param_2, param_3, ticker, benchmark, length, data_cut, data_type, k_position):
    #param_1 = length super trend
    #param_2 = non asigned
    #param_3 = non asigned
    
    #Load data
    backtesting = classes_backt_GAMMA_1.backtest(ticker = ticker, benchmark = benchmark, length = length)
    load_time_series = backtesting.load_time_series(data_cut = data_cut, data_type = data_type)
    
    
    #Compute indicator
    supertrend = at.supertrend(high = load_time_series['High'], low = load_time_series['Low'], 
                               close = load_time_series['Close'], length = param_1)
    
    load_time_series['Super_trend'] = supertrend[f'SUPERT_{param_1}_3.0']
    load_time_series['Super_trend_position'] = supertrend[f'SUPERTd_{param_1}_3.0']
    load_time_series['Super_trend_long'] = supertrend[f'SUPERTl_{param_1}_3.0']
    load_time_series['Super_trend_short'] = supertrend[f'SUPERTs_{param_1}_3.0']
    
    
    #Logica del trade & Loop for backtest
    size = load_time_series.shape[0]
    columns = ['Position', 'Entry_signal', 'Exit_signal', 'Pnl_daily', 'Trade', 'Pnl_trade']
    position = 0
    can_trade = False
    mtx_backtest = np.zeros((size, len(columns)))
    
    for n in range(size):
        #Input data for the day
        data = load_time_series['Close'][n]
        data_prev = load_time_series['Close_prev'][n]
        supertrend = load_time_series['Super_trend'][n]
        l_s = load_time_series['Super_trend_position'][n]
        l = load_time_series['Super_trend_long'][n]
        s = load_time_series['Super_trend_short'][n]
        
        
        #Reset output data for the day
        pnl_daily = 0.
        trade = 0
        pnl_trade = 0.
        
        #Enter new position
        if position == 0:
            entry_signal = 0
            exit_signal = 0
            if l_s == 1:
                entry_signal = 1 #Buy signal
                position = 1
                entry_price = data
            
            elif l_s == -1 :
                entry_signal = -1 #Sell signal
                position = -1
                entry_price = data
        
        #Exit long position
        elif position == 1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if n == size - 1 or l_s == -1:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price) ##
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        #Exit short position
        elif position == -1:
            entry_signal = 0
            pnl_daily = position * (data - data_prev)
            if l_s == 1:
                exit_signal = 1 #Last day or take profit or stop loss
                pnl_trade = position * (data - entry_price)
                position = 0
                trade = 1
            
            else:
                exit_signal = 0
        
        
        #Save data for the day
        m = 0
        mtx_backtest[n][m] = position 
        m = m + 1
        
        mtx_backtest[n][m] = entry_signal
        m = m + 1
        
        mtx_backtest[n][m] = exit_signal
        m = m + 1
        
        mtx_backtest[n][m] = pnl_daily * k_position
        m = m + 1
        
        mtx_backtest[n][m] = trade 
        m = m + 1
        
        mtx_backtest[n][m] = pnl_trade * k_position
        m = m + 1
    

        
    
    return load_time_series, mtx_backtest, columns 