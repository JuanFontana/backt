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
import ta

import classes_backt_GAMMA_1
importlib.reload(classes_backt_GAMMA_1)

import functions_backt_GAMMA_1 as functions
importlib.reload(functions)

import tickers_backt_GAMMA_1 as tickers_list
importlib.reload(tickers_list)



def tickers_sectors(sector):
    if sector == 'communication_services':
        tickers = ['VZ', 'FB', 'DIS', 'GOOG', 'GOOGL', 'VZ', 'T', 'TMUS', 'EA', 'NFLX', 'CMCSA', 'ATVI', 'CHTR']
        return tickers
    
    elif sector == 'consumer_discretionary':
        tickers = ['HD', 'MCD', 'TSLA', 'HD', 'LOW', 'NKE']
        return tickers
    
    elif sector == 'consumer_staples':
        tickers = ['PG', 'WMT', 'KO', 'PEP', 'PM', 'MO', 'EL', 'CL', 'COST']
        return tickers
    
    elif sector == 'industrial':
        tickers = ['UNP', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'FDX']
        return tickers
    
    elif sector == 'information_tecnology':
        tickers = ['V', 'MSFT', 'NVDA', 'AAPL', 'ORCL', 'CRM', 'NOW', 'ADBE', 'INTU', 'MA', 'ACN', 'IBM', 'CSCO', 'AMAT', 'AVGO', 'MU', 'ADI'\
                   'INTC', 'AMD', 'QCOM']
        return tickers
    
    elif sector == 'health_care':
        tickers = ['JNJ', 'PFE', 'UNH', 'ABT', 'CVS', 'ABBV', 'GILD', 'LLY', 'CI', 'BMY', 'HCA', 'TMO', 'DHR', 'AMGN']
        return tickers
    
    elif sector == 'financial':
        tickers = ['JPM', 'BAC', 'WFC', 'C', 'TFC', 'CB', 'PGR', 'MS', 'SCHW', 'GS', 'SPGI', 'CME', 'BLK', 'AXP']
        return tickers
    
    elif sector == 'all':
        tickers = ['AAPL', 'AMD', 'AMZN', 'AXP', 'BAC', 'BA', 'BABA', 'BBD', 'V', 'C', 'CL', 'CCL', 'COST', 'D', 'DAL', 
                    'DIS', 'EA', 'FB', 'GOOG', 'GOLD', 'HLT', 'INTC', 'KO', 'LVS', 'MCD', 'MELI', 'NEE', 'NFLX', 'NOK', 'NKE', 'NVDA',
                    'PBR', 'PEP', 'PG', 'PM', 'SBUX', 'T', 'TSLA', 'UBER', 'WMT', 'MSFT', 'WFC', 'XOM', 'AA', 'X', 'CX', 'HL', 'F', 'GM', 
                    'ABBV', 'CMCSA', 'MRK', 'ORCL', 'CSCO', 'VZ', 'CVX', 'UNH', 'JNJ', 'ZM', 'MRNA', 'PLUG', 'ROKU', 'MU', 'GE', 'AAL',
                    'TWTR', 'UNP', 'CAT', '^GSPC']
        return tickers


def simulations_initialize(benchmark, length, data_cut, data_type, param_1, param_2, param_3, capital, 
                           k_position, strategy, bool_plot, bool_print, nb_decimals, sector):
    
    tickers = tickers_sectors(sector)
    
    
    pnl_sharpe_list = []
    drawdown_max_list = []
    drawdown_promedio_list = []
    pnl_mean_list = []
    pnl_volatility_list = []
    win_rate_list = []
    win_rate_long_list = []
    win_rate_short_list = []
    risk_benefit_ratio_list = []
    vcont_list = []
    var_95_list = []
    cvar_95_list = []
    nb_trades_list = []
    
    
    for i in tickers:
        #Inputs
        backtest = classes_backt_GAMMA_1.backtest(ticker = i, benchmark = benchmark, length = length)
        
        #Load data & split them between in_sample or out_of_sample
        backtest.load_time_series(data_cut, data_type) #out_of_sample // in_sample
        
        #Compute strategy
        backtest.compute_strategy(param_1, param_2, param_3, capital, k_position, strategy) 
        
        #Compute account
        backtest.compute_account(bool_plot)
        
        #Risk metrics
        backtest.risk_metrics(nb_decimals, bool_print, bool_plot = bool_plot)
        
        #Returns distribution
        backtest.returns_distribution(nb_decimals, bool_plot, bool_print)
        
        
        pnl_sharpe_list.append(backtest.pnl_sharpe)
        drawdown_max_list.append(backtest.drawdown_max)
        drawdown_promedio_list.append(backtest.drawdown_promedio)
        pnl_mean_list.append(backtest.pnl_mean)
        pnl_volatility_list.append(backtest.pnl_volatility)
        win_rate_list.append(backtest.win_rate)
        win_rate_long_list.append(backtest.win_rate_long)
        win_rate_short_list.append(backtest.win_rate_short)
        risk_benefit_ratio_list.append(backtest.risk_benefit_ratio)
        vcont_list.append(backtest.vcont)
        var_95_list.append(backtest.var_95)
        cvar_95_list.append(backtest.cvar_95)
        nb_trades_list.append(backtest.nb_trades)
    
    
    pnl_sharpe_list_mean = np.mean(pnl_sharpe_list)
    pnl_sharpe_list_std = np.std(pnl_sharpe_list)
    
    drawdown_max_list_mean = np.mean(drawdown_max_list)
    drawdown_max_list_std = np.std(drawdown_max_list)
    
    drawdown_promedio_list_mean = np.mean(drawdown_promedio_list)
    drawdown_promedio_list_std = np.std(drawdown_promedio_list)
    
    pnl_mean_list_mean = np.mean(pnl_mean_list)
    pnl_mean_list_std = np.std(pnl_mean_list)
    
    pnl_volatility_list_mean = np.mean(pnl_volatility_list)
    pnl_volatility_list_std = np.std(pnl_volatility_list)
    
    win_rate_list_mean = np.mean(win_rate_list)
    win_rate_list_std = np.std(win_rate_list)
    
    win_rate_long_list_mean = np.mean(win_rate_long_list)
    win_rate_long_list_std = np.std(win_rate_long_list)
    
    win_rate_short_list_mean = np.mean(win_rate_short_list)
    win_rate_short_list_std = np.std(win_rate_short_list)
    
    risk_benefit_ratio_list_mean = np.mean(risk_benefit_ratio_list)
    risk_benefit_ratio_list_std = np.std(risk_benefit_ratio_list)
    
    vcont_list_mean = np.mean(vcont_list)
    vcont_list_std = np.std(vcont_list)
    
    var_95_list_mean = np.mean(var_95_list)
    var_95_list_std = np.std(var_95_list)
    
    cvar_95_list_mean = np.mean(cvar_95_list)
    cvar_95_list_std = np.std(cvar_95_list)
    
    nb_trades_list_mean = np.mean(nb_trades_list)
    nb_trades_list_std = np.std(nb_trades_list)
    
    
    print('----------')
    print('Mean market / Std market')
    print('Total trades: ' + str(int(nb_trades_list_mean)) + ' / ' + str(int(nb_trades_list_std)))
    print('Dradown max: ' + str(round(drawdown_max_list_mean, 3)) + ' / ' + str(round(drawdown_max_list_std, 3)))
    print('Dradown prom: ' + str(round(drawdown_promedio_list_mean, 3)) + ' / ' + str(round(drawdown_promedio_list_std, 3)))
    print('Return: ' + str(round(pnl_mean_list_mean, 3)) + ' / ' + str(round(pnl_mean_list_std, 3)))
    print('Volatility: ' + str(round(pnl_volatility_list_mean, 3)) + ' / ' + str(round(pnl_volatility_list_std, 3)))
    print('Sharpe: ' + str(round(pnl_sharpe_list_mean, 3)) + ' / ' + str(round(pnl_sharpe_list_std, 3)))
    print('Win rate: ' + str(round(win_rate_list_mean, 3)) + ' / ' + str(round(win_rate_list_std, 3)))
    print('Win rate long: ' + str(round(win_rate_long_list_mean, 3)) + ' / ' + str(round(win_rate_long_list_std, 3)))
    print('Win rate short: ' + str(round(win_rate_short_list_mean, 3)) + ' / ' + str(round(win_rate_short_list_std, 3)))
    print('Benefit/Risk ratio: ' + str(round(risk_benefit_ratio_list_mean, 3)) + ' / ' + str(round(risk_benefit_ratio_list_std, 3)))
    print('Variation coefficient of net profit: ' + str(round(vcont_list_mean, 3)) + ' / ' + str(round(vcont_list_std, 3)))
    print('VaR : ' + str(round(var_95_list_mean, 3)) + ' / ' + str(round(var_95_list_std, 3)))
    print('CVaR : ' + str(round(cvar_95_list_mean, 3)) + ' / ' + str(round(cvar_95_list_std, 3)))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    