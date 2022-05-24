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
import pandas as pd
import numpy as np
import ta
import random
import yahoo_fin.stock_info as si
import pandas_ta as at

import classes_backt_GAMMA_1
importlib.reload(classes_backt_GAMMA_1)

import functions_backt_GAMMA_1 as functions
importlib.reload(functions)

import tickers_backt_GAMMA_1 as tickers_list
importlib.reload(tickers_list)

#Strategy used
strategy = functions.supertrend #cruce_de_medias // rsi_sobrecompra_sobreventa // suba_x_percent

#Inputs
backtest = classes_backt_GAMMA_1.backtest(ticker = 'X', benchmark = '^GSPC', length = 365 * 10)

#Load data & split them between in_sample or out_of_sample
load_time_series = backtest.load_time_series(data_cut = 0.7, data_type = 'in_sample') #out_of_sample // in_sample

#Compute strategy
backtest.compute_strategy(param_1 = 138, param_2 = 1, param_3 = 1, capital = 10_000.00, k_position = 15, strategy = strategy) 
# 1 = lvl_sobrecompra // 2 = lvl_sobreventa // 3 = length rsi

#Compute account
compute_account, resume = backtest.compute_account(bool_plot = True)

#Risk metrics
backtest.risk_metrics(nb_decimals = 3, bool_print = True, bool_plot = False)

#Returns distribution
backtest.returns_distribution(nb_decimals = 5, bool_plot = False, bool_print = True)

#Comparisson
backtest.comparison(df = backtest.data, bool_print = True)

#Simulations of trategu with market
backtest.simulations(bool_print = False, bool_plot = False, sector = 'all')

# Optimization
param_1_list = list(range(130, 150, 1))
param_2_list = list(range(1, 2, 1))
param_3_list = list(range(1, 250, 1))

df_sharpe, df_nb_trades, df_returns, df_volatility, df_win_rate, df_max_dd, df_benefit_risk_ratio,\
    df_variation_coefficient, df_win_rate_long, df_win_rate_short = \
    backtest.optimize(param_1_list, param_2_list, param_3_list, bool_plot = False, bool_print = False, optimize = '1&2') 










































