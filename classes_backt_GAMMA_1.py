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
import random

import classes_backt_GAMMA_1
importlib.reload(classes_backt_GAMMA_1)

import functions_backt_GAMMA_1 as functions
importlib.reload(functions)

import tickers_backt_GAMMA_1 as tickers_list
importlib.reload(tickers_list)


class backtest():
    
    def __init__(self, ticker, benchmark , length):
        self.ticker = ticker
        self.benchmark = benchmark
        self.length = length
        
        #load_time_series
        self.data_cut = 0.00
        self.data_type = None
        
        #Compute strategy
        self.pnl_mean = 0.00
        self.pnl_volatility = 0.00
        self.pnl_sharpe = 0.00
        self.nb_trades = 0.00
        
        
        #Risk merics
        self.win_rate = 0.00
        self.average_win = 0.00
        self.average_loose = 0.00
        
        self.nb_decimals = 3
    
    
    def load_time_series(self, data_cut, data_type):
        self.data_cut = data_cut
        self.data_type = data_type
        
        #length - calculo
        hoy = dt.now()
        start = hoy - datetime.timedelta(days = self.length)
        
        
        #Get market data
        accion = yf.download(tickers = self.ticker, start = start , end = hoy )
        benchmark = yf.download(tickers = self.benchmark, start = start , end = hoy )
        
        
        # Create a table of returns - stock
        self.data = pd.DataFrame()
        self.data['Open'] = accion['Open']
        self.data['High'] = accion['High']
        self.data['Low'] = accion['Low']
        self.data['Close'] = accion['Adj Close']
        self.data['Close_prev'] = accion['Adj Close'].shift(1)
        self.data['Ret'] = accion['Adj Close'] / accion['Adj Close'].shift(1) - 1
        self.data['Ret_cum'] = self.data['Ret'].cumsum()
        self.data['Date'] = pd.to_datetime(accion.index, dayfirst = True)
        self.data['Volume'] = accion['Volume']

        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop = True)
        
        self.data.index = self.data['Date']
        
        
        # Create a table of returns - benchmark
        self.dataB = pd.DataFrame()
        self.dataB['Open'] = benchmark['Open']
        self.dataB['High'] = benchmark['High']
        self.dataB['Low'] = benchmark['Low']
        self.dataB['Close'] = benchmark['Adj Close']
        self.dataB['Ret'] = benchmark['Adj Close'] / benchmark['Adj Close'].shift(1) - 1
        self.dataB['Ret_cum'] = self.dataB['Ret'].cumsum()
        self.dataB['Date'] = pd.to_datetime(benchmark.index, dayfirst = True)
        self.dataB['Volume'] = benchmark['Volume']

        self.dataB = self.dataB.dropna()
        self.dataB = self.dataB.reset_index(drop = True)
        
        self.dataB.index = self.dataB['Date']
        
        
        #Definiendo in_sample & out_of_sample para data
        cut = int(self.data_cut * self.data.shape[0])
        
        if self.data_type == 'in_sample':
            df = self.data.head(cut) #Me devuelve el 70% de los dias
       
        elif self.data_type == 'out_of_sample':
            df = self.data.tail(self.data.shape[0] - cut) #Me devuelve el 30% de los dias
        
        df = df.reset_index(drop = True)
        
        self.dataframe_data = df
        
        
        #Definiendo in_sample & out_of_sample para dataB
        cut = int(self.data_cut * self.dataB.shape[0])
        
        if self.data_type == 'in_sample':
            dfB = self.dataB.head(cut) #Me devuelve el 70% de los dias
       
        elif self.data_type == 'out_of_sample':
            dfB = self.dataB.tail(self.data.shape[0] - cut) #Me devuelve el 30% de los dias
        
        dfB = dfB.reset_index(drop = True)
        
        self.dataframe_dataB = dfB
        
    
        return  self.dataframe_data
    
    
    def synchronise_timeseries(self): 
        #Load time series
        t1 = self.dataframe_strategy
        t2 = self.dataframe_dataB
        
        
        #Sincronizar series de tiempo
        timestamp1 = list(t1['Date'].values)
        timestamp2 = list(t2['Date'].values)
        timestamps = list(set(timestamp1) & set(timestamp2)) #Que me agarre solo los dias que estan en ambos
        
        
        #Sincronizando la serie de tiempo para x1
        t1_sync = t1[t1['Date'].isin(timestamps)] #Le pido que filtre solo por los valores que esten en timestamps
        t1_sync.sort_values(by = 'Date', ascending = True) #Ordeno los valores por fechas de manera ascendente
        t1_sync = t1_sync.reset_index(drop = True)
        
        
        #Sincronizando la serie de tiempo para x2
        t2_sync = t2[t2['Date'].isin(timestamps)] #Le pido que filtre solo por los valores que esten en timestamps
        t2_sync.sort_values(by = 'Date', ascending = True) #Ordeno los valores por fechas de manera ascendente
        t2_sync = t2_sync.reset_index(drop = True)
        
        
        #Tabla de retornos para el ticker y el benchmark
        t = pd.DataFrame()
        
        t['Date'] = t1_sync['Date']
        
        t['Price_1'] = t1_sync['Cum_pnl_daily']
        t['Price_2'] = t2_sync['Close']
        
        t['Price_1_previous'] = t1_sync['Cum_pnl_daily'].shift(1)
        t['Price_2_previous'] = t2_sync['Close'].shift(1)
        
        t = t.drop(t.index[[0]])
        t = t.drop(t.index[[0]])
        
        
        t['Return_1'] = t1_sync['Returns_daily']
        t['Return_2'] = t2_sync['Ret']
        
        
        #Compute vectors of returns
        returns_ticker = t['Return_2'].values #Y
        returns_benchmark = t['Return_1'].values #X 
        
        return returns_ticker, returns_benchmark, t
    
    
    def compute_strategy(self, param_1, param_2, param_3, capital, k_position, strategy):
        self.capital = capital
        self.k_position = k_position
        self.strategy_used = strategy
        
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3
        self.strategy = self.strategy_used(ticker = self.ticker, benchmark = self.benchmark, length = self.length,
                                            data_cut = self.data_cut, data_type = self.data_type,
                                            param_1 = param_1, param_2 = param_2, param_3 = param_3, k_position = self.k_position)
        
    
    
    def compute_account(self, bool_plot = False):
        _, mtx_backtest, columns = self.strategy_used(ticker = self.ticker, benchmark = self.benchmark,
                                                             length = self.length, data_cut = self.data_cut, 
                                                             data_type = self.data_type, param_1 = self.param_1, 
                                                             param_2 = self.param_2, param_3 = self.param_3, k_position = self.k_position)
        
        df1 = pd.DataFrame(data = mtx_backtest, columns = columns)
        df1 = df1.dropna()
        df1 = df1.reset_index(drop = True)
        df1['Cum_pnl_daily'] = np.cumsum(df1['Pnl_daily']) + self.capital
        self.dataframe_strategy = df1
        
        self.dataframe_strategy['Returns_daily'] = self.dataframe_strategy['Cum_pnl_daily'].pct_change()[1:]
        self.dataframe_strategy['Returns_daily'] = self.dataframe_strategy['Returns_daily']\
            .replace([np.inf, -np.inf], 0) #Le saco los valores infinitos
        
        self.dataframe_strategy['Cum_returns_daily'] = np.cumsum(self.dataframe_strategy['Returns_daily'])
        
        self.dataframe_strategy['Date'] = self.dataframe_data['Date']
        
        #Compute Sarpe ratio and numbers of trades
        vec_pnl = df1['Returns_daily'].values
        self.pnl_mean = np.round(np.mean(vec_pnl[1:]) * 252, 4)
        self.pnl_volatility = np.round(np.std(vec_pnl[1:]) * np.sqrt(252), 4)
        self.pnl_sharpe = np.round(self.pnl_mean / self.pnl_volatility, 4)
        self.resume = df1[df1['Trade'] == 1]
        self.nb_trades = len(self.resume)
        

        
        if bool_plot == True:
            self.plot_pnl_strategy()
            self.plot_rdto_strategy()
        
        
        return self.dataframe_strategy, self.resume
    
        
    
    def plot_pnl_strategy(self):
        equity_curve = pd.DataFrame()
        window = 50
        
        equity_curve['Promedio cada ' + str(window)] = \
            self.dataframe_strategy['Cum_pnl_daily'].rolling(window = window).mean()
            
        equity_curve['Desviacion cada ' + str(window)] = \
            equity_curve['Promedio cada ' + str(window)].rolling(window = window).std()
        
        equity_curve['Desviacion cada ' + str(window) + '+'] = (equity_curve['Promedio cada ' + str(window)])\
            + (equity_curve['Desviacion cada ' + str(window)])
            
        equity_curve['Desviacion cada ' + str(window) + '-'] = equity_curve['Promedio cada ' + str(window)]\
            - equity_curve['Desviacion cada ' + str(window)]
        
        #Equity curve
        
        plt.figure()
        plt.title('Cumulative P&L of strategy with ' + self.ticker  + '\n' \
                  + 'Return(Y) ' + str(np.round(self.pnl_mean, 3))  + '\n' \
                  + 'Volatility(Y) ' + str(np.round(self.pnl_volatility, 3))  + '\n' \
                  + 'Sharpe(Y) ' + str(np.round(self.pnl_sharpe, 3))) 
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        plt.plot(self.dataframe_strategy['Cum_pnl_daily'], 'k', label = 'Strategy')
        plt.plot(equity_curve['Promedio cada ' + str(window)], 'y', label = 'Promedio cada ' + str(window))
        plt.plot(equity_curve['Desviacion cada ' + str(window) + '+'], 'g', label = 'Desviacion cada ' + str(window) + ' +')
        plt.plot(equity_curve['Desviacion cada ' + str(window) + '-'], 'r', label = 'Desviacion cada ' + str(window) + ' -')
        
        
        plt.grid()
        plt.legend()
        plt.show()

        
    
    def plot_rdto_strategy(self):
        #Equity curve rdto
        plt.figure()
        plt.title('Cumulative Return of strategy with ' + self.ticker  + '\n' \
                  + ' / K_position ' + str(np.round(self.k_position, 3))\
                  + ' / Nb trades ' + str(np.round(self.nb_trades, 3))  + '\n' \
                  + ' / Return(Y) ' + str(np.round(self.pnl_mean, 3))\
                  + ' / Volatility(Y) ' + str(np.round(self.pnl_volatility, 3))  + '\n' \
                  + ' / Sharpe(Y) ' + str(np.round(self.pnl_sharpe, 3))) 
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        plt.plot(self.dataframe_strategy['Cum_returns_daily'], 'k', label = 'Strategy')
        
        plt.grid()
        plt.legend()
        plt.show()
        
    

    
    
    def risk_metrics(self, nb_decimals = 3, bool_print = False, bool_plot = False):
        #Parameters
        total_win_trades_long = np.where((self.dataframe_strategy['Pnl_trade'] > 0) &\
                                              (self.dataframe_strategy['Position'].shift(1) == 1), 1, 0).sum()

        total_trades_long = np.where((self.dataframe_strategy['Exit_signal'] == 1) &\
                                          (self.dataframe_strategy['Position'].shift(1) == 1), 1, 0).sum()
        
            
        total_win_trades_short = np.where((self.dataframe_strategy['Pnl_trade'] > 0) &\
                                              (self.dataframe_strategy['Position'].shift(1) == -1), 1, 0).sum()

        total_trades_short = np.where((self.dataframe_strategy['Exit_signal'] == 1) &\
                                          (self.dataframe_strategy['Position'].shift(1) == -1), 1, 0).sum()        
        
        
        #Profits
        self.average_win = np.round(np.where(self.dataframe_strategy['Pnl_trade'] > 0, self.dataframe_strategy['Pnl_trade'], 0).sum()\
             / self.dataframe_strategy['Trade'].sum(), nb_decimals) #Dollars
            
        
        self.average_loose = np.round(np.where(self.dataframe_strategy['Pnl_trade'] < 0, self.dataframe_strategy['Pnl_trade'], 0).sum()\
             / self.dataframe_strategy['Trade'].sum(), nb_decimals) #Dollars
        
        
        self.best_trade = np.round(self.dataframe_strategy['Pnl_trade'].max(), nb_decimals)
        
        
        self.worst_trade = np.round(self.dataframe_strategy['Pnl_trade'].min(), nb_decimals)
        
        
        self.total_pnl = np.round(list(self.dataframe_strategy['Cum_pnl_daily'])[-1], nb_decimals)
        
        
        #vcont = Variation coefficient of net profit
        self.vcont = np.round(np.std(self.resume['Pnl_trade']) / self.average_win, nb_decimals)
        
        
        self.risk_benefit_ratio = np.round(self.average_win / abs(self.average_loose), nb_decimals)
        
        
        self.drawdown = pd.DataFrame()
        self.drawdown = (self.dataframe_strategy['Cum_pnl_daily'] - self.dataframe_strategy['Cum_pnl_daily'].cummax())\
            / self.dataframe_strategy['Cum_pnl_daily'].cummax()
        
        self.drawdown = self.drawdown.replace([np.inf, -np.inf], 0)
        
        self.drawdown_max = np.round(self.drawdown.min(), nb_decimals)
        self.drawdown_promedio = np.round(self.drawdown.mean(), nb_decimals)
        
        
        #Probabilities
        self.win_rate = np.round(np.where(self.dataframe_strategy['Pnl_trade'] > 0, 1, 0).sum()\
            / self.dataframe_strategy['Trade'].sum(), nb_decimals)
        
        
        self.win_rate_long = np.round(total_win_trades_long / total_trades_long, nb_decimals)
        
        
        self.win_rate_short = np.round(total_win_trades_short / total_trades_short, nb_decimals)
        
        
        self.average_win_p = np.round(np.where(self.dataframe_strategy['Pnl_trade'] >= self.average_win, 1, 0).sum()\
            / self.dataframe_strategy['Trade'].sum(), nb_decimals)
        
        
        self.average_loose_p = np.round(np.where(self.dataframe_strategy['Pnl_trade'] <= self.average_loose, 1, 0).sum()\
            / self.dataframe_strategy['Trade'].sum(), nb_decimals)
        
        
        #Linear regression
        self.x, self.y, self.t = self.synchronise_timeseries()
        
        slope, intercept, r_value, p_value, std_err = linregress(self.x,self.y) #slope = Beta
        
        self.beta = np.round(slope, nb_decimals)
        self.alpha = np.round(intercept, nb_decimals)
        self.p_value = np.round(p_value, nb_decimals)
        
        self.null_hypothesis = p_value > 0.05 
        #Si p_value < 0.05 reject null hypothesis 
        #La null hypothesis, si la puedo rechazar me sirve para saber si los datos son estadisticamente significativos
        
        self.correlation = np.round(r_value, nb_decimals)
        self.r_squared = np.round(r_value**2, nb_decimals) #pct de varianza de y explicada por x
        self.predictor_linreg = self.beta * self.x + self.alpha
        
        
        if bool_print:
            self.print_riskmetrics()
        
        if bool_plot:
            self.plot_riskmetrics()
    
    
    def print_riskmetrics(self):
        print('----------')
        print('Ticker: ' + str(self.ticker))
        print('Benchmark: ' + str(self.benchmark))
        print('Length: ' + str(len(self.dataframe_strategy)))
        print('Data cut: ' + str(self.data_cut))
        print('Data type: ' + str(self.data_type))
        print('Total trades: ' + str(self.nb_trades))
        print('Total P&L: ' + str(self.total_pnl))
        print('Dradown max: ' + str(self.drawdown_max))
        print('Dradown prom: ' + str(self.drawdown_promedio))
        print('Return: ' + str(self.pnl_mean))
        print('Volatility: ' + str(self.pnl_volatility))
        print('Sharpe: ' + str(self.pnl_sharpe))
        print('Win rate: ' + str(self.win_rate))
        print('Win rate long: ' + str(self.win_rate_long))
        print('Win rate short: ' + str(self.win_rate_short))
        print('Expected profit: ' + str(self.average_win))
        print('Expected profit probability: ' + str(self.average_win_p))
        print('Expected loose: ' + str(self.average_loose))
        print('Expected loose probability: ' + str(self.average_loose_p))
        print('Benefit/Risk ratio: ' + str(self.risk_benefit_ratio))
        print('Variation coefficient of net profit: ' + str(self.vcont))
        print('Correlation with benchmark: ' + str(self.correlation))
        print('Beta with benchmark: ' + str(self.beta))
        print('Best trade: ' + str(self.best_trade))
        print('Worst trade: ' + str(self.worst_trade))
    

    def plot_riskmetrics(self):
        #Drawdown
        plt.figure()
        plt.title('Dradown ' + self.ticker)
            
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        plt.plot(self.drawdown, 'r', label = 'Strategy_DD')
        
        plt.grid()
        plt.legend()
        plt.show()
        
        
        #Linear regression
        str_tittle = 'Scatterplot of returns' + '\n'\
            + 'Linear regression / ticker ' + self.ticker\
            + ' / benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' / beta (slope) ' +str(self.beta)\
            + ' / p_value ' + str(self.p_value)\
            + ' / null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'Correlacion ' + str(self.correlation)\
            + ' / Informacion explicada (r_squared) ' + str(self.r_squared)
            
        plt.figure()
        plt.title(str_tittle)
        
        plt.scatter(self.x,self.y, color = 'k')
        plt.plot(self.x, self.predictor_linreg, color = 'r')
        
        plt.ylabel(self.ticker)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()
        
        
        #Rdto account & benchmark & accion(buy and hold)
        plt.figure()
        plt.title('Rdto account & benchmark')
        plt.xlabel('Time')
        plt.ylabel('%') 
        
        self.dataframe_strategy['Cum_returns_daily'].plot(color = 'k', label = self.ticker)
        
        self.dataframe_dataB['Ret_cum'].plot(color = 'b', label = self.benchmark)
        self.dataframe_data['Ret_cum'].plot(color = 'r', label = self.ticker + ' B&H')
        
        plt.legend()
        plt.show()
    
    
    
    def returns_distribution(self, nb_decimals, bool_print = False, bool_plot = False):
        self.nb_decimals = nb_decimals
        
        self.rdto_clean = pd.DataFrame(index = self.data['Date'])
        
        self.rdto_clean =self.dataframe_strategy['Returns_daily'].replace(0, np.nan)
        self.rdto_clean = self.rdto_clean.dropna()
        self.rdto_clean = self.rdto_clean.reset_index(drop = True)
        
        #Riskemtrics - returns distribution
        self.mean = np.mean(self.rdto_clean)
        self.std = np.std(self.rdto_clean)
        self.skew = skew(self.rdto_clean)
        self.kurtosis = kurtosis(self.rdto_clean)
        self.var_95 = np.percentile(self.rdto_clean, 5)
        self.cvar_95 = np.mean(self.rdto_clean[self.rdto_clean <= self.var_95])
        self.jb =  len(self.rdto_clean)/6*(self.skew**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - chi2.cdf(self.jb, df = 2)
        self.is_normal = (self.p_value > 0.05) 
        
        if bool_print == True:
            self.print_returns_distribution()
        
        if bool_plot == True:
            self.plot_returns_distribution()
    
    
    def print_returns_distribution(self):
        print('----------')
        print('Mean ' + str(round(self.mean,  self.nb_decimals)))
        print('Std ' + str(round(self.std,  self.nb_decimals)))
        print('Skewness ' + str(round(self.skew,  self.nb_decimals)))
        print('Kurtosis ' + str(round(self.kurtosis,  self.nb_decimals)))
        print('VaR ' + str(round(self.var_95,  self.nb_decimals)))
        print('CVar ' + str(round(self.cvar_95,  self.nb_decimals)))
        print('Is normal ' + str(self.is_normal))
    
    
    def plot_returns_distribution(self):
        #Drawdown
        plt.figure()
        plt.title('Returns ' + self.ticker)
            
        plt.xlabel('Time')
        plt.ylabel('%')
        
        plt.hist(self.rdto_clean, bins = 100, edgecolor='black')
        
        plt.grid()
        plt.show()
        
    
    def optimize(self, param_1_list, param_2_list, param_3_list, bool_plot = False, bool_print = False, optimize = '1&2'):
        self.param_1_list = param_1_list
        self.param_2_list = param_2_list
        self.param_3_list = param_3_list
        
        if optimize == '1&2':
            mtx_sharpe = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_nb_trades = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_returns = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_volatility = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_win_rate =  np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_win_rate_long = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_win_rate_short = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_max_dd = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_benefit_risk_ratio = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            mtx_variation_coefficient = np.zeros((len(self.param_1_list), len(self.param_2_list)))
            
            p1 = 0
            for i in self.param_1_list:
                p2 = 0
                for z in self.param_2_list:
                    if i > z:
                        #Load data
                        self.load_time_series(self.data_cut, self.data_type)
                        
                        #Compute indicator
                        self.compute_strategy(i, z, self.param_3_list[0], self.capital, self.k_position, self.strategy_used)                                   
                        
                        #Run backtest of traiding strategy
                        self.compute_account(bool_plot = bool_plot)
                        
                        #Risk metrics
                        self.risk_metrics(self.nb_decimals, bool_print = bool_print, bool_plot = bool_plot)
                        
                        #Save data in tables min 
                        mtx_sharpe[p1][p2] = self.pnl_sharpe
                        
                        mtx_nb_trades[p1][p2] = self.nb_trades
                        
                        mtx_returns[p1][p2]= self.pnl_mean
                        
                        mtx_volatility[p1][p2] = self.pnl_volatility
                        
                        mtx_win_rate[p1][p2] = self.win_rate
                        
                        mtx_win_rate_long[p1][p2] = self.win_rate_long
                        
                        mtx_win_rate_short[p1][p2] = self.win_rate_short
                        
                        mtx_max_dd[p1][p2] = self.drawdown_max
                        
                        mtx_benefit_risk_ratio[p1][p2] = self.risk_benefit_ratio
                        
                        mtx_variation_coefficient[p1][p2] = self.vcont
                            
                        p2 += 1
                    else:
                        continue
                p1 += 1
    
            
            #Df_Sharpe
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_sharpe, columns = column_names)
            self.df_sharpe = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_sharpe = self.df_sharpe.dropna()
            self.df_sharpe = self.df_sharpe.reset_index(drop = True)
            
            #Df_nb_trades
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_nb_trades, columns = column_names)
            self.df_nb_trades = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_nb_trades = self.df_nb_trades.dropna()
            self.df_nb_trades = self.df_nb_trades.reset_index(drop = True)
            
            #Df_returns
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_returns, columns = column_names)
            self.df_returns = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_returns = self.df_returns.dropna()
            self.df_returns = self.df_returns.reset_index(drop = True)
            
            #Df_volatility
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data =  mtx_volatility, columns = column_names)
            self.df_volatility = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_volatility = self.df_volatility.dropna()
            self.df_volatility = self.df_volatility.reset_index(drop = True)
            
            #Df_win_rate
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_win_rate, columns = column_names)
            self.df_win_rate = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_win_rate = self.df_win_rate.dropna()
            self.df_win_rate = self.df_win_rate.reset_index(drop = True)
            
            #Df_win_rate_long
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_win_rate_long, columns = column_names)
            self.df_win_rate_long = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_win_rate_long = self.df_win_rate_long.dropna()
            self.df_win_rate_long = self.df_win_rate_long.reset_index(drop = True)
            
            #Df_win_rate_short
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_win_rate_short, columns = column_names)
            self.df_win_rate_short = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_win_rate_short = self.df_win_rate_short.dropna()
            self.df_win_rate_short = self.df_win_rate_short.reset_index(drop = True)
            
            #Df_max_dd
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_max_dd, columns = column_names)
            self.df_max_dd = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_max_dd = self.df_max_dd.dropna()
            self.df_max_dd = self.df_max_dd.reset_index(drop = True)
            
            #Df_benefit_risk_ratio
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_benefit_risk_ratio, columns = column_names)
            self.df_benefit_risk_ratio = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_benefit_risk_ratio = self.df_benefit_risk_ratio.dropna()
            self.df_benefit_risk_ratio = self.df_benefit_risk_ratio.reset_index(drop = True)
            
            #Df_variation_coefficient
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P2_' + str(z) for z in self.param_2_list]
            df2 = pd.DataFrame(data = mtx_variation_coefficient, columns = column_names)
            self.df_variation_coefficient = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_variation_coefficient = self.df_variation_coefficient.dropna()
            self.df_variation_coefficient = self.df_variation_coefficient.reset_index(drop = True)
        
        elif optimize == '1&3':
            mtx_sharpe = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_nb_trades = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_returns = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_volatility = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_win_rate =  np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_win_rate_long = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_win_rate_short = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_max_dd = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_benefit_risk_ratio = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            mtx_variation_coefficient = np.zeros((len(self.param_1_list), len(self.param_3_list)))
            
            p1 = 0
            for i in self.param_1_list:
                p2 = 0
                for z in self.param_3_list:
                    if i > z:
                        #Load data
                        self.load_time_series(self.data_cut, self.data_type)
                        
                        #Compute indicator
                        self.compute_strategy(i, self.param_2_list[0], z, self.capital, self.k_position, self.strategy_used)                                   
                        
                        #Run backtest of traiding strategy
                        self.compute_account(bool_plot = bool_plot)
                        
                        #Risk metrics
                        self.risk_metrics(self.nb_decimals, bool_print = bool_print, bool_plot = bool_plot)
                        
                        #Save data in tables min 
                        mtx_sharpe[p1][p2] = self.pnl_sharpe
                        
                        mtx_nb_trades[p1][p2] = self.nb_trades
                        
                        mtx_returns[p1][p2]= self.pnl_mean
                        
                        mtx_volatility[p1][p2] = self.pnl_volatility
                        
                        mtx_win_rate[p1][p2] = self.win_rate
                        
                        mtx_win_rate_long[p1][p2] = self.win_rate_long
                        
                        mtx_win_rate_short[p1][p2] = self.win_rate_short
                        
                        mtx_max_dd[p1][p2] = self.drawdown_max
                        
                        mtx_benefit_risk_ratio[p1][p2] = self.risk_benefit_ratio
                        
                        mtx_variation_coefficient[p1][p2] = self.vcont
                            
                        p2 += 1
                    else:
                        continue
                p1 += 1
                
            
            #Df_Sharpe
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_sharpe, columns = column_names)
            self.df_sharpe = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_sharpe = self.df_sharpe.dropna()
            self.df_sharpe = self.df_sharpe.reset_index(drop = True)
            
            #Df_nb_trades
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_nb_trades, columns = column_names)
            self.df_nb_trades = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_nb_trades = self.df_nb_trades.dropna()
            self.df_nb_trades = self.df_nb_trades.reset_index(drop = True)
            
            #Df_returns
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_returns, columns = column_names)
            self.df_returns = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_returns = self.df_returns.dropna()
            self.df_returns = self.df_returns.reset_index(drop = True)
            
            #Df_volatility
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data =  mtx_volatility, columns = column_names)
            self.df_volatility = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_volatility = self.df_volatility.dropna()
            self.df_volatility = self.df_volatility.reset_index(drop = True)
            
            #Df_win_rate
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_win_rate, columns = column_names)
            self.df_win_rate = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_win_rate = self.df_win_rate.dropna()
            self.df_win_rate = self.df_win_rate.reset_index(drop = True)
            
            #Df_win_rate_long
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_win_rate_long, columns = column_names)
            self.df_win_rate_long = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_win_rate_long = self.df_win_rate_long.dropna()
            self.df_win_rate_long = self.df_win_rate_long.reset_index(drop = True)
            
            #Df_win_rate_short
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_win_rate_short, columns = column_names)
            self.df_win_rate_short = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_win_rate_short = self.df_win_rate_short.dropna()
            self.df_win_rate_short = self.df_win_rate_short.reset_index(drop = True)
            
            #Df_max_dd
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_max_dd, columns = column_names)
            self.df_max_dd = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_max_dd = self.df_max_dd.dropna()
            self.df_max_dd = self.df_max_dd.reset_index(drop = True)
            
            #Df_benefit_risk_ratio
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_benefit_risk_ratio, columns = column_names)
            self.df_benefit_risk_ratio = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_benefit_risk_ratio = self.df_benefit_risk_ratio.dropna()
            self.df_benefit_risk_ratio = self.df_benefit_risk_ratio.reset_index(drop = True)
            
            #Df_variation_coefficient
            df1 = pd.DataFrame()
            df1['P1'] = param_1_list                                                         
            column_names = ['P3_' + str(a) for a in self.param_3_list]
            df2 = pd.DataFrame(data = mtx_variation_coefficient, columns = column_names)
            self.df_variation_coefficient = pd.concat([df1, df2], axis = 1) #Axis 0 for rows, axis = 1 for columns
            self.df_variation_coefficient = self.df_variation_coefficient.dropna()
            self.df_variation_coefficient = self.df_variation_coefficient.reset_index(drop = True)
        
        
        
        return self.df_sharpe, self.df_nb_trades, self.df_returns, self.df_volatility, self.df_win_rate,\
            self.df_max_dd, self.df_benefit_risk_ratio, self.df_variation_coefficient, self.df_win_rate_long, self.df_win_rate_short
            
    
    def comparison(self, df, bool_print = False):
        #Compute Sarpe ratio, return and volatility
        vec_pnl = df['Ret'].values
        self.pnl_mean_comparison = np.round(np.mean(vec_pnl[1:]) * 252, self.nb_decimals)
        self.pnl_volatility_comparison = np.round(np.std(vec_pnl[1:]) * np.sqrt(252), self.nb_decimals)
        self.pnl_sharpe_comparison = np.round(self.pnl_mean_comparison / self.pnl_volatility_comparison, self.nb_decimals)

        
        #Drawdown
        drawdown = pd.DataFrame()
        drawdown = (df['Close'] - df['Close'].cummax())\
            / df['Close'].cummax()
        
        drawdown = drawdown.replace([np.inf, -np.inf], 0)
        
        self.drawdown_max_comparison = np.round(drawdown.min(), self.nb_decimals)
        self.drawdown_promedio_comparison = np.round(drawdown.mean(), self.nb_decimals)
        
        if bool_print == True:
            self.print_comparison()
    
    
    def print_comparison(self):
        print('----------')
        print('Comparison You / Other')
        print('Return: ' + str(self.pnl_mean) + ' / ' + str(self.pnl_mean_comparison))
        print('Volatility: ' + str(self.pnl_volatility) + ' / ' + str(self.pnl_volatility_comparison))
        print('Sharpe: '  + str(self.pnl_sharpe) + ' / ' + str(self.pnl_sharpe_comparison))
        print('Drawdown max: '  + str(self.drawdown_max) + ' / ' + str(self.drawdown_max_comparison))
        print('Drawdown prom: '  + str(self.drawdown_promedio) + ' / ' + str(self.drawdown_promedio_comparison))
    
    
    def simulations(self, bool_print = False, bool_plot = False, sector = 'all'):
        tickers_list.simulations_initialize(self.benchmark, 
                                            self.length, 
                                            self.data_cut, 
                                            self.data_type, 
                                            self.param_1, 
                                            self.param_2, 
                                            self.param_3, 
                                            self.capital, 
                                            self.k_position, 
                                            self.strategy_used, 
                                            bool_plot, 
                                            bool_print, 
                                            self.nb_decimals, 
                                            sector)

        
    #     if bool_print == True:
    #         self.print_simulations()
        
        
    # def print_simulations(self):
    #     print('----------')
    #     print('Mean market / Std market')
    #     print('Total trades: ' + str(round(self.nb_trades_list_mean, self.nb_decimals)) + ' / ' + str(round(self.nb_trades_list_std, self.nb_decimals)))
    #     print('Dradown max: ' + str(round(self.drawdown_max_list_mean, self.nb_decimals)) + ' / ' + str(round(self.drawdown_max_list_std, self.nb_decimals)))
    #     print('Dradown prom: ' + str(round(self.drawdown_promedio_list_mean, self.nb_decimals)) + ' / ' + str(round(self.drawdown_promedio_list_std, self.nb_decimals)))
    #     print('Return: ' + str(round(self.pnl_mean_list_mean, self.nb_decimals)) + ' / ' + str(round(self.pnl_mean_list_std, self.nb_decimals)))
    #     print('Volatility: ' + str(round(self.pnl_volatility_list_mean, self.nb_decimals)) + ' / ' + str(round(self.pnl_volatility_list_std, self.nb_decimals)))
    #     print('Sharpe: ' + str(round(self.pnl_sharpe_list_mean, self.nb_decimals)) + ' / ' + str(round(self.pnl_sharpe_list_std, self.nb_decimals)))
    #     print('Win rate: ' + str(round(self.win_rate_list_mean, self.nb_decimals)) + ' / ' + str(round(self.win_rate_list_std, self.nb_decimals)))
    #     print('Win rate long: ' + str(round(self.win_rate_long_list_mean, self.nb_decimals)) + ' / ' + str(round(self.win_rate_long_list_std, self.nb_decimals)))
    #     print('Win rate short: ' + str(round(self.win_rate_short_list_mean, self.nb_decimals)) + ' / ' + str(round(self.win_rate_short_list_std, self.nb_decimals)))
    #     print('Benefit/Risk ratio: ' + str(round(self.risk_benefit_ratio_list_mean, self.nb_decimals)) + ' / ' + str(round(self.risk_benefit_ratio_list_std, self.nb_decimals)))
    #     print('Variation coefficient of net profit: ' + str(round(self.vcont_list_mean, self.nb_decimals)) + ' / ' + str(round(self.vcont_list_std, self.nb_decimals)))
    #     print('VaR : ' + str(round(self.var_95_list_mean, self.nb_decimals)) + ' / ' + str(round(self.var_95_list_std, self.nb_decimals)))
    #     print('CVaR : ' + str(round(self.cvar_95_list_mean, self.nb_decimals)) + ' / ' + str(round(self.cvar_95_list_std, self.nb_decimals)))



