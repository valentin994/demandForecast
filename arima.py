import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, \
    explained_variance_score, max_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import statsmodels.api as sm

PATH = './train.csv'


#   Metode

def week_sale(sales):
    weekly_sale = []
    sum = 0
    for i in range(7):
        weekly_sale.append(0)
    for i in range((len(sales) - 7)):
        for j in range(i, i + 7):
            sum += sales[j]
        weekly_sale.append(sum)
        sum = 0
    return weekly_sale


def week_avg(sales):
    weekly_avg = []
    sum = 0
    for i in range(7):
        weekly_avg.append(0)
    for i in range((len(sales) - 7)):
        for j in range(i, i + 7):
            sum += sales[j]
        weekly_avg.append(sum / 7)
        sum = 0
    return weekly_avg


def three_day_avg(sales):
    sum = 0
    day_avg = []

    for i in range(2, -1, -1):
        for j in range(i):
            sum += sales[j]
        if (i != 0):
            day_avg.append(sum / i)
        sum = 0
    day_avg.append(sales[0])
    day_avg.reverse()

    for row in range(len(sales) - 3):
        for row in range(row, row + 3):
            sum += sales[row]
        day_avg.append(sum / 3)
        sum = 0
    return day_avg


def two_day_avg(sales):
    sum = 0
    day_avg = []

    for i in range(1, -1, -1):
        for j in range(i):
            sum += sales[j]
        if (i != 0):
            day_avg.append(sum / i)
        sum = 0
    day_avg.append(sales[0])
    day_avg.reverse()

    for row in range(len(sales) - 2):
        for row in range(row, row + 2):
            sum += sales[row]
        day_avg.append(sum / 2)
        sum = 0
    return day_avg


def four_day_avg(sales):
    sum = 0
    day_avg = []

    for i in range(3, -1, -1):
        for j in range(i):
            sum += sales[j]
        if (i != 0):
            day_avg.append(sum / i)
        sum = 0
    day_avg.append(sales[0])
    day_avg.reverse()
    for row in range(len(sales) - 4):
        for row in range(row, row + 4):
            sum += sales[row]
        day_avg.append(sum / 4)
        sum = 0
    return day_avg


def monthly_sales(sales):
    sum = 0
    day_avg = []

    for i in range(29, -1, -1):
        for j in range(i):
            sum += sales[j]
        if (i != 0):
            day_avg.append(sum / i)
        sum = 0
    day_avg.append(sales[0])
    day_avg.reverse()

    for row in range(len(sales) - 30):
        for row in range(row, row + 30):
            sum += sales[row]
        day_avg.append(sum)
        sum = 0
    return day_avg


if __name__ == '__main__':
    df = pd.read_csv(PATH)
    #   Data preparation
    #
    #   Filtracija, grupiranje i dodavanje datuma za item trgovackog lanca
    #

    temp = df
    df['date'] = pd.to_datetime(df['date'])
    daily_sale = df.groupby('date')['sales'].sum().to_frame()
    mean_sales = df.groupby('date')['sales'].mean().to_frame()

    for itemNumber in range(15, 16):
        df = temp[temp['item'] == itemNumber]
        df = df.groupby('date')['sales'].sum().to_frame().reset_index()
        df['year'] = df.date.dt.year
        df['dayofmonth'] = df.date.dt.day
        df['dayofweek'] = df.date.dt.dayofweek
        df['month'] = df.date.dt.month

        #   Parametri

        for i in range(1, 15):
           df[f'shift_sales+{i}'] = df['sales'].shift(i)

        df['daily_store_sales'] = daily_sale['sales'].tolist()
        for i in range(1, 31):
           df[f'daily_store_sales{i}'] = df['daily_store_sales'].shift(i)
        df = df.drop('daily_store_sales', axis=1)

        df['mean_store_sales'] = mean_sales['sales'].tolist()
        for i in range(1, 31):
           df[f'mean_store_sales{i}'] = df['mean_store_sales'].shift(i)
        df = df.drop('mean_store_sales', axis=1)

        for i in range(1, 31):
           df[f'shift_day+{i}'] = df['dayofweek'].shift(i)

        df['week_sale'] = week_sale(df['sales'].tolist())
        df['week_avg'] = week_avg(df['sales'].tolist())
        df['monthly_sales'] = monthly_sales(df['sales'].tolist())
        df['two_day_sale_avg'] = two_day_avg(df['sales'].tolist())
        df['three_day_sale_avg'] = three_day_avg(df['sales'].tolist())
        df['four_day_sale_avg'] = four_day_avg(df['sales'].tolist())
        for i in range(1, 8):
           df[f'three_day_sale_avg_shift{i}'] = df['three_day_sale_avg'].shift(i)
           df[f'four_day_sale_avg_shift{i}'] = df['four_day_sale_avg'].shift(i)
           df[f'two_day_sale_avg_shift{i}'] = df['two_day_sale_avg'].shift(i)
           df[f'week_sale{i}'] = df['week_sale'].shift(i)
           df[f'week_avg{i}'] = df['week_avg'].shift(i)
           df[f'monthly_sales{i}'] = df['monthly_sales'].shift(i)

        tes = df[df['year'] > 2016]
        tra = df[(df['year'] < 2017)].drop('year', axis=1)
        tra = tra[30:]
        tra = tra.set_index('date')
        tes = tes.set_index('date')
        df = df.drop('year', axis=1)
        #   Model
        exog_train = tra.drop('sales', axis=1)
        tra = tra['sales']
        print(tra)
        print(exog_train)

        #resDiff = sm.tsa.arma_order_select_ic(tra, max_ar=7, max_ma=7, ic='aic', trend='c')
        #print('ARMA(p,q) =', resDiff['aic_min_order'], 'is the best.')

        #sarimax = sm.tsa.statespace.SARIMAX(tra, order=(6, 1, 1), seasonal_order=(1,1,1,7),
        #                                    enforce_stationarity=False, enforce_invertibility=False, freq='D').fit()

        sarima = sm.tsa.statespace.SARIMAX(tra, trend='n', freq='D', enforce_invertibility=False,
                                           order=(6, 1, 1), seasonal_order=(1, 1, 1, 7))

        results = sarima.fit()
        print(results.summary())
        #res = sarimax.resid

        pred = results.predict('2017-01-01', '2017-12-31', dynamic=True)
        print('ARIMA model MSE:{}'.format(explained_variance_score(tes['sales'], pred)))


        #   Evaluation

        # print(f'Model fit results:\n'
        #      f'r2_score {r2_score(y_test, predictions)} \t MSE {mean_squared_error(y_test, predictions)}'
        #      f'\tEVS {explained_variance_score(y_test, predictions)} \n MAE {mean_absolute_error(y_test, predictions)}'
        #      f'\tMAD {median_absolute_error(y_test, predictions)}\t ME {max_error(y_test, predictions)}')
