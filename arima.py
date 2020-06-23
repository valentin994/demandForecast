import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from numpy import log
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, \
    explained_variance_score, max_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time

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
    executionTime = []
    r = []
    evs = []
    df['date'] = pd.to_datetime(df['date'])
    temp = df
    daily_sale = df.groupby('date')['sales'].sum().to_frame()
    for timeWindow in range(1, 30):
        df = temp[temp['item'] == 15]
        df = df.groupby('date')['sales'].sum().to_frame().reset_index()
        df['year'] = df.date.dt.year
        df['dayofmonth'] = df.date.dt.day
        df['dayofweek'] = df.date.dt.dayofweek
        df['month'] = df.date.dt.month
        df = df.drop('date', axis=1)

        #   Parametri

        for i in range(1, timeWindow):
            df[f'shift_sales+{i}'] = df['sales'].shift(i)

        df['daily_store_sales'] = daily_sale['sales'].tolist()
        for i in range(1, timeWindow):
            df[f'daily_store_sales{i}'] = df['daily_store_sales'].shift(i)
        df = df.drop('daily_store_sales', axis=1)

        for i in range(1, timeWindow):
            df[f'shift_day+{i}'] = df['dayofweek'].shift(i)

    test = df[df['year'] > 2016]
    df = df[(df['year'] < 2017)]
    df = df.iloc[timeWindow:]
    df = df.drop('year', axis=1)
    test = test.drop('year', axis=1)

    #   Model
    X = df.drop('sales', axis=1)
    y = df['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #SARIMAX MODEL
    print(df)
    result = adfuller(df['sales'].tolist())
    print('ADF Stat: %f' % result[0])
    print('p-val: %f' % result[1])




        #   SVR MODEL
        #svr = svm.SVR(C=0.001, kernel='linear', degree=8, gamma='scale', coef0=10, verbose=3)
        #svr.fit(X_train, y_train)
        #
        #predictions = svr.predict(X_test)
        #predictions = svr.predict(test.drop('sales', axis=1))

        #   LINEAR REGRESSION
        #start = time.time()
        #lin_reg = LinearRegression()
        #lin_reg.fit(X_train, y_train)
        #predictions = lin_reg.predict(test.drop('sales', axis=1))
        #end = time.time()
        #y_test = test['sales'].tolist()
        #r.append(r2_score(y_test, predictions))
        #evs.append((explained_variance_score(y_test, predictions)))
        #executionTime.append(end-start)
    #results = pd.DataFrame()
    #results['time'] = executionTime
    #results['r2_score'] = r
    #results['evs'] = evs
    #print(results)
    #results.to_csv('./plots_linear/time_metrics/r2_time_metrics.csv', index=False)




