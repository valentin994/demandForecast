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

PATH = './train.csv'


if __name__ == '__main__':
    df = pd.read_csv(PATH)
    #   Data preparation
    #
    #   Filtracija, grupiranje i dodavanje datuma za item trgovackog lanca
    #
    executionTime = []
    mse = []
    mae = []
    df['date'] = pd.to_datetime(df['date'])
    temp = df
    daily_sale = df.groupby('date')['sales'].sum().to_frame()
    for timeWindow in range(1, 30):
        df = temp[temp['item'] == 5]
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

        #   SVR MODEL
        #svr = svm.SVR(C=0.001, kernel='linear', degree=8, gamma='scale', coef0=10, verbose=3)
        #svr.fit(X_train, y_train)
        #
        #predictions = svr.predict(X_test)
        #predictions = svr.predict(test.drop('sales', axis=1))

        #   LINEAR REGRESSION
        start = time.time()
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        predictions = lin_reg.predict(test.drop('sales', axis=1))
        end = time.time()
        y_test = test['sales'].tolist()
        mse.append(mean_squared_error(y_test, predictions))
        mae.append((mean_absolute_error(y_test, predictions)))
        executionTime.append(end-start)
    results = pd.DataFrame()
    results['time'] = executionTime
    results['mse'] = mse
    results['mae'] = mae
    print(results)
    results.to_csv('./plots_linear/time_metrics/mse_mae_time_metrics.csv', index=False)




