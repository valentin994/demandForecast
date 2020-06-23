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

    df['date'] = pd.to_datetime(df['date'])
    temp = df
    daily_sale = df.groupby('date')['sales'].sum().to_frame()
    mean_sales = df.groupby('date')['sales'].mean().to_frame()


    for itemNumber in range(14, 15):
        df = temp[temp['item'] == itemNumber]
        df = df.groupby('date')['sales'].sum().to_frame().reset_index()
        df['year'] = df.date.dt.year
        df['dayofmonth'] = df.date.dt.day
        df['dayofweek'] = df.date.dt.dayofweek
        df['month'] = df.date.dt.month
        df = df.drop('date', axis=1)

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

        #df['week_sale'] = week_sale(df['sales'].tolist())
        #df['week_avg'] = week_avg(df['sales'].tolist())
        #df['monthly_sales'] = monthly_sales(df['sales'].tolist())
        #df['two_day_sale_avg'] = two_day_avg(df['sales'].tolist())
        #df['three_day_sale_avg'] = three_day_avg(df['sales'].tolist())
        #df['four_day_sale_avg'] = four_day_avg(df['sales'].tolist())
        #for i in range(1, 8):
        #    df[f'three_day_sale_avg_shift{i}'] = df['three_day_sale_avg'].shift(i)
        #    df[f'four_day_sale_avg_shift{i}'] = df['four_day_sale_avg'].shift(i)
        #    df[f'two_day_sale_avg_shift{i}'] = df['two_day_sale_avg'].shift(i)
        #    df[f'week_sale{i}'] = df['week_sale'].shift(i)
        #    df[f'week_avg{i}'] = df['week_avg'].shift(i)
        #    df[f'monthly_sales{i}'] = df['monthly_sales'].shift(i)
        test = df[df['year'] > 2016]
        df = df[(df['year'] < 2017)]
        df = df.iloc[30:]
        df = df.drop('year', axis=1)
        test = test.drop('year', axis=1)
        print(df.columns)

        #   Model

        X = df.drop('sales', axis=1)
        y = df['sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #   SVR MODEL
        svr = svm.SVR(C=0.001, kernel='linear', degree=8, gamma='scale', coef0=10, verbose=3)
        svr.fit(X_train, y_train)
        predictions = svr.predict(X_test)
        predictions = svr.predict(test.drop('sales', axis=1))
        y_test = test['sales'].tolist()

        #   LINEAR REGRESSION

        #lin_reg = LinearRegression()
        #lin_reg.fit(X_train, y_train)
        #predictions = lin_reg.predict(test.drop('sales', axis=1))
        #y_test = test['sales'].tolist()

        results = pd.DataFrame()
        results['Predicted Values'] = predictions
        results['True Values'] = y_test
        results['dayofmonth'] = test['dayofmonth'].tolist()
        results['dayofweek'] = test['dayofweek'].tolist()
        results['month'] = test['month'].tolist()

        #results.to_csv(f'./rjesenja_linear/results_{itemNumber}.csv', index=False)


        #   Evaluation

        print(f'Model fit results:\n'
              f'r2_score {r2_score(y_test, predictions)} \t MSE {mean_squared_error(y_test, predictions)}'
              f'\tEVS {explained_variance_score(y_test, predictions)} \n MAE {mean_absolute_error(y_test, predictions)}'
              f'\tMAD {median_absolute_error(y_test, predictions)}\t ME {max_error(y_test, predictions)}')
