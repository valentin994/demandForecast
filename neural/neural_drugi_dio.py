import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, \
    explained_variance_score, max_error
import time

PATH = '../train.csv'

executionTime = []
r2 = []
if __name__ == '__main__':
    data = pd.read_csv(PATH)
    daily_sale = data.groupby('date')['sales'].sum().to_frame()
    for step in range(1, 31):
        df = data[(data['item'] == step)]
        df = df.groupby('date')['sales'].sum().to_frame().reset_index()
        df['daily_store_sales'] = daily_sale['sales'].tolist()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df.date.dt.year

        df['dayofmonth'] = df.date.dt.day
        df['dayofweek'] = df.date.dt.dayofweek
        df['month'] = df.date.dt.month

        mean_sales = df.groupby('date')['sales'].mean().to_frame()
        labels = []
        labels.append('dayofmonth')
        labels.append('dayofweek')
        labels.append('month')

        for i in range(1, step):
            df[f'shift_sales+{i}'] = df['sales'].shift(i)
            labels.append(f'shift_sales+{i}')

        df['daily_store_sales'] = daily_sale['sales'].tolist()
        for i in range(1, step):
            df[f'daily_store_sales{i}'] = df['daily_store_sales'].shift(i)
            labels.append(f'daily_store_sales{i}')
        df = df.drop('daily_store_sales', axis=1)

        df['mean_store_sales'] = mean_sales['sales'].tolist()
        for i in range(1, step):
            df[f'mean_store_sales{i}'] = df['mean_store_sales'].shift(i)
            labels.append(f'mean_store_sales{i}')
        df = df.drop('mean_store_sales', axis=1)

        for i in range(1, step):
            df[f'shift_day+{i}'] = df['dayofweek'].shift(i)
            labels.append(f'shift_day+{i}')

        df = df[step:]
        temp = df[df['year'] == 2017]
        df = df[df['year'] < 2017]
        X = df[labels].values
        y = df['sales'].values
        eval_X = temp[labels].values
        eval_y = temp['sales'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        eval_X = scaler.transform(eval_X)

        start = time.time()
        model = Sequential()
        model.add(Dense(1156, 'relu'))
        model.add(Dense(500, 'relu'))
        model.add(Dense(250, 'relu'))
        model.add(Dense(125, 'relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(x=X_train, y=y_train, epochs=10)
        end = time.time()
        executionTime.append(end - start)
        loss_df = pd.DataFrame(model.history.history)
        loss_df.plot()
        test_predictions = model.predict(eval_X)

        r2.append(r2_score(eval_y, test_predictions))
        plt.clf()
        print("THIS IS STEP ", step)
    time_data = pd.DataFrame()
    time_data['r2'] = r2
    time_data["time"] = executionTime
    time_data.to_csv("./time/time.csv")
