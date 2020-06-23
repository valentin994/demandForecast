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

PATH = '../train.csv'


if __name__ == '__main__':
    data = pd.read_csv(PATH)
    daily_sale = data.groupby('date')['sales'].sum().to_frame()
    for item in range(1, 51):
        df = data[(data['item'] == item)]
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

        for i in range(1, 15):
            df[f'shift_sales+{i}'] = df['sales'].shift(i)
            labels.append(f'shift_sales+{i}')

        df['daily_store_sales'] = daily_sale['sales'].tolist()
        for i in range(1, 15):
            df[f'daily_store_sales{i}'] = df['daily_store_sales'].shift(i)
            labels.append(f'daily_store_sales{i}')
        df = df.drop('daily_store_sales', axis=1)

        df['mean_store_sales'] = mean_sales['sales'].tolist()
        for i in range(1, 15):
            df[f'mean_store_sales{i}'] = df['mean_store_sales'].shift(i)
            labels.append(f'mean_store_sales{i}')
        df = df.drop('mean_store_sales', axis=1)

        for i in range(1, 15):
            df[f'shift_day+{i}'] = df['dayofweek'].shift(i)
            labels.append(f'shift_day+{i}')
        print(len(labels))
        df = df[15:]
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

        print(X_train.shape)
        model = Sequential()
        model.add(Dense(1156, 'relu'))
        model.add(Dense(500, 'relu'))
        model.add(Dense(250, 'relu'))
        model.add(Dense(125, 'relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(x=X_train, y=y_train, epochs=10)

        loss_df = pd.DataFrame(model.history.history)
        loss_df.plot()
        plt.show()
        test_predictions = model.predict(eval_X)


        print(f'Model fit results:\n'
              f'r2_score {r2_score(eval_y, test_predictions)} \t MSE {mean_squared_error(eval_y, test_predictions)}'
              f'\tEVS {explained_variance_score(eval_y, test_predictions)} \n MAE {mean_absolute_error(eval_y, test_predictions)}'
              f'\tMAD {median_absolute_error(eval_y, test_predictions)}\t ME {max_error(eval_y, test_predictions)}')

        test_results = pd.DataFrame(eval_y, columns=['True Sales'])
        test_results['Predicted Sales'] = test_predictions
        test_results['dayofweek'] = temp['dayofweek'].values
        test_results['month'] = temp['month'].values
        test_results.to_csv(f'./test_results/test_item_{item}_results.csv')