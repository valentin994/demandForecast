import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

PATH = './train.csv'

if __name__ == '__main__':
    df = pd.read_csv(PATH)
    df['date'] = pd.to_datetime(df['date'])

    df = df[df['item'] == 1]
    df = df.groupby('date')['sales'].sum().to_frame().reset_index()
    df['year'] = df.date.dt.year
    print(df)

    plt.figure(figsize=(40, 25))
    sns.lineplot(y=df[df['year'] == 2013]['sales'], x=df[df['year'] == 2013]['date'])
    sns.lineplot(y=df[df['year'] == 2014]['sales'], x=df[df['year'] == 2014]['date'])
    sns.lineplot(y=df[df['year'] == 2015]['sales'], x=df[df['year'] == 2015]['date'])
    sns.lineplot(y=df[df['year'] == 2016]['sales'], x=df[df['year'] == 2016]['date'])
    sns.lineplot(y=df[df['year'] == 2017]['sales'], x=df[df['year'] == 2017]['date'])
    plt.title('Prodaja proizvoda 1', fontsize=40)
    plt.xlabel('Datum', fontsize=40)
    plt.ylabel('Prodaja', fontsize=40)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(f'./neko.png')
    plt.clf()

    plt.figure(figsize=(50, 20))
    sns.lineplot(y=df['sales'], x=df['date'])
    plt.savefig(f'./neko_ime.png')