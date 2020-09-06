import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, \
    explained_variance_score, max_error

PATH = './train.csv'

if __name__ == '__main__':
    df = pd.read_csv(PATH)
    df['date'] = pd.to_datetime(df['date'])

    df = df[df['item'] == 15]
    df = df.groupby('date')['sales'].sum().to_frame().reset_index()
    df['year'] = df.date.dt.year
    print(df)

    #plt.figure(figsize=(40, 25))
    #sns.lineplot(y=df[df['year'] == 2013]['sales'], x=df[df['year'] == 2013]['date'], label='2013')
    #sns.lineplot(y=df[df['year'] == 2014]['sales'], x=df[df['year'] == 2014]['date'], label='2014')
    #sns.lineplot(y=df[df['year'] == 2015]['sales'], x=df[df['year'] == 2015]['date'], label='2015')
    #sns.lineplot(y=df[df['year'] == 2016]['sales'], x=df[df['year'] == 2016]['date'], label='2016')
    #sns.lineplot(y=df[df['year'] == 2017]['sales'], x=df[df['year'] == 2017]['date'], label='2017')
    #plt.title('Dnevna prodaja proizvoda 15', fontsize=40)
    #plt.xlabel('Datum', fontsize=40)
    #plt.ylabel('Prodaja', fontsize=40)
    #plt.xticks(fontsize=25)
    #plt.yticks(fontsize=25)
    #plt.legend(loc=2, fontsize=30)
    #plt.savefig(f'./neko.png')
    #plt.clf()

    #plt.figure(figsize=(50, 20))
    #sns.lineplot(y=df['sales'], x=df['date'])
    #plt.savefig(f'./

    #df_time_linear = pd.read_csv("./plots_linear/time_metrics/r2_time_metrics.csv")
    #sns.lineplot(x="time", y="r2_score", data=df_time_linear)
    #plt.title("Kretanje R\u00b2 metrike za linearnu regresiju")
    #plt.xlabel("Vrijeme u sekundama")
    #plt.ylabel("R\u00b2 vrijednost")
    #plt.savefig('./subplotovi/New folder/r2_linear_time_metrics.png')
    #plt.clf()
#
    #df_time_svr = pd.read_csv("./plots_svr/time_metrics/r2_time_metrics.csv")
    #sns.lineplot(x="time", y="r2_score", data=df_time_svr)
    #plt.title("Kretanje R\u00b2 metrike za SVR")
    #plt.xlabel("Vrijeme u sekundama")
    #plt.ylabel("R\u00b2 vrijednost")
    #plt.savefig('./subplotovi/New folder/r2_svr_time_metrics.png')
    #plt.clf()
#
    df_time_neural = pd.read_csv("./neural/time/time.csv")
    sns.lineplot(x="time", y="r2", data=df_time_neural)
    plt.title("Kretanje R\u00b2 metrike za neuronsku mrežu")
    plt.xlabel("Vrijeme u sekundama")
    plt.ylabel("R\u00b2 vrijednost")
    plt.savefig('./subplotovi/New folder/r2_neural_time_metrics.png')
    plt.clf()
#