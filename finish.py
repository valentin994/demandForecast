import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, \
    explained_variance_score, max_error

#Metrike crtat

def plot(metrika_values, metrika_name):
    sns.lineplot(x=np.arange(1, 51), y=metrika_values)
    plt.savefig(f'./plots_linear/metrike/metrika_{metrika_name}.png')
    plt.clf()

def plot_metrics_vs_time_win(df, metric):
    sns.lineplot(x=np.arange(1, 30), y=metric, data=df)
    plt.savefig(f'./plots_linear/time_metrics/{metric}_vs_time_win.png')
    plt.clf()

def plot_time_vs_time_win(df):
    sns.lineplot(x='time', y=np.arange(1, 30), data=df)
    plt.savefig(f'./plots_linear/time_metrics/time_vs_time_win.png')
    plt.clf()

def plot_metric_vs_time(df, metric):
    sns.lineplot(x='time', y=metric, data=df)
    plt.savefig(f'./plots_linear/time_metrics/{metric}_vs_time.png')
    plt.clf()

if __name__ == '__main__':

    #r2 = []
    #mse = []
    #evs = []
    #mae = []
    #mad = []
    #me = []
#
    #for itemNumber in range(1, 51):
    #    df = pd.read_csv(f'./rjesenja_linear/results_{itemNumber}.csv')
    #    sns.set_style('darkgrid')
#
    #    predicted_values = df['Predicted Values']
    #    true_values = df['True Values']
#
    #    r2.append(r2_score(true_values, predicted_values))
    #    mse.append(mean_squared_error(true_values, predicted_values))
    #    evs.append(explained_variance_score(true_values, predicted_values))
    #    mae.append(mean_absolute_error(true_values, predicted_values))
    #    mad.append(median_absolute_error(true_values, predicted_values))
    #    me.append(max_error(true_values, predicted_values))
#
    #print(max(r2))
    #print(r2.index(max(r2)) + 1)
#
    #print(max(evs))
    #print(evs.index(max(evs)) + 1)
#
    #print(max(mse))
    #print(mse.index(min(mse)) + 1)
#
    #print(max(mae))
    #print(mae.index(min(mae)) + 1)
#
    #print(max(mad))
    #print(mad.index(min(mad)) + 1)
#
    #print(max(me))
    #print(me.index(min(me)) +1)
#
#
    #plot(r2, 'r2')
    #plot(mse, 'mse')
    #plot(evs, 'evs')
    #plot(mae, 'mae')
    #plot(mad, 'mad')
    #plot(me, 'me')

    #   Najbolji itemi 15, 5, 1, 41


    df = pd.read_csv('./plots_linear/time_metrics/mse_mae_time_metrics.csv')
    plot_metrics_vs_time_win(df, 'mse')
    plot_metrics_vs_time_win(df, 'mae')
    #plot_time_vs_time_win(df)
    plot_metric_vs_time(df, 'mae')
    plot_metric_vs_time(df, 'mse')


    df = pd.read_csv('./plots_linear/time_metrics/mad_me_time_metrics.csv')
    plot_metrics_vs_time_win(df, 'mad')
    plot_metrics_vs_time_win(df, 'me')
    plot_metric_vs_time(df, 'me')
    plot_metric_vs_time(df, 'mad')