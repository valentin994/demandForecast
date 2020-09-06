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
    plt.savefig(f'./plots_neural/metrika_{metrika_name}.png')
    plt.clf()


if __name__ == '__main__':
    r2 = []
    mse = []
    evs = []
    mae = []
    mad = []
    me = []

    sns.set_style('darkgrid')

    for i in range(1, 51):
        df = pd.read_csv(f'./test_results/test_item_{i}_results.csv')

        predicted_values = df['Predicted Sales']
        true_values = df['True Sales']

        r2.append(r2_score(true_values, predicted_values))
        mse.append(mean_squared_error(true_values, predicted_values))
        evs.append(explained_variance_score(true_values, predicted_values))
        mae.append(mean_absolute_error(true_values, predicted_values))
        mad.append(median_absolute_error(true_values, predicted_values))
        me.append(max_error(true_values, predicted_values))

        #   Predvidanja plotana uz prave vrijednosti
      #  plt.figure(figsize=(15, 4))
      #  sns.lineplot(x=df.index, y='Predicted Sales', data=df, label='Predictions')
      #  sns.lineplot(x=df.index, y='True Sales', data=df, label='Sales')
      #  plt.legend(loc=2)
      #  plt.savefig(f'./plots_neural/item{i}/value_comparison.png')
      #  plt.clf()

        #   Raspodjela prodaje po tjednu i vrijednosti
       # sns.jointplot(x='dayofweek', y='True Sales', data=df)
       # plt.savefig(f'./plots_neural/item{i}/value_weekly_sales_spread_points.png')
       # plt.clf()
#
       # #   Tjedna usporedba
       # sns.lineplot(x='dayofweek', y='True Sales', data=df, label='sales')
        #sns.lineplot(x='dayofweek', y='Predicted Sales', data=df, label='predictions')
        #plt.legend(loc=2)
        #plt.savefig(f'./plots_neural/item{i}/weekly_comparison.png')
 #       plt.clf()
#
        ##   Error
       # sns.distplot(df['Predicted Sales'] - df['True Sales'])
       # sns.kdeplot(df['Predicted Sales'] - df['True Sales'], shade=True, shade_lowest=False)
       # plt.savefig(f'./plots_neural/item{i}/error_swarm.png')
       # plt.clf()

        #   Error u postocima
        sns.lineplot(x=np.arange(1,366), y=abs(df['True Sales'] - df['Predicted Sales'])/df['True Sales']*100, label='error')
        plt.savefig(f'./plots_neural/item{i}/error_percentage.png')
        plt.clf()

        #   Mjesecno ponasanje
       # sns.barplot(x='month', y='True Sales', data=df, label='Monthly Sales')
       # plt.savefig(f'./plots_neural/item{i}/monthly_sales.png')
       # plt.clf()

        #   Usporedba svih mjeseci
        #for j in range(1, 13):
        #    sns.lineplot(y='True Sales', x=np.arange(len(df[df['month'] == j])), data=df[df['month'] == j],
        #                 label='Sales')
        #    sns.lineplot(y='Predicted Sales', x=np.arange(len(df[df['month'] == j])), data=df[df['month'] == j],
        #                 label='Predictions')
        #    plt.savefig(f'./plots_neural/item{i}/mothly_comparison/monthly_sales_{j}.png')
        #    plt.clf()

    plot(r2, 'r2')
    plot(mse, 'mse')
    plot(evs, 'evs')
    plot(mae, 'mae')
    plot(mad, 'mad')
    plot(me, 'me')