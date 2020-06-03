import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt




if __name__ == '__main__':
    for itemNumber in range(2,51):
        df = pd.read_csv(f'./rjesenja/results_{itemNumber}.csv')
        sns.set_style('darkgrid')

        #   Predvidanja plotana uz prave vrijednosti
        plt.figure(figsize=(15, 4))
        sns.lineplot(x=df.index, y='Predicted Values', data=df, label='Predictions')
        sns.lineplot(x=df.index, y='True Values', data=df, label='Sales')
        plt.legend(loc=2)
        plt.savefig(f'./plots/item{itemNumber}/value_comparison.png')
        plt.clf()

        #   Raspodjela prodaje po tjednu i vrijednosti
        sns.jointplot(x='dayofweek', y='True Values', data=df)
        plt.savefig(f'./plots/item{itemNumber}/value_weekly_sales_spread_points.png')
        plt.clf()

        #   Tjedna usporedba
        sns.lineplot(x='dayofweek', y='True Values', data=df, label='sales')
        sns.lineplot(x='dayofweek', y='Predicted Values', data=df, label='predictions')
        plt.legend(loc=2)
        plt.savefig(f'./plots/item{itemNumber}/weekly_comparison.png')
        plt.clf()

        #   Error
        sns.distplot(df['Predicted Values']-df['True Values'])
        sns.kdeplot(df['Predicted Values']-df['True Values'], shade=True, shade_lowest=False)
        plt.savefig(f'./plots/item{itemNumber}/error_swarm.png')
        plt.clf()

        #   Error u postocima
        sns.lineplot(x=np.arange(1,366), y=df['True Values']/df['Predicted Values'], label='error')
        plt.savefig(f'./plots/item{itemNumber}/error_percentage.png')
        plt.clf()

    #   Mjesecno ponasanje
        sns.barplot(x='month', y='True Values', data=df, label='Monthly Sales')
        plt.savefig(f'./plots/item{itemNumber}/monthly_sales.png')
        plt.clf()

        #   Usporedba svih mjeseci
        for i in range(1, 13):
            sns.lineplot(y='True Values', x=np.arange(len(df[df['month'] == i])), data=df[df['month'] == i], label='Sales')
            sns.lineplot(y='Predicted Values', x=np.arange(len(df[df['month'] == i])), data=df[df['month'] == i], label='Predictions')
            plt.savefig(f'./plots/item{itemNumber}/monthly_comparison/monthly_sales_{i}.png')
            plt.clf()