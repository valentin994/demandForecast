import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
# https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/


PATH_LINEAR = "./subplotovi/linear_metrics.csv"
PATH_SVR = "./subplotovi/svr_metrics.csv"
PATH_NEURAL = "./subplotovi/neural_metrics.csv"
PATH_RFR = "./subplotovi/neural_metrics.csv"

PATH_RJESENJA_LINEAR = "./rjesenja_linear/results_15.csv"
PATH_RJESENJA_SVR = "./rjesenja_svr/results_15.csv"
PATH_RJESENJA_NEURAL = "./neural/test_results/test_item_15_results.csv"

df_linear = pd.read_csv(PATH_LINEAR)
df_svr = pd.read_csv(PATH_SVR)
df_neural = pd.read_csv(PATH_NEURAL)

df_linear_rjesenja = pd.read_csv(PATH_RJESENJA_LINEAR)
df_svr_rjesenja = pd.read_csv(PATH_RJESENJA_SVR)
df_neural_rjesenja = pd.read_csv(PATH_RJESENJA_NEURAL)


sns.set_style("darkgrid")

#sns.lineplot(x=df_linear.index, y="r2", data=df_linear)
#plt.title("Linearna regresija")
#plt.xlabel("Proizvod")
#plt.ylabel("R\u00b2")
#plt.savefig("./subplotovi/linear_r2.png")
#plt.clf()
#sns.lineplot(x=df_svr.index, y="r2", data=df_svr)
#plt.xlabel("Proizvod")
#plt.ylabel("R\u00b2")
#plt.title("SVR")
#plt.savefig("./subplotovi/svr_r2.png")
#plt.clf()
#sns.lineplot(x=df_neural.index, y="r2", data=df_neural)
#plt.title("Neuronska mreža")
#plt.xlabel("Proizvod")
#plt.ylabel("R\u00b2")
#plt.savefig("./subplotovi/neural_r2.png")
#plt.clf()

# plt.show()
#


# sns.lineplot(x=df_linear.index, y="evs", data=df_linear)
# plt.title("Linear")
# plt.xlabel("Proizvod")
# plt.ylabel("evs")
# plt.savefig("./subplotovi/linear_evs.png")
# plt.clf()
# sns.lineplot(x=df_svr.index, y="evs", data=df_svr)
# plt.xlabel("Proizvod")
# plt.ylabel("evs")
# plt.title("SVR")
# plt.savefig("./subplotovi/svr_evs.png")
# plt.clf()
# sns.lineplot(x=df_neural.index, y="evs", data=df_neural)
# plt.title("Neural")
# plt.xlabel("Proizvod")
# plt.ylabel("evs")
# plt.savefig("./subplotovi/neural_evs.png")
# plt.clf()

# sns.lineplot(x=df_linear.index, y="mse", data=df_linear)
# plt.title("Linear")
# plt.xlabel("Proizvod")
# plt.ylabel("mse")
# plt.savefig("./subplotovi/linear_mse.png")
# plt.clf()
# sns.lineplot(x=df_svr.index, y="mse", data=df_svr)
# plt.xlabel("Proizvod")
# plt.ylabel("mse")
# plt.title("SVR")
# plt.savefig("./subplotovi/svr_mse.png")
# plt.clf()
# sns.lineplot(x=df_neural.index, y="mse", data=df_neural)
# plt.title("Neural")
# plt.xlabel("Proizvod")
# plt.ylabel("mse")
# plt.savefig("./subplotovi/neural_mse.png")
# plt.clf()

# sns.lineplot(x=df_linear.index, y="mae", data=df_linear)
# plt.title("Linear")
# plt.xlabel("Proizvod")
# plt.ylabel("mae")
# plt.savefig("./subplotovi/linear_mae.png")
# plt.clf()
# sns.lineplot(x=df_svr.index, y="mae", data=df_svr)
# plt.xlabel("Proizvod")
# plt.ylabel("mae")
# plt.title("SVR")
# plt.savefig("./subplotovi/svr_mae.png")
# plt.clf()
# sns.lineplot(x=df_neural.index, y="mae", data=df_neural)
# plt.title("Neural")
# plt.xlabel("Proizvod")
# plt.ylabel("mae")
# plt.savefig("./subplotovi/neural_mae.png")
# plt.clf()

# sns.lineplot(x=df_linear.index, y="mad", data=df_linear)
# plt.title("Linear")
# plt.xlabel("Proizvod")
# plt.ylabel("mad")
# plt.savefig("./subplotovi/linear_mad.png")
# plt.clf()
# sns.lineplot(x=df_svr.index, y="mad", data=df_svr)
# plt.xlabel("Proizvod")
# plt.ylabel("mad")
# plt.title("SVR")
# plt.savefig("./subplotovi/svr_mad.png")
# plt.clf()
# sns.lineplot(x=df_neural.index, y="mad", data=df_neural)
# plt.title("Neural")
# plt.xlabel("Proizvod")
# plt.ylabel("mad")
# plt.savefig("./subplotovi/neural_mad.png")
# plt.clf()

# sns.lineplot(x=df_linear.index, y="me", data=df_linear)
# plt.title("Linear")
# plt.xlabel("Proizvod")
# plt.ylabel("me")
# plt.savefig("./subplotovi/linear_me.png")
# plt.clf()
# sns.lineplot(x=df_svr.index, y="me", data=df_svr)
# plt.xlabel("Proizvod")
# plt.ylabel("me")
# plt.title("SVR")
# plt.savefig("./subplotovi/svr_me.png")
# plt.clf()
# sns.lineplot(x=df_neural.index, y="me", data=df_neural)
# plt.title("Neural")
# plt.xlabel("Proizvod")
# plt.ylabel("me")
# plt.savefig("./subplotovi/neural_me.png")
# plt.clf()

#sns.distplot(df_linear_rjesenja['Predicted Values']-df_linear_rjesenja['True Values'])
#sns.kdeplot(df_linear_rjesenja['Predicted Values']-df_linear_rjesenja['True Values'], shade=True, shade_lowest=False)
#plt.title("Linearna regresija")
#plt.xlabel("Vrijednost")
#plt.ylabel("Gustoća vjerojatnosti")
#plt.savefig("./subplotovi/raspodjela_greske_linear.png")
#plt.clf()
#
#sns.distplot(df_svr_rjesenja['Predicted Values']-df_svr_rjesenja['True Values'])
#sns.kdeplot(df_svr_rjesenja['Predicted Values']-df_svr_rjesenja['True Values'], shade=True, shade_lowest=False)
#plt.title("SVR")
#plt.xlabel("Vrijednost")
#plt.ylabel("Gustoća vjerojatnosti")
#plt.savefig("./subplotovi/raspodjela_greske_svr.png")
#plt.clf()
#
#sns.distplot(df_neural_rjesenja['Predicted Sales']-df_neural_rjesenja['True Sales'])
#sns.kdeplot(df_neural_rjesenja['Predicted Sales']-df_neural_rjesenja['True Sales'], shade=True, shade_lowest=False)
#plt.title("Neuronska mreža")
#plt.xlabel("Vrijednost")
#plt.ylabel("Gustoća vjerojatnosti")
#plt.savefig("./subplotovi/raspodjela_greske_neural.png")
#plt.clf()


#sns.lineplot(x=np.arange(1,366), y=abs(df_neural_rjesenja['True Sales'] - df_neural_rjesenja['Predicted Sales'])/df_neural_rjesenja['True Sales']*100, label='error')
#plt.title("Neuronska mreža")
#plt.xlabel("Dan")
#plt.ylabel("Postotak pogreške")
#plt.savefig(f'./subplotovi/pogreska_u_postotcima_neuronska.png')
#plt.clf()
#
#sns.lineplot(x=np.arange(1,366), y=abs(df_svr_rjesenja['True Values'] - df_svr_rjesenja['Predicted Values'])/df_svr_rjesenja['True Values']*100, label='error')
#plt.title("SVR")
#plt.xlabel("Dan")
#plt.ylabel("Postotak pogreške")
#plt.savefig(f'./subplotovi/pogreska_u_postotcima_svr.png')
#plt.clf()
#
#sns.lineplot(x=np.arange(1,366), y=abs(df_linear_rjesenja['True Values'] - df_linear_rjesenja['Predicted Values'])/df_linear_rjesenja['True Values']*100, label='error')
#plt.title("Linearna regresija")
#plt.xlabel("Dan")
#plt.ylabel("Postotak pogreške")
#plt.savefig(f'./subplotovi/pogreska_u_postotcima_linearna.png')
#plt.clf()

#sns.barplot(x='month', y='True Values', data=df, label='Monthly Sales')
#plt.savefig(f'./subplotovi/mjesečna_prodaja.png')
#plt.clf()

#plt.figure(figsize=(15, 4))
#sns.lineplot(x=df_linear_rjesenja.index, y='Predicted Values', data=df_linear_rjesenja, label='Predviđene vrijednosti')
#sns.lineplot(x=df_linear_rjesenja.index, y='True Values', data=df_linear_rjesenja, label='Prodaja')
#plt.title("Linearna regresija")
#plt.xlabel("Dan")
#plt.ylabel("Prodaja")
#plt.legend(loc=2)
#plt.savefig(f'./subplotovi/value_comparison_linear.png')
#plt.clf()
#
#plt.figure(figsize=(15, 4))
#sns.lineplot(x=df_svr_rjesenja.index, y='Predicted Values', data=df_svr_rjesenja, label='Predviđene vrijednosti')
#sns.lineplot(x=df_svr_rjesenja.index, y='True Values', data=df_svr_rjesenja, label='Prodaja')
#plt.title("SVR")
#plt.xlabel("Dan")
#plt.ylabel("Prodaja")
#plt.legend(loc=2)
#plt.savefig(f'./subplotovi/value_comparison_svr.png')
#plt.clf()
#
#plt.figure(figsize=(15, 4))
#sns.lineplot(x=df_neural_rjesenja.index, y='Predicted Sales', data=df_neural_rjesenja, label='Predviđene vrijednosti')
#sns.lineplot(x=df_neural_rjesenja.index, y='True Sales', data=df_neural_rjesenja, label='Prodaja')
#plt.title("Neuronska mreža")
#plt.xlabel("Dan")
#plt.ylabel("Prodaja")
#plt.legend(loc=2)
#plt.savefig(f'./subplotovi/value_comparison_neural.png')
#plt.clf()

for i in range(1, 13):
    sns.lineplot(y='Predicted Values',
                 x=np.arange(len(df_linear_rjesenja[df_linear_rjesenja['month'] == i])),
                 data=df_linear_rjesenja[df_linear_rjesenja['month'] == i],
                 label='Predviđene vrijednosti')
    sns.lineplot(y='True Values',
                 x=np.arange(len(df_linear_rjesenja[df_linear_rjesenja['month'] == i])),
                 data=df_linear_rjesenja[df_linear_rjesenja['month'] == i], label='Prodaja')
    plt.title(f"{i}. mjesec")
    plt.xlabel("Dan")
    plt.ylabel("Prodaja")
    plt.savefig(f'./subplotovi/monthly/monthly_lin/monthly_sales_{i}.png')
    plt.clf()

for i in range(1, 13):
    sns.lineplot(y='Predicted Values', x=np.arange(len(df_svr_rjesenja[df_svr_rjesenja['month'] == i])), data=df_svr_rjesenja[df_svr_rjesenja['month'] == i],
                 label='Predviđene vrijednosti')
    sns.lineplot(y='True Values', x=np.arange(len(df_svr_rjesenja[df_svr_rjesenja['month'] == i])), data=df_svr_rjesenja[df_svr_rjesenja['month'] == i], label='Prodaja')
    plt.title(f"{i}. mjesec")
    plt.xlabel("Dan")
    plt.ylabel("Prodaja")
    plt.savefig(f'./subplotovi/monthly/monthly_svr/monthly_sales_{i}.png')
    plt.clf()

for i in range(1, 13):
    sns.lineplot(y='Predicted Sales', x=np.arange(len(df_neural_rjesenja[df_neural_rjesenja['month'] == i])), data=df_neural_rjesenja[df_neural_rjesenja['month'] == i],
                 label='Predviđene vrijednosti')
    sns.lineplot(y='True Sales', x=np.arange(len(df_neural_rjesenja[df_neural_rjesenja['month'] == i])), data=df_neural_rjesenja[df_neural_rjesenja['month'] == i], label='Prodaja')
    plt.title(f"{i}. mjesec")
    plt.xlabel("Dan")
    plt.ylabel("Prodaja")
    plt.savefig(f'./subplotovi/monthly/monthly_neural/monthly_sales_{i}.png')
    plt.clf()