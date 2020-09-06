import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, \
    explained_variance_score, max_error

#Metrike crtat
if __name__ == '__main__':

    r2_linear = []
    mse_linear = []
    evs_linear = []
    mae_linear = []
    mad_linear = []
    me_linear = []
    r2_svr = []
    mse_svr = []
    evs_svr = []
    mae_svr = []
    mad_svr = []
    me_svr = []
    r2_neural = []
    mse_neural = []
    evs_neural = []
    mae_neural = []
    mad_neural = []
    me_neural = []

    for itemNumber in range(1, 51):
        sns.set_style('darkgrid')


        ### Collect dataframes for every method##
        #df_linear = pd.read_csv(f'./rjesenja_linear/results_{itemNumber}.csv')
        #df_svr = pd.read_csv(f'./rjesenja_svr/results_{itemNumber}.csv')
        #df_neural = pd.read_csv(f'./neural/test_results/test_item_{itemNumber}_results.csv')
        df_rfr = pd.read_csv(f'./rjesenja_rf/results_{itemNumber}.csv')

        #predicted_values_linear = df_linear['Predicted Values']
        #predicted_values_svr = df_svr['Predicted Values']
        #predicted_values_neural = df_neural['Predicted Sales']
        predicted_values_rfr = df_rfr['Predicted Values']

        #true_values_linear = df_linear['True Values']
        #true_values_svr = df_svr['True Values']
        #true_values_neural = df_neural['True Sales']
        true_values_rfr = df_rfr['True Values']

        #r2_linear.append(r2_score(true_values_linear, predicted_values_linear))
        #mse_linear.append(mean_squared_error(true_values_linear, predicted_values_linear))
        #evs_linear.append(explained_variance_score(true_values_linear, predicted_values_linear))
        #mae_linear.append(mean_absolute_error(true_values_linear, predicted_values_linear))
        #mad_linear.append(median_absolute_error(true_values_linear, predicted_values_linear))
        #me_linear.append(max_error(true_values_linear, predicted_values_linear))

        #r2_svr.append(r2_score(true_values_svr, predicted_values_svr))
        #mse_svr.append(mean_squared_error(true_values_svr, predicted_values_svr))
        #evs_svr.append(explained_variance_score(true_values_svr, predicted_values_svr))
        #mae_svr.append(mean_absolute_error(true_values_svr, predicted_values_svr))
        #mad_svr.append(median_absolute_error(true_values_svr, predicted_values_svr))
        #me_svr.append(max_error(true_values_svr, predicted_values_svr))

        #r2_neural.append(r2_score(true_values_neural, predicted_values_neural))
        #mse_neural.append(mean_squared_error(true_values_neural, predicted_values_neural))
        #evs_neural.append(explained_variance_score(true_values_neural, predicted_values_neural))
        #mae_neural.append(mean_absolute_error(true_values_neural, predicted_values_neural))
        #mad_neural.append(median_absolute_error(true_values_neural, predicted_values_neural))
        #me_neural.append(max_error(true_values_neural, predicted_values_neural))

        r2_rfr.append(r2_score(true_values_rfr, predicted_values_rfr))
        mse_rfr.append(mean_squared_error(true_values_rfr, predicted_values_rfr))
        evs_rfr.append(explained_variance_score(true_values_rfr, predicted_values_rfr))
        mae_rfr.append(mean_absolute_error(true_values_rfr, predicted_values_rfr))
        mad_rfr.append(median_absolute_error(true_values_rfr, predicted_values_rfr))
        me_rfr.append(max_error(true_values_rfr, predicted_values_rfr))

    #linear_metrics = pd.DataFrame(columns=['r2', 'mse', 'evs', 'mae', 'mad', 'me'])
    #svr_metrics = pd.DataFrame(columns=['r2', 'mse', 'evs', 'mae', 'mad', 'me'])
    #neural_metrics = pd.DataFrame(columns=['r2', 'mse', 'evs', 'mae', 'mad', 'me'])
    print(r2_linear)
    rfr_metrics = pd.DataFrame(columns=['r2', 'mse', 'evs', 'mae', 'mad', 'me'])



    #linear_metrics['r2'] = r2_linear
    #linear_metrics['mse'] = mse_linear
    #linear_metrics['evs'] = evs_linear
    #linear_metrics['mae'] = mae_linear
    #linear_metrics['mad'] = mad_linear
    #linear_metrics['me'] = me_linear

    #svr_metrics['r2'] = r2_svr
    #svr_metrics['mse'] = mse_svr
    #svr_metrics['evs'] = evs_svr
    #svr_metrics['mae'] = mae_svr
    #svr_metrics['mad'] = mad_svr
    #svr_metrics['me'] = me_svr
#
    #neural_metrics['r2'] = r2_neural
    #neural_metrics['mse'] = mse_neural
    #neural_metrics['evs'] = evs_neural
    #neural_metrics['mae'] = mae_neural
    #neural_metrics['mad'] = mad_neural
    #neural_metrics['me'] = me_neural

    rfr_metrics['r2'] = r2_rfr
    rfr_metrics['mse'] = mse_rfr
    rfr_metrics['evs'] = evs_rfr
    rfr_metrics['mae'] = mae_rfr
    rfr_metrics['mad'] = mad_rfr
    rfr_metrics['me'] = me_rfr

    #linear_metrics.to_csv("./subplotovi/linear_metrics.csv")
    #svr_metrics.to_csv("./subplotovi/svr_metrics.csv")
    #neural_metrics.to_csv("./subplotovi/neural_metrics.csv")
    rfr_metrics.to_csv("./subplotovi/rfr_metrics.csv")
