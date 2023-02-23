import os
import pandas as pd
import numpy as np
import random
import seaborn as sns
from scipy.stats import linregress
from sklearn import datasets
import matplotlib
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import scipy.stats as stats
import math
from sklearn.linear_model import LinearRegression
import itertools
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
import json
from scipy import stats
from scipy.stats import ttest_rel

def plot_linear_regression_line(X, Y):
    X = X.values.reshape(-1, 1)
    y = Y.values.reshape(-1, 1)
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    line_X = np.arange(X.min(), X.max() + 1)[:, np.newaxis]
    line_y = lr.predict(line_X)

    params = np.append(lr.intercept_, lr.coef_)
    predictions = lr.predict(X)
    newX = np.append(np.ones((len(X), 1)), X, axis=1)
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    return line_X, line_y, p_values, params

_Uncert_Error_path = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyDistances/GT_Predicted_Points_distance_UNCERT_R3_20_100_40_MIN/Error VS Uncertainty.csv'
_TGT = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/Analysis'
_Uncert_Error = pd.read_csv(_Uncert_Error_path, header=0, index_col=0)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
sns.kdeplot(
    data=_Uncert_Error, x="Error ; mm", y="Uncertainty", hue="vessel/side",alpha = 0.8, fill=False,legend=True)
plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='30')
print(_Uncert_Error['vessel/side'].unique())
plt.ylabel('Uncertainty',fontsize=45)
plt.ylim(0, 0.03)
plt.xlabel(' Error of GT-Auto Minimum Distance ; mm', fontsize=45)
plt.title(' Error of GT-Auto Minimum Distance \n with Uncertainty', fontsize=45)
# plt.xlabel('Organs', fontsize=30)
# ax.legend(loc='upper left',fontsize=30)
plt.xticks(fontsize=40)
plt.yticks(fontsize=35)

# from matplotlib.colors import to_hex
#
# cmap = plt.get_cmap('viridis')
# colors = cmap(np.linspace(0, 1, len(set(_Uncert_Error['vessel/side']))))
# hex_colors = [to_hex(color) for color in colors]
# colors = [i.get_facecolor() for i in ax.get_children() if isinstance(i, matplotlib.patches.PathPatch)]
# colors = list(map(lambda x : x[:-1],colors))[0:5]
# sns.lmplot(data=_Uncert_Error, x='Error ; mm', y='Uncertainty', hue='vessel/side', markers='_', fit_reg=True, height=15, aspect=1.5)
target_regions = ['artery_left', 'artery_right', 'vein_left', 'vein_right' ]
colors = ['darkorange','b','r','g']
for  i,region in enumerate(target_regions):
    _Data = _Uncert_Error[_Uncert_Error["vessel/side"] == region]
    X_test, y_pred, p_value, Coefficients = plot_linear_regression_line(X= _Data['Error ; mm'], Y= _Data['Uncertainty'])
    a = (y_pred[1] - y_pred[0]) / (X_test[1] - X_test[0])
    b = y_pred[1] - X_test[1] * a
    r, p = stats.pearsonr( _Data['Error ; mm'],  _Data['Uncertainty'])
#     ax.annotate(' DC = {:.3f} error + {:.3f}\n      ρ = {:.3f}   p value = {:.3f} '.format(a[0], b[0], r,
#                                                                                            p_value[1]),
#                 xy=(np.max(X_test) - 0.3, np.min(y_pred) - 0.05),
#                 color='g', fontsize=25)
#     slope, intercept, r_value, p_value, std_err = linregress(data['Error ; mm'], data['Uncertainty'])

    equation = 'uncertainty = {:.3f}error + {:.3f}'.format(a[0], b[0])
    r_squared = 'ρ = {:.3f} '.format(r)
    p_value = 'p value = {:e} '.format(p)
    plt.annotate(region + ":\n"+equation + "\n" + r_squared + ";" + p_value,  xy=(2.5, np.max(y_pred) - 0.005),
                  color=colors[i], fontsize=30)
# X_test, y_pred, p_value, Coefficients = plot_linear_regression_line(X=data['Error ; mm'], Y=data['Uncertainty'])
    plt.plot(X_test, y_pred, linewidth=2, linestyle='dashed', color=colors[i])
plt.show()
_TGT_FIG = _TGT + '/20_w_rev_nerves_5fold_Error_Distance'
os.makedirs(_TGT_FIG, exist_ok=True)
fig.savefig(os.path.join(_TGT_FIG,  'Uncertainty_distance_error_20_100_40_MIN.png'))

# for i, region in enumerate(target_regions):
#         data =_Uncert_Error [ _Uncert_Error ['vessel/side'] == region]
#         X_test, y_pred, p_value, Coefficients = plot_linear_regression_line(X=data['Error ; mm'], Y=data['Uncertainty'])
#         a = (y_pred[1] - y_pred[0]) / (X_test[1] - X_test[0])
#         b = y_pred[1] - X_test[1] * a
#         r, p = stats.pearsonr(data['Error ; mm'], data['Uncertainty'])
#     #     ax.annotate(' DC = {:.3f} error + {:.3f}\n      ρ = {:.3f}   p value = {:.3f} '.format(a[0], b[0], r,
#     #                                                                                            p_value[1]),
#     #                 xy=(np.max(X_test) - 0.3, np.min(y_pred) - 0.05),
#     #                 color='g', fontsize=25)
#     #     slope, intercept, r_value, p_value, std_err = linregress(data['Error ; mm'], data['Uncertainty'])
#
#         equation = 'uncertainty = {:.2f}error + {:.2f}'.format(a[0], b[0])
#         r_squared =  'ρ = {:.3f} '.format(r)
#         p = 'p value = {:.3f} '.format(p_value[1])
#         plt.annotate(equation + "\n" + r_squared + ";" + p, xy=(np.max(data['Error ; mm']), np.max(data['Uncertainty'])), xycoords='axes fraction' ,fontsize=25)
#         # X_test, y_pred, p_value, Coefficients = plot_linear_regression_line(X=data['Error ; mm'], Y=data['Uncertainty'])
#         plt.plot(X_test, y_pred, linewidth=2, linestyle='dashed', color= hex_colors[i])
    #     a = (y_pred[1] - y_pred[0]) / (X_test[1] - X_test[0])
    #     b = y_pred[1] - X_test[1] * a
    #     r, p = stats.pearsonr(Error, DC)
    #     ax.annotate(' DC = {:.3f} error + {:.3f}\n      ρ = {:.3f}   p value = {:.3f} '.format(a[0], b[0], r,
    #                                                                                            p_value[1]),
    #                 xy=(np.max(X_test) - 0.3, np.min(y_pred) - 0.05),
    #                 color='g', fontsize=25)
    #     plt.plot(X_test, y_pred, color='g', linewidth=2, linestyle='dashed')
