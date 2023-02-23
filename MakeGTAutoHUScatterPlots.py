from dataclasses import replace
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from scipy import stats
from scipy.stats import wilcoxon, pearsonr
import statsmodels.api as sm
import tqdm
from functools import partial

from utils import vis

font = {'family': 'Calibri',
        'weight': 'normal',
        'size': 30}

matplotlib.rc('font', **font)


def replace_text(x, tag_in='', tag_out=''):
    return x.replace(tag_in, tag_out)


def outlier_annotated_bland_altman_plot(data1, data2, *args, **kwargs):
    pat_list = None
    if 'pat_list' in kwargs:
        pat_list = kwargs['pat_list']
    plt.figure(figsize=kwargs['figsize'], dpi=80)

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    upper_lim = md + 1.96 * sd
    lower_lim = md - 1.96 * sd

    # Plot
    plt.scatter(mean, diff, alpha=kwargs['alpha'])  # , *args, **kwargs
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(upper_lim, color='gray', linestyle='--')
    plt.axhline(lower_lim, color='gray', linestyle='--')

    # Add annotations
    _x = np.min(mean)
    plt.text(_x, upper_lim, '+1.96 SD: %6.3f' % (upper_lim), fontsize=12)
    plt.text(_x, lower_lim, '-1.96 SD: %6.3f' % (lower_lim), fontsize=12)

    if pat_list:
        idxs = []
        sorted_diff_idx = np.argsort(diff)
        idxs = np.concatenate([sorted_diff_idx[0:5], sorted_diff_idx[-5:]])

        for _idx in idxs:
            _x = mean[_idx] + 0.1 * mean[_idx] * (np.random.uniform() - 0.5)
            _y = diff[_idx] + 0.05 * mean[_idx] * (np.random.uniform() - 0.5)
            _txt = pat_list[_idx]
            plt.text(_x, _y, _txt)

        return idxs


GT_data_root = '//Salmon/User/mazen/Segmentation/Data/HipMusclesDataset/Polygons_SingleSides/DistanceFigures'
AUTO_data_root = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_20_5fold_wo_muscles/Polygons/DistanceFigures'
# AUTO_data_root = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_20_5fold_with_muscles/Polygons/DistanceFigures'
Accuracy_path = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_20_5fold_wo_muscles/Evaluations/Accuracy/DC.csv'
_PATIENT_LIST = '//Salmon/User/Chen/Vessel_data/new_8_muscle_vessel/caseid_list_20.txt'
with open(_PATIENT_LIST) as f:
    case_IDs = f.read().splitlines()
Dice = pd.read_csv(Accuracy_path, header=0, index_col=0)
print(Dice)
target_regions = ['artery_left', 'artery_right', 'vein_left', 'vein_right']
plot_color =['red','green','blue','yellow']
DC_class =['Vein_DC','Vein_DC','Artery_DC','Artery_DC']
_TGT = 'D:/temp/visualization'
os.makedirs(_TGT, exist_ok=True)
for i, target_region in enumerate(target_regions):
    _GT_DF_F = os.path.join(GT_data_root, target_region + '.csv')
    _AUTO_DF_F = os.path.join(AUTO_data_root, target_region + '.csv')

    _GT_DF = pd.read_csv(_GT_DF_F, header=0, index_col=0)
    _GT_DF.rename(columns={target_region: 'GT_distance (mm)'}, inplace=True)
    _GT_DF.head()
    _AUTO_DF = pd.read_csv(_AUTO_DF_F, header=0, index_col=0)
    _AUTO_DF.rename(columns={target_region: 'Predicted_distance (mm)'}, inplace=True)
    _AUTO_DF.head()
    _FINAL = pd.concat([_GT_DF, _AUTO_DF], axis=1)
    print(_FINAL)
    # _FINAL = _FINAL.rename(columns={"Disease category (1-5)":"Disease category"})

    # Unaffected versus Affected
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    vis.confidence_ellipse(_FINAL['GT_distance (mm)'], _FINAL['Predicted_distance (mm)'], ax, col='red', edgecolor='red',
                           linestyle='--', n_std=2, facecolor='red')
    # ax = sns.jointplot(x = "GT_distance (mm)", y = "AUTO_distance (mm)",
    #                kind = "reg", data =_FINAL, dropna = True)
    sc = sns.scatterplot(data=_FINAL, x='GT_distance (mm)', y='Predicted_distance (mm)',
                         edgecolor='k',
                         alpha=0.6, s=50, linewidths=5)
    ax.grid(True)
    r, p = stats.pearsonr(_FINAL['GT_distance (mm)'], _FINAL['Predicted_distance (mm)'])
    lim = math.ceil(np.max([_FINAL['GT_distance (mm)'], _FINAL['Predicted_distance (mm)']]))
    x = y = range(-2, 22)
    ax.plot(x, y)
    # sns.set(font_scale=2)
    ax.annotate('r = {:.2f} '.format(r), xy=(.7, .9), xycoords=ax.transAxes)


    ax.set_facecolor('w')
    plt.title(target_region, fontsize=30)
    plt.grid(True)
    plt.xlim(-2, 20)
    plt.ylim(-2, 20)
    plt.xticks(np.arange(0, 22, step=2))
    plt.yticks(np.arange(0, 22, step=2))
    plt.show()
    fig.savefig(os.path.join(_TGT, target_region+'_without_muscles.png'))
    diff = abs(_FINAL['Predicted_distance (mm)'] - _FINAL['GT_distance (mm)'])
    _FINAL.insert(2, 'Error(mm)', diff)
    print(_FINAL)
    # _FINAL.to_csv(os.path.join(_TGT, target_region+'distance_without_muscles.csv'))

    vein_dice = [Dice.loc[case_ID.lower()][0] for case_ID in _FINAL.index]
    artery_dice = [Dice.loc[case_ID.lower()][1] for case_ID in _FINAL.index]
    print(vein_dice)
    _FINAL.insert(3, 'Vein_DC', vein_dice)
    _FINAL.insert(4, 'Artery_DC', artery_dice)
    _FINAL.sort_values("Error(mm)", inplace=True)
    print(_FINAL)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # vis.confidence_ellipse(_FINAL['GT_distance (mm)'], _FINAL['Predicted_distance (mm)'], ax, col='red',
    #                        edgecolor='red',
    #                        linestyle='--', n_std=2, facecolor='red')
    # ax = sns.jointplot(x = "GT_distance (mm)", y = "AUTO_distance (mm)",
    #                kind = "reg", data =_FINAL, dropna = True)
    sns.scatterplot(data=_FINAL, x='Error(mm)', y=DC_class[i],
                    edgecolor=plot_color[i], palette=plot_color[i],color=plot_color[i],
                    alpha=0.6, s=50, linewidths=5)
    ax.grid(True)
    ax.plot(_FINAL['Error(mm)'], _FINAL[DC_class[i]],color=plot_color[i])
    # sns.set(font_scale=2)
    # ax.annotate('r = {:.2f} '.format(r), xy=(.7, .9), xycoords=ax.transAxes)
    ax.set_aspect('equal', 'box')
    ax.set_facecolor('w')
    plt.title('Distance error VS Dice', fontsize=20)
    plt.grid(True)
    plt.xlim(0, 4)
    plt.ylim(0, 1.2)
    plt.xticks(np.arange(0, 4, step=0.5))
    plt.yticks(np.arange(0, 1.2, step=0.5))
    plt.ylabel('DC',fontsize=20)
    plt.xlabel('Error(mm)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
plt.show()







# b=list(_GT_DF.columns)
# print(b)
# _STRUCTURES =list([struct for struct in list(_GT_DF.columns) if (('aff' in struct) or ('un' in struct)) if 'Average' not in struct if 'side' not in struct])
# print(*_STRUCTURES, sep='\n')

# struct_idxs = range(0,len(_STRUCTURES),2)
#
# _HU_COLS = [col for col in _GT_DF.columns if 'HU' in col] + ['Disease category (1-5)']
# _GT_DF = _GT_DF.loc[:_GT_DF.index]
# _AUTO_DF = _AUTO_DF.loc[:,_GT_DF.index]
# print(_GT_DF.columns)
# print(_AUTO_DF.columns)

# _func_un = partial(replace_text, tag_in='un_')
# _func_aff = partial(replace_text, tag_in='aff_')
# _func_feat = partial(replace_text, tag_in='_mean_HU')
#
# _FINAL = pd.concat([_GT_DF, _AUTO_DF], axis=1)
# print(_FINAL)
# # _FINAL = _FINAL.rename(columns={"Disease category (1-5)":"Disease category"})
#
#
# # Unaffected versus Affected
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
#
#
#
vis.confidence_ellipse(_FINAL['GT_distance (mm)'], _FINAL['Predicted_distance (mm)'], ax, col='red', edgecolor='red',
                       linestyle='--', n_std=2, facecolor='red')
# ax = sns.jointplot(x = "GT_distance (mm)", y = "AUTO_distance (mm)",
#                kind = "reg", data =_FINAL, dropna = True)
ax = sns.scatterplot(data=_FINAL, x='GT_distance (mm)', y='Predicted_distance (mm)',
                     edgecolor='k',
                     alpha=0.6, s=50, linewidths=5)
r, p = stats.pearsonr(_FINAL['GT_distance (mm)'], _FINAL['Predicted_distance (mm)'])
lim = np.max([_FINAL['GT_distance (mm)'], _FINAL['Predicted_distance (mm)']])
x = y = range(-2, math.ceil(lim))
ax.plot(x, y)
sns.set(font_scale=2)
ax.annotate('r = {:.2f} '.format(r), xy=(.7, .9), xycoords=ax.transAxes)
ax.set_aspect('equal', 'box')
plt.title('vein_left', fontsize=30)
plt.xlim(-2, lim + 1.5)
plt.ylim(-2, lim + 1.5)
plt.xticks(np.arange(0, lim + 2, step=2))
plt.yticks(np.arange(0, lim + 2, step=2))
plt.show()
fig.savefig(os.path.join(_TGT, 'vein_left_without_muscles.png'))




# j.ax_marg_x.set_axis_off()
# j.ax_marg_y.set_axis_off()
# ax.set(xlabel='GT_distance (mm)')
# ax.set(ylabel='AUTO_distance (mm)')
# # plt.title('Artery_Left',fontsize=12)
# ax.xaxis.get_label().set_fontsize(8)
# # show the plot
# plt.show()
# ax = sns.scatterplot(data=_FINAL, x='artery_left_GT', y='artery_left_AUTO',
#                      edgecolor='k',
#                      alpha=0.6, palette=['r', 'y', 'b'])
#
# _GT_MEANS, _AUTO_MEANS = [], []
#
# for _stg, _col in zip([1, 2, 3], ['r', 'y', 'b']):
#     x = np.array(_FINAL.loc[(_FINAL['Disease category'] == _stg) & \
#                             (_FINAL['Type'] == 'GT'), 'Unaffected']).astype(np.float32)
#     y = np.array(_FINAL.loc[(_FINAL['Disease category'] == _stg) & \
#                             (_FINAL['Type'] == 'GT'), 'Affected']).astype(np.float32)
#     vis.confidence_ellipse(x, y, ax, col=_col, edgecolor=_col, linestyle='-', n_std=2)
#     ax.plot(np.mean(x), np.mean(y), color=_col, marker='o')
#     _GT_MEANS.append([np.mean(x), np.mean(y)])
#
# for _stg, _col in zip([1, 2, 3], ['r', 'y', 'b']):
#     x = np.array(_FINAL.loc[(_FINAL['Disease category'] == _stg) & \
#                             (_FINAL['Type'] == 'Auto'), 'Unaffected']).astype(np.float32)
#     y = np.array(_FINAL.loc[(_FINAL['Disease category'] == _stg) & \
#                             (_FINAL['Type'] == 'Auto'), 'Affected']).astype(np.float32)
#     vis.confidence_ellipse(x, y, ax, col=_col, edgecolor=_col, linestyle='--', n_std=2)
#     ax.plot(np.mean(x), np.mean(y), color=_col, marker='x')
#     _AUTO_MEANS.append([np.mean(x), np.mean(y)])
#
# # for _stg, _col in zip([1,2], ['r', 'y']):
# #     x, y = _GT_MEANS[_stg-1]
# #     dx, dy = _GT_MEANS[_stg][0]-_GT_MEANS[_stg-1][0],\
# #              _GT_MEANS[_stg][1]-_GT_MEANS[_stg-1][1]
# #     plt.arrow(x=x, y=y, dx=dx, dy=dy, color=_col, width=0.5, length_includes_head=True)
#
# #     x, y = _AUTO_MEANS[_stg-1]
# #     dx, dy = _AUTO_MEANS[_stg][0]-_AUTO_MEANS[_stg-1][0],\
# #              _AUTO_MEANS[_stg][1]-_AUTO_MEANS[_stg-1][1]
# #     plt.arrow(x=x, y=y, dx=dx, dy=dy, color=_col, width=0.5, linestyle='--', length_includes_head=True)
#
# ax.set_aspect('equal', 'box')
# ax.set_xlabel(x_label, fontsize=30)
# ax.set_ylabel(y_label, fontsize=30)
#
# # plt.legend(['Affected (\u03C1=%.3f)' % ro_aff,
# #             'Unaffected (\u03C1=%.3f)' % ro_un])
# lims_x, lims_y = [np.min([_FINAL['Affected'], _FINAL['Unaffected']]),
#                   np.max([_FINAL['Affected'], _FINAL['Unaffected']])], \
#                  [np.min([_FINAL['Affected'], _FINAL['Unaffected']]),
#                   np.max([_FINAL['Affected'], _FINAL['Unaffected']])]
# lims_x = [lims_x[0] - 0.05 * lims_x[0], lims_x[1] + 0.05 * lims_x[1]]
# lims_y = [lims_y[0] - 0.05 * lims_y[0], lims_y[1] + 0.05 * lims_y[1]]
# plt.plot(lims_x, lims_y, '--k', alpha=0.3)
# ax.set_xlim(lims_x)
# ax.set_ylim(lims_y)
# plt.tight_layout()
# plt.savefig(os.path.join(_TGT, 'All_muscle_HU_un_aff.png'), dpi=300)
# plt.close()
