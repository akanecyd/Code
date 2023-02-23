import os
import pandas as pd
import numpy as np
import random
import seaborn as sns
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


def linear_regression_line(X, Y):
    X = X.values.reshape(-1, 1)
    y = Y.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    lr = LinearRegression()
    y_pred = lr.fit(X_train, y_train).predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return r2


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


def mul(x, y):
    try:
        return pd.to_numeric(x) * y
    except:
        return x


def is_scalar(param):
    pass


def multi_melt(
        df: pd.DataFrame,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
        ignore_index=True,
) -> pd.DataFrame:
    # Note: we don't broadcast value_vars ... that would seem unintuitive
    value_vars = value_vars if not is_scalar(value_vars[0]) else [value_vars]
    var_name = var_name if not is_scalar(var_name) else itertools.cycle([var_name])
    value_name = value_name if not is_scalar(value_name) else itertools.cycle([value_name])

    melted_dfs = [
        (
            df.melt(
                id_vars,
                *melt_args,
                col_level,
                ignore_index,
            ).pipe(lambda df: df.set_index([*id_vars, df.groupby(id_vars).cumcount()]))
        )
        for melt_args in zip(value_vars, var_name, value_name)
    ]

    return (
        pd.concat(melted_dfs, axis=1)
            .sort_index(level=2)
            .reset_index(level=2, drop=True)
            .reset_index()
    )


def point2point_distance(p1, p2):
    distance = ((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2 + (
                float(p1[2]) - float(p2[2])) ** 2) ** 0.5
    return distance


def readtxt(fpath):
    my_file = open(fpath, "r")
    data = my_file.read()
    data_into_list = data.replace('\n', ' ').split(",")
    print(data_into_list)
    my_file.close()
    return (data_into_list)

GT_data_root = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/PolyFigures_Risk'
# GT_points_root = '//Salmon/User/Chen/Vessel_data/revised_nerve/crop_hipjiont_75mm_ver3/Polygons/PolyDistances'
# AUTO_data_root = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_5fold/seperation_left_right/Polygons/PolyFigures'
AUTO_data_root ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyFigures_Risk'
# AUTO_points_root = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_5fold/seperation_left_right/Polygons/PolyDistances'
# AUTO_data_muscles_root ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold/seperation_left_right/Polygons/PolyFigures_nerve_back'
# AUTO_points_muscles_root = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold/seperation_left_right/Polygons/PolyDistances'
# AUTO_data_root = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_20_5fold_with_muscles/Polygons/DistanceFigures'
Accuracy_path = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/Evaluations/Accuracy/DC.csv'
# Accuracy_muscles_path = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold/Evaluations/Accuracy/DC.csv'
_PATIENT_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
auto_Affected_csv = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/Analysis/20221112_alldata_results.csv'
manual_Affected_csv = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/Analysis/20221123_OgawaIwasa50_ClinInfo.csv'
_color = './structures_json/vessels.json'
with open(_PATIENT_LIST) as f:
    case_IDs = f.read().splitlines()
to_remove = ['k10387', 'k8699', 'k8748', 'k9089', 'k8454','k8795']

# for item in to_remove:
#     while item in case_IDs:
#         case_IDs.remove(item)

Dice = pd.read_csv(Accuracy_path, header=0, index_col=0)
# Dice.index.name = 'case_id'
Dice.columns = ['pelvis_dice','femur_dice','vein_dice', 'artery_dice']
Affected = pd.read_csv(auto_Affected_csv, header=0, index_col= False)
manual_Affected = pd.read_csv(manual_Affected_csv, header=0, index_col= False)
# Dice['artery_dice']=mul(Dice['artery_dice'],5.0)
# Dice['vein_dice']=mul(Dice['vein_dice'],5.0)
print(Dice)
target_regions = ['artery_left', 'artery_right', 'vein_left', 'vein_right' ]
plot_color = ['red', 'green', 'blue', 'violet']
DC_class = ['artery_dice', 'artery_dice', 'vein_dice', 'vein_dice']
side = ['Left', 'Left', "Right", "Right"]
_TGT = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/Analysis'
markers = ['p', '*', '+', '^']
os.makedirs(_TGT, exist_ok=True)
for i, target_region in enumerate(target_regions):
    _GT_DF_F = os.path.join(GT_data_root, target_region + '.csv')
    _AUTO_DF_F = os.path.join(AUTO_data_root, target_region + '.csv')
    # _AUTO_DF_F_MUSCLES = os.path.join(AUTO_data_muscles_root, target_region + '.csv')

    _GT_DF = pd.read_csv(_GT_DF_F, header=0, index_col=0)
    _GT_DF.rename(columns={target_region: 'GT_distance (mm)'}, inplace=True)
    _GT_DF.head()
    _AUTO_DF = pd.read_csv(_AUTO_DF_F, header=0, index_col=0)
    _AUTO_DF.rename(columns={target_region: 'Predicted_distance (mm)'}, inplace=True)
    _AUTO_DF.head()
    # _AUTO_DF_Muscles = pd.read_csv(_AUTO_DF_F_MUSCLES, header=0, index_col=0)
    # _AUTO_DF_Muscles.rename(columns={target_region: 'Predicted_distance_with_muscles (mm)'}, inplace=True)
    # _AUTO_DF_Muscles.head()
    _FINAL = pd.concat([_GT_DF, _AUTO_DF], axis=1)
    print(_FINAL)
    diff = abs(_FINAL['Predicted_distance (mm)'] - _FINAL['GT_distance (mm)'])
    Dice.insert(i + 3, target_region, diff)
    # diff_muscles = abs(_FINAL['Predicted_distance_with_muscles (mm)'] - _FINAL['GT_distance (mm)'])
    # Dice.insert(i + 4, target_region, diff_muscles)
    Dice.insert(i + 4, target_region + '_GT', _FINAL['GT_distance (mm)'])
    Dice.insert(i + 5, target_region + '_Auto', _FINAL['Predicted_distance (mm)'])
    # Dice.insert(i + 6, target_region + '_Auto_with muscles', _FINAL['Predicted_distance_with_muscles (mm)'])
    print(Dice)
_FINAL = pd.melt(Dice.reset_index(), id_vars='PatientID',
                 value_vars=["artery_left", "artery_right", "vein_left", "vein_right"],
                 var_name=['vessel/side'],
                 value_name='Distance_error')
_FINAL.reset_index(inplace=True)
df = _FINAL.rename(columns={'PatientID': 'ID'})
print(df)
b = df.iloc[1, 2].split("_", 1)[0] + '_dice'
Vessels_dice = [Dice.loc[case_ID.lower()][df.iloc[i, 2].split("_", 1)[0]+'_dice' ] for i, case_ID in
                enumerate(df['ID'])]
df.insert(4, 'Dice', Vessels_dice)
GT_Distance = [Dice.loc[case_ID][df.iloc[i, 2] + '_GT'] for i, case_ID in enumerate(df['ID'])]
df.insert(5, 'GT Distance(mm)', GT_Distance)
Predicted_Distance = [Dice.loc[case_ID][df.iloc[i, 2] + '_Auto'] for i, case_ID in enumerate(df['ID'])]
df.insert(6, 'Predicted Distance(mm)', Predicted_Distance)
df['Institute'] = 'Institute 1'
# Predicted_Distance = [Dice.loc[case_ID.lower()][df.iloc[i, 2] + '_Auto_with muscles'] for i, case_ID in enumerate(df['ID'])]
# df.insert(9, 'Predicted Distance with muscles (mm)', Predicted_Distance)
# df = df[~df['ID'].isin(to_remove)]
# df.insert(7, 'affection',value=None)
a= [df.iloc[1, 2].split("_", 1)[1] ]
Affected['ImageID']=Affected['ImageID'].apply(lambda x: x.split('.')[0].lower())
Affected = Affected.set_index('ImageID')
imageid=[]
for i, case_ID in enumerate(df['ID']):
    id =case_ID.lower()
    side = df.iloc[i, 2].split("_", 1)[1]
    imgid = id +'_'+ side
    imageid.append(imgid)
import re
ckl = []
for imgid in imageid:
    try:
        pred_label = Affected.loc[imgid]['PredLabel']
        ckl.append(pred_label)
    except KeyError:
        ckl.append(None)



KID = [x for x in manual_Affected['KID'] if x in case_IDs]
manual_Affected=manual_Affected.set_index('KID')
crow=[]
CKL = []
for i, case_ID in enumerate(df['ID']):
    id =case_ID.lower()
    side = df.iloc[i, 2].split("_", 1)[1].capitalize()
    # Cr = manual_Affected.loc[id]['Crowe_' + side]
    try:
      Cr = manual_Affected.loc[id]['Crowe_'+side]
      kl = manual_Affected.loc[id]['KL_'+side]
      cc = 'Crowe {},KL {}'.format(Cr, kl)
      CKL.append(cc)
    # try:
    #     pred_label = Affected.loc[imgid]['PredLabel']
    #     ckl.append(pred_label)
    except KeyError:
        CKL.append(None)


result = []
for x, y in zip(CKL, ckl):
    if x is None and y is not None:
        result.append(y)
    else:
        result.append(x)
# print(result)
df.insert(7, 'affection', result)

# crowe = [Affected.loc[case_ID.lower()+'_'+df.iloc[i, 2].split("_", 1)[1]]['PredLabel'] for i, case_ID in enumerate(df['ID'])]
print(df)
# df.to_csv('//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/Analysis/CKL_GT_AUTO_33_Osaka.csv')





##### box_plot
df = df.dropna(subset=['affection'])
_Box_data = df
regions = ['artery_left', 'artery_right', 'vein_left', 'vein_right']
for region in regions:
    a = region.split("_", 1)[0]
    _Box_data.replace(region, a, inplace=True)
    # _Box_data.replace('vessel/side', 'vessel_nerve', inplace=True)
_Box_data.replace(['Crowe 1,KL 0','Crowe 1,KL 1'],'Crowe 1,KL 0,1', inplace=True)
_Box_data.replace(['Crowe 1,KL 3','Crowe 1,KL 4'],'Crowe 1,KL 3,4', inplace=True)
# _Distance = pd.concat([_Box_data['GT Distance(mm)'],_Box_data['Predicted Distance(mm)']])


# Use seaborn's boxplot function to create the plot
# _Distance_box = _Box_data[_Box_data ['vessel/side'] =='vessel/side']
_Distance_box = pd.melt(_Box_data.reset_index(), id_vars=['ID','vessel/side','Institute','affection'],
                 value_vars=['GT Distance(mm)','Predicted Distance(mm)'],
                 var_name=['GT/Auto'],value_name='Minimum Distance(mm)')
_Distance_box.replace('GT Distance(mm)', 'GT', inplace=True)
_Distance_box.replace('Predicted Distance(mm)', 'Predicted', inplace=True)
# _Distance_box = pd.melt(_Box_data.reset_index(), id_vars=['ID','vessel/side'],
#                  value_vars=['Institute','GT/Auto'],
#                  var_name=['Institute/Label'],value_name='Institute+Label')
# _Distance_box = _Box_data
_Distance_box['affection+organ'] = _Distance_box['vessel/side'].str.cat(_Distance_box['affection'], sep=' ')
_Distance_box['affection+label'] = _Distance_box['affection'].str.cat(_Distance_box['GT/Auto'], sep=' ')
# Use seaborn's boxplot function to create the plot
# Use seaborn's boxplot function to create the plot
# sns.catplot(x='Institute', y='Minimum Distance(mm)', hue='GT/Auto', col='vessel/side', data=vein_artery_data, kind='box', showfliers=False)

# Show the plot
# plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 45))
with open('structures_json/vessels.json', 'r') as color_file:
    full_color = json.load(color_file)
full_color_list = np.array(full_color['color'])
full_color_list = np.array(full_color_list[:, 2:6], dtype=np.float)
full_color_list = full_color_list.tolist()
_STRUCT = 'Minimum Distance(mm)'
my_pal = {"vein": full_color_list[0], "artery": full_color_list[1]}
# com_color = {"GT Distance(mm)": 'paleturquoise', "Predicted Distance(mm)": 'mediumpurple'}
com_color = {"GT": 'paleturquoise', "Predicted": 'lightpink'}
# box = sns.boxplot(x='vessel_nerve/side', y=_STRUCT, data=_Distance_box, ax=ax,
#                   showmeans=True, hue = 'GT/Auto', palette= com_color,showfliers = False,
#                   meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue",
#                              "markersize": 10.0})
# sns.stripplot(x='vessel_nerve/side', y=_STRUCT, data=_Distance_box, color='k', size=10, hue = 'GT/Auto',
#               jitter=True, dodge=True, marker='.', edgecolor='k', alpha=0.8, ax=ax)
# box= sns.violinplot(x="vessel/side",  y=_STRUCT, data=_Distance_box, ax=ax, hue='Institute', palette=com_color,split=True,inner='stick')
# sns.swarmplot(x="vessel/side",  y=_STRUCT, data=_Distance_box, hue='Institute', color="w", alpha=.5)


box = sns.boxplot(x='affection+organ', y=_STRUCT, data=_Distance_box, ax=ax, hue='GT/Auto',
                   showmeans=True, showfliers = False, width= 0.8, palette= com_color,
                               order = ['artery Crowe 1,KL 0,1','artery Crowe 1,KL 2',
                                        'artery Crowe 1,KL 3,4','artery Crowe 2,KL 4',
                                        'vein Crowe 1,KL 0,1','vein Crowe 1,KL 2',
                                        'vein Crowe 1,KL 3,4','vein Crowe 2,KL 4',
                                        ],

                  meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue",
                             "markersize": 10.0})
sns.stripplot(x='affection+organ', y=_STRUCT, data=_Distance_box, color='k', size=15, hue='GT/Auto',
              order=['artery Crowe 1,KL 0,1', 'artery Crowe 1,KL 2',
                     'artery Crowe 1,KL 3,4', 'artery Crowe 2,KL 4',
                     'vein Crowe 1,KL 0,1', 'vein Crowe 1,KL 2',
                     'vein Crowe 1,KL 3,4', 'vein Crowe 2,KL 4',
                     ],
              jitter=True, dodge=True, marker='.', edgecolor='k', alpha=0.8, ax=ax)
_X_TICKS = ax.get_xticklabels()
colors = [i.get_facecolor() for i in ax.get_children() if isinstance(i, matplotlib.patches.PathPatch)]
colors = list(map(lambda x : x[:-1],colors))[0:16]
# cell_text = []
# _EXPs = ["Institute 1", "Institute 2"]
# _TMP_STRUCT_NAMES = ['artery', 'vein']
# for _EXP in _EXPs:
#
#     sub_dat = _Distance_box [_Distance_box ['Institute'] == _EXP]
#     row = ['%.3f±%.3f mm' % (sub_dat[sub_dat['vessel/side']== _STR][_STRUCT].mean(skipna=True),
#                           sub_dat[sub_dat['vessel/side'] == _STR][_STRUCT].std(skipna=True))
#            for _STR in _TMP_STRUCT_NAMES]
#     cell_text.append(row)
# Add p-values
# _TMP_STRUCT_NAMES = ['artery', 'vein', 'nerve']
# _EXPs = ["Points Distance(mm)", "Points Distance with Muscles(mm)"]
# for _x, _STR in enumerate(_TMP_STRUCT_NAMES):
#     sub_df = _Distance_box[_Distance_box['vessel_nerve/side'] == _STR]
#     print(sub_df[sub_df['w/wo muscles'] == _EXPs[0]]['Points Distance(mm)'].values)
#     print(sub_df[sub_df['w/wo muscles'] == _EXPs[1]]['Points Distance(mm)'].values)
#     [u_test_val, p_val] = stats.ttest_rel(sub_df[sub_df['w/wo muscles'] == _EXPs[0]]['Points Distance(mm)'].values,
#                     sub_df[sub_df['w/wo muscles']==_EXPs[1]]['Points Distance(mm)'].values,
#                     nan_policy='omit')
#     # _y = np.max(sub_df[_MEASURE['name']].values)
#     _Measure = np.max(sub_df['Points Distance(mm)'])+1
#     _y =_Measure - 0.005 * _Measure
#     # if _y > _MEASURE['range'][1]:
#     #     _y = _MEASURE['range'][1] - 0.1 * _MEASURE['range'][1]
#     # p_str = convert_pval_to_asterisk(p_val)
#     # main_axs.text(_x, _y+_y*0.05, '%s' % (p_str),
#     #         color='red' if p_str!='n.s.' else 'black')
#     ax.text(_x - 0.2, _y, 'P: %0.3f' % (p_val),
#                   color='red' if p_val < 0.05 else 'black',fontsize=25)
# Add table
#
cell_text = []

_LABELs = ['Crowe 1,KL 0,1 GT', 'Crowe 1,KL 0,1 Predicted',
            'Crowe 1,KL 2 GT', 'Crowe 1,KL 2 Predicted',
            'Crowe 1,KL 3,4 GT', 'Crowe 1,KL 3,4 Predicted',
            'Crowe 2,KL 4 GT', 'Crowe 2,KL 4 Predicted',
                     ]
_TMP_STRUCT_NAMES = ['artery', 'vein']

for _LABEL in _LABELs:
    sub_dat = _Distance_box [ _Distance_box ['affection+label'] == _LABEL]
    print(f"Number of rows in sub_dat: {sub_dat.shape[0]}")
    row = ['%.3f±%.3f mm' % (sub_dat[sub_dat['vessel/side']== _STR][_STRUCT].mean(skipna=True),
                          sub_dat[sub_dat['vessel/side'] == _STR][_STRUCT].std(skipna=True))
           for _STR in _TMP_STRUCT_NAMES]
    cell_text.append(row)
# row = ['%.3f±%.3f ' % (_Box_data[_Box_data['vessel_nerve/side'] == _STR][_STRUCT].mean(skipna=True),
#                       _Box_data[_Box_data['vessel_nerve/side'] == _STR][_STRUCT].std(skipna=True))
#        for _STR in _TMP_STRUCT_NAMES]
# cell_text.append(row)
print(cell_text)
# the_table = plt.table(cellText=cell_text,
#                       colLabels=_TMP_STRUCT_NAMES,
#                       rowLoc='center',
#                       # rowLabels=_EXPs,
#                       # rowColours=['paleturquoise', 'mediumpurple'],
#                       # colColours=[[0.85, 0.4, 0.31, 0.65],
#                       #             [0.0, 0.59, 0.81, 0.65],
#                       #             [0.96, 0.84, 0.19, 0.65]],
#                       loc='bottom',
#                       colLoc='center')
the_table = plt.table(cellText=cell_text,
                      colLabels=_TMP_STRUCT_NAMES,
                      rowLoc='center',
                      rowLabels=['Crowe 1,KL 0,1 GT', 'Crowe 1,KL 0,1 Predicted',
                                 'Crowe 1,KL 2 GT', 'Crowe 1,KL 2 Predicted',
                                 'Crowe 1,KL 3,4 GT', 'Crowe 1,KL 3,4 Predicted',
                                 'Crowe 2,KL 4 GT', 'Crowe 2,KL 4 Predicted',
                     ],
                      rowColours=colors,
                      colColours=[[0.85, 0.4, 0.31, 0.65],
                                  [0.0, 0.59, 0.81, 0.65],
                                  [0.96, 0.84, 0.19, 0.65]],
                      loc='bottom',
                      colLoc='center',
                      cellLoc = 'center')
ax.set_xticklabels('')
ax.set_xlabel('')
# plt.xticks(np.arange(len(_X_TICKS))
# ax.set_xticks(np.arange(len(_X_TICKS)))

Xticks =['Crowe 1 \nKL 0,1\n N = 25','Crowe 1 \nKL 2\n N = 8',
        'Crowe 1 \nKL 3,4\n N = 29','Crowe 2 \nKL 4\n N = 4',
        'Crowe 1 \nKL 0,1\n N = 25','Crowe 1 \nKL 2\n N = 8',
        'Crowe 1 \nKL 3,4\n N = 29','Crowe 2 \nKL 4\n N = 4']
for i, label in enumerate(Xticks):
    ax.text(i, 0.01, label, ha='center',fontsize=35)
the_table.scale(1, 4.5)
# # the_table.auto_set_font_size(True)
the_table.set_fontsize(45)
# # main_axs.set_xticklabels('')
# # main_axs.set_xlabel('')
# the_table.scale(1.0, 3.0)
plt.ylim(-0.5, max(_Distance_box[_STRUCT])+1)
plt.ylabel(' Minimum distance ; mm',fontsize=45)
plt.title(' Minimum distance Distribution of GT-Auto with Affection', fontsize=45)
# plt.xlabel('Organs', fontsize=30)
plt.xticks(fontsize=40)
plt.yticks(fontsize=35)
# plt.legend()
plt.legend([],[], frameon=False)
plt.subplots_adjust(left=0.1, bottom=0)
plt.tight_layout()
plt.show()
_TGT_FIG = _TGT + '/20_w_rev_nerves_5fold_Error_Distance'
os.makedirs(_TGT_FIG, exist_ok=True)
fig.savefig(os.path.join(_TGT_FIG,  'GT_AUto_Mindistance_Affection_2.png'))
# #
#
# ##   variable      value
# ## 0     leaf   9.446465
# ## 1     leaf  11.163702
# ## 2     leaf  14.296799
# ## 3     leaf   7.441026
# ## 4     leaf  11.004554
#
# # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 13))
# # fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw={'width_ratios': [3, 1], 'height_ratios’: [3, 3]})
# # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10),
# #                        gridspec_kw={'width_ratios': [1, 1],'height_ratios': [1, 1]})
#
# # _Artery = df[(df['vessel_nerve/side'] == 'artery_left') | (df['vessel_nerve/side'] == 'artery_right')]
# # _Vein = df[(df['vessel_nerve/side'] == 'vein_left') | (df['vessel_nerve/side'] == 'vein_right')]
# # print(_Artery)

# #### Distance VS Dice plot
# vessels = ["artery", "vein"]
# sides = ["left", "right"]
# df = df_N
# for vessel in vessels:
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
#     _Data = df[(df['vessel/side'] == vessel + '_left') | (df['vessel/side'] == vessel + '_right')]
#     _Data["Distance_error"] = abs(_Data["Distance_error"])
#     # _Data = df[(df['vessel/side'] == "artery_" + side) | (df['vessel/side'] == "vein_" + side)]
#     sns.scatterplot(data=_Data, x="Distance_error", y="Dice",
#                     edgecolor=plot_color[2], palette='bright',
#                     alpha=0.6, s=90, linewidths=5)
#     # Dataset = pd.concat([_Artery["Distance_error"], _Artery["Dice"]], axis=1).to_xarray()
#     # sns.jointplot(x='Distance_error', y='Dice', data=_Data,
#     #               kind="hex", color="r",  # 主图为六角箱图
#     #               size=6, space=0.1,
#     #               joint_kws=dict(gridsize=20, edgecolor='w'),  # 主图参数设置
#     #               marginal_kws=dict(bins=20, color='g',
#     #                                 hist_kws={'edgecolor': 'k'}))  # 边缘图设置
#     #                   # annot_kws=dict(stat='r', fontsize=15))
#     # plt.show()
#
#     affects = ['unaffected', 'affected']
#     regression_color = ['blue', 'orange']
#
#     if vessel == 'nerve':
#         _outlier = _Data[(_Data['Dice'] <= 0.8) & (_Data["Distance_error"] <= 2.0)]
#         _lager = _Data[_Data["Distance_error"] >= 10.0]
#         _outliers = pd.concat([_outlier,_lager])
#     else:
#         _outlier = _Data[(_Data['Dice'] <= 0.7) & (_Data["Distance_error"]<= 2.0)]
#         _lager = _Data[( _Data["Distance_error"] )>= 2.0]
#         _outliers = pd.concat([_outlier, _lager])
#     outliers = _outliers['ID']
#
#     for i, outlier in enumerate(outliers):
#         print(_outliers.iloc[i, 3], _outliers.iloc[i, 4])
#         ax.annotate(outlier,
#                     xy=(_outliers.iloc[i, 3]+random.uniform(-0.01, 0.01), _outliers.iloc[i, 4]+random.uniform(-0.01, 0.01)),
#                     color='r', fontsize=15)
#     Error = _Data["Distance_error"]
#     DC = _Data["Dice"]
#     X_test, y_pred, p_value, Coefficients = plot_linear_regression_line(X=Error, Y=DC)
#     a = (y_pred[1] - y_pred[0]) / (X_test[1] - X_test[0])
#     b = y_pred[1] - X_test[1] * a
#     r, p = stats.pearsonr(Error, DC)
#     ax.annotate(' DC = {:.3f} error + {:.3f}\n      ρ = {:.3f}   p value = {:.3f} '.format(a[0], b[0], r,
#                                                                                            p_value[1]),
#                 xy=(np.max(X_test) - 0.3, np.min(y_pred) - 0.05),
#                 color='g', fontsize=25)
#     plt.plot(X_test, y_pred, color='g', linewidth=2, linestyle='dashed')
#     # for i, affect in enumerate(affects):
#     #     _affect = _Data[(_Data['Affected'] == affect)]
#     #     _affect.sort_values('Distance_error')
#     #     Error = _affect["Distance_error"]
#     #     DC = _affect["Dice"]
#     #     X_test, y_pred, p_value, Coefficients = plot_linear_regression_line(X=Error, Y=DC)
#     #     a = (y_pred[1] - y_pred[0]) / (X_test[1] - X_test[0])
#     #     b = y_pred[1] - X_test[1] * a
#     #     r, p = stats.pearsonr(Error, DC)
#     #     ax.annotate(affect + ' DC = {:.2f} error + {:.2f}\n      ρ = {:.2f}   p value = {:.2f} '.format(a[0], b[0], r,
#     #                                                                                                     p_value[1]),
#     #                 xy=(np.max(X_test) - 0.08, np.min(y_pred) - 0.05),
#     #                 color=regression_color[i], fontsize=20)
#     #     plt.plot(X_test, y_pred, color=regression_color[i], linewidth=2, linestyle='dashed')
#     ax.grid(True)
#     ax.set_aspect('equal', 'box')
#     ax.set_facecolor('w')
#     plt.title(vessel + ' Error of GT-predicted Minimal Distance  VS Dice', fontsize=35)
#     plt.grid(True)
#     plt.xlim(-0.1, 4)
#
#     plt.xticks(np.arange(0, 4.5, 0.5))
#     if vessel == 'nerve':
#         plt.ylim(0.0, 1.0)
#         plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     else:
#         plt.ylim(0.2, 1.0)
#         plt.yticks([ 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
#
#     plt.ylabel('DC', fontsize=35)
#     plt.xlabel('Error of GT-predicted minimum distance\n (Prediction error; mm)', fontsize=35)
#     plt.xticks(fontsize=30)
#     plt.yticks(fontsize=30)
#     # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = OrderedDict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc='upper right',
#                fontsize=25)
#
#     ratio = 1
#
#     # get x and y limits
#     x_left, x_right = ax.get_xlim()
#     y_low, y_high = ax.get_ylim()
#
#     # set aspect ratio
#     ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
#     plt.tight_layout()
#     plt.show()
#     _TGT_FIG = _TGT + '/20_w_rev_nerves_5fold_Error_Distance'
#     os.makedirs(_TGT_FIG, exist_ok=True)
#     fig.savefig(os.path.join(_TGT_FIG, vessel + '_Nara_18.png'))
