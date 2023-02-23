import os
import numpy as np
import pandas as pd
import glob
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt
import math
import imageio

from itertools import combinations
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold


#%%

_nara = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold/seperation_left_right/Polygons/PolyDistances_20230214/Nara18_index_rimpoints_to_vessels_distance.csv'
_nara_DF = pd.read_csv(_nara, header=0,index_col=1)
_osaka = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyDistances_20230214/Osaka36_index_rimpoints_to_vessels_distance.csv'
_osaka_DF = pd.read_csv(_osaka, header=0,index_col=1)
_TGT_path = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/Analysis_20230214'
Osaka_affection ='//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/Treat_side.csv'
_osaka_affect = pd.read_csv(Osaka_affection, header=0,index_col=0)
# organ = ['artery_right', 'artery_left', 'vein_right', 'vein_left']
operate = _osaka_affect.loc[_osaka_DF.index, 'TrighteatedSide']
a=_osaka_DF.iloc[0,1].split("_", 1)[1]
b= operate.iloc[0]
osaka_operated = ['operated' if _osaka_DF.iloc[i,1].split("_", 1)[1] == operate.iloc[i] else 'unoperated' for i in range(0, len(_osaka_DF.index))]
_osaka_DF.insert(2, 'Operated', osaka_operated)
regions = ['artery_left', 'artery_right', 'vein_left', 'vein_right']
for region in regions:
    a = region.split("_", 1)[0]
    # _Box_data.replace('vessel/side', 'vessel_nerve', inplace=True)
    _osaka_DF.replace(region, a, inplace=True)
    _nara_DF.replace(region, a, inplace=True)
# a = side.split("_", 1)[1]
print(_nara_DF)
print(_osaka_DF)

#%%

for df, title in zip([_nara_DF, _osaka_DF], ['Institute 2', 'Institute 1']):
  if title == 'Institute 1':
    for _organ in ['artery', 'vein']:
        for _type in ['operated','unoperated']:#['GT', 'Auto']:
            _tmp_df = df.loc[(df['vessel/side']==_organ) & (df['Operated']==_type)] #& (_nara_DF['GT/AUto']==_type)
            _tmp_df = _tmp_df[[x for x in _tmp_df.columns if any([('index' in x) or ('GT' in x)])]]
            _tmp_df.columns = ['GT/Auto'] + [int(x.replace('index ', '')) for x in _tmp_df.columns if 'index' in x]
            _tmp_df.reset_index(inplace=True)
            _tmp_df = _tmp_df.melt(id_vars=['ID', 'GT/Auto'],value_vars=list(range(0,200)), ignore_index=False)
            _tmp_df = _tmp_df.rename(columns = {'variable':'point', 'value':'distance'})
            a = _tmp_df.loc[_tmp_df['GT/Auto']=='GT','distance'].to_numpy()
            dangerous_points = _tmp_df[(_tmp_df['GT/Auto'] == 'GT') & (_tmp_df['point'] >= 0) & (_tmp_df['point'] <= 100)][
             'point'].to_numpy()
            dangerous_points = np.unique(dangerous_points)
            a_dangerous = _tmp_df[(_tmp_df['GT/Auto'] =='GT') & (_tmp_df['point'].isin(dangerous_points))]['distance'].to_numpy()
            b = _tmp_df.loc[_tmp_df['GT/Auto']=='Auto', 'distance'].to_numpy()
            b_dangerous = _tmp_df[(_tmp_df['GT/Auto'] == 'Auto') & (_tmp_df['point'].isin(dangerous_points))]['distance'].to_numpy()
            mae = np.mean(np.abs(a-b))
            mae_100 = np.mean(np.abs(a_dangerous-b_dangerous))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
            sns.lineplot(data=_tmp_df, x ='point', y='distance', hue='GT/Auto')
            ax.vlines(x=0, color='r', ymin=0, ymax=100, linestyle='--')
            ax.vlines(x=100, color='r',ymin=0, ymax=100, linestyle='--')
            # ax.axhline(y=40, color='r', linestyle='--')
            # ax.fill_betweenx(y=np.linspace(0, 100, 100), x1=20, x2=100, where=((np.linspace(0, 100, 100)  <= 40)), color='r',
            #          alpha=0.2)
            plt.title(title + ' ' + _organ+ ' ' + _type, fontsize=30)
            plt.text(110, 75, f'MAE Error = {mae:0.3f} mm', fontsize=25)
            plt.text(30, 10, f'MAE Error= {mae_100:0.3f} mm', fontsize=25)
            plt.xlim(1, 200)

            plt.xticks([0,  20,  40, 60, 80, 100, 120, 140, 160,  180,  200])
            plt.ylim(1, 100)
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # plt.yticks(np.arange(1, 100, 10))

            plt.ylabel('Rim points to closest vessel distance; mm', fontsize=35)
            plt.xlabel('Rim point', fontsize=35)
            plt.xticks(fontsize=35)
            plt.yticks(fontsize=35)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc='upper right',
               fontsize=30)
            plt.show()
            _TGT_FIG = _TGT_path + '/GT_Predicted_Points_distance/40-80Points_40_MIN'
            os.makedirs(_TGT_FIG, exist_ok=True)
            fig.savefig(os.path.join(_TGT_FIG, title + '_' + _organ +'_' + _type+ '_GT_Predicted_distance_dangerous.png'))
  elif title == 'Institute 2':
      for _organ in ['artery', 'vein']:
           # ['GT', 'Auto']:
              _tmp_df = df.loc[(df['vessel/side'] == _organ)]  # & (_nara_DF['GT/AUto']==_type)
              _tmp_df = _tmp_df[[x for x in _tmp_df.columns if any([('index' in x) or ('GT' in x)])]]
              _tmp_df.columns = ['GT/Auto'] + [int(x.replace('index ', '')) for x in _tmp_df.columns if 'index' in x]
              _tmp_df.reset_index(inplace=True)
              _tmp_df = _tmp_df.melt(id_vars=['ID', 'GT/Auto'], value_vars=list(range(0, 200)), ignore_index=False)
              _tmp_df = _tmp_df.rename(columns={'variable': 'point', 'value': 'distance'})
              a = _tmp_df.loc[_tmp_df['GT/Auto'] == 'GT', 'distance'].to_numpy()
              dangerous_points = _tmp_df[
                  (_tmp_df['GT/Auto'] == 'GT') & (_tmp_df['point'] >= 20) & (_tmp_df['point'] <= 100) ][
                  'point'].to_numpy()
              dangerous_points = np.unique(dangerous_points)
              a_dangerous = _tmp_df[(_tmp_df['GT/Auto'] == 'GT') & (_tmp_df['point'].isin(dangerous_points))][
                  'distance'].to_numpy()
              b = _tmp_df.loc[_tmp_df['GT/Auto'] == 'Auto', 'distance'].to_numpy()
              b_dangerous = _tmp_df[(_tmp_df['GT/Auto'] == 'Auto') & (_tmp_df['point'].isin(dangerous_points))][
                  'distance'].to_numpy()
              mae = np.mean(np.abs(a - b))
              mae_100 = np.mean(np.abs(a_dangerous - b_dangerous))
              fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
              sns.lineplot(data=_tmp_df, x='point', y='distance', hue='GT/Auto')
              ax.vlines(x=0, color='r', ymin=0, ymax=100, linestyle='--')
              ax.vlines(x=100, color='r', ymin=0, ymax=100, linestyle='--')
              # ax.axhline(y=40, color='r', linestyle='--')
              # ax.fill_betweenx(y=np.linspace(0, 100, 100), x1=20, x2=100, where=((np.linspace(0, 100, 100) <= 40)),
              #                  color='r',
              #                  alpha=0.2)
              plt.title(title + ' ' + _organ , fontsize=30)
              plt.text(110, 75, f'MAE Error = {mae:0.3f} mm', fontsize=25)
              plt.text(30, 10, f'MAE Error= {mae_100:0.3f} mm', fontsize=25)
              plt.xlim(1, 200)

              plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
              plt.ylim(1, 100)
              plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

              # plt.yticks(np.arange(1, 100, 10))

              plt.ylabel('Rim points to closest vessel distance; mm', fontsize=35)
              plt.xlabel('Rim point', fontsize=35)
              plt.xticks(fontsize=35)
              plt.yticks(fontsize=35)
              handles, labels = plt.gca().get_legend_handles_labels()
              by_label = OrderedDict(zip(labels, handles))
              plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc='upper right',
                         fontsize=30)
              plt.show()
              _TGT_FIG = _TGT_path + '/GT_Predicted_Points_distance/40-80Points_40_MIN'
              os.makedirs(_TGT_FIG, exist_ok=True)
              fig.savefig(
                  os.path.join(_TGT_FIG, title + '_' + _organ + '_' + '_GT_Predicted_distance_dangerous.png'))
