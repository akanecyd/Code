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

_nara = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold/seperation_left_right/Polygons/PolyDistances/Nara_18_index_rimpoints_to_vessels_distance.csv'
_nara_DF = pd.read_csv(_nara, header=0,index_col=0)
_osaka = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyDistances/Osaka_36_index_rimpoints_to_vessels_distance.csv'
_osaka_DF = pd.read_csv(_osaka, header=0,index_col=0)
_TGT_path = '//Salmon/User/Chen/Vessel_data/Points to rim distance Analyze'
Osaka_affection ='//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/Treat_side.csv'
_osaka_affect = pd.read_csv(Osaka_affection, header=0,index_col=0)
not_in_affect = _osaka_DF.index[~_osaka_DF.index.isin(_osaka_affect.index)].drop_duplicates().tolist()

print(_nara_DF)
print(_osaka_DF)

#%%

for df, title in zip([_nara_DF, _osaka_DF], ['Nara', 'Osaka']):
    for _organ in ['artery_right', 'artery_left', 'vein_right', 'vein_left']:
        # for _type in ['Auto']:#['GT', 'Auto']:
            _tmp_df = df.loc[(df['vessel/side']==_organ)] #& (_nara_DF['GT/AUto']==_type)
            _tmp_df = _tmp_df[[x for x in _tmp_df.columns if any([('index' in x) or ('GT' in x)])]]
            _tmp_df.columns = ['GT/Auto'] + [int(x.replace('index ', '')) for x in _tmp_df.columns if 'index' in x]
            _tmp_df.reset_index(inplace=True)
            _tmp_df = _tmp_df.melt(id_vars=['ID', 'GT/Auto'],value_vars=list(range(0,200)), ignore_index=False)
            _tmp_df = _tmp_df.rename(columns = {'variable':'point', 'value':'distance'})
            a = _tmp_df.loc[_tmp_df['GT/Auto']=='GT','distance'].to_numpy()
            dangerous_points = _tmp_df[(_tmp_df['GT/Auto'] == 'GT') & (_tmp_df['point'] >= 20) & (_tmp_df['point'] <= 100)& (_tmp_df['distance'] <= 40)][
             'point'].to_numpy()
            dangerous_points = np.unique(dangerous_points)
            a_dangerous = _tmp_df[(_tmp_df['GT/Auto'] =='GT') & (_tmp_df['point'].isin(dangerous_points))]['distance'].to_numpy()
            b = _tmp_df.loc[_tmp_df['GT/Auto']=='Auto', 'distance'].to_numpy()
            b_dangerous = _tmp_df[(_tmp_df['GT/Auto'] == 'Auto') & (_tmp_df['point'].isin(dangerous_points))]['distance'].to_numpy()
            mae = np.mean(np.abs(a-b))
            mae_100 = np.mean(np.abs(a_dangerous-b_dangerous))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
            sns.lineplot(data=_tmp_df, x ='point', y='distance', hue='GT/Auto')
            ax.vlines(x=20, color='r', ymin=0, ymax=100, linestyle='--')
            ax.vlines(x=100, color='r',ymin=0, ymax=100, linestyle='--')
            ax.axhline(y=40, color='r', linestyle='--')
            ax.fill_betweenx(y=np.linspace(0, 100, 100), x1=20, x2=100, where=((np.linspace(0, 100, 100)  <= 40)), color='r',
                     alpha=0.2)
            plt.title(title + ' ' + _organ, fontsize=30)
            plt.text(110, 75, f'MAE Error = {mae:0.3f} mm\nin total', fontsize=20)
            plt.text(30, 10, f'MAE Error= {mae_100:0.3f} mm\nin danger', fontsize=20)
            plt.xlim(1, 200)

            plt.xticks([0,  20,  40, 60, 80, 100, 120, 140, 160,  180,  200])
            plt.ylim(1, 100)
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # plt.yticks(np.arange(1, 100, 10))

            plt.ylabel('Rim points to closest vessel distance; mm', fontsize=30)
            plt.xlabel('Rim points index', fontsize=30)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc='upper right',
               fontsize=30)
            plt.show()
            _TGT_FIG = _TGT_path + '/GT_Predicted_Points_distance/40-80Points_40_MIN'
            os.makedirs(_TGT_FIG, exist_ok=True)
            fig.savefig(os.path.join(_TGT_FIG, title + '_' + _organ + '_GT_Predicted_distance_dangerous.png'))