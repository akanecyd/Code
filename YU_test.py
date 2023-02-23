from math import atan2
import seaborn as sns
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from pickle import TRUE
from decimal import Decimal
import vtk
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import vis_utils
from utils.VTK import VTK_Helper
import logging
from Distance_Risk_score import distance_risk_score
import json
from joblib import delayed,Parallel
import pandas as pd
import tqdm
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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

def read_datalist(fpath, root = None, ext=None,field='fileID'):
    datalist = []
    if fpath.endswith('.txt'):
        datalist = np.genfromtxt(fpath, dtype=str)
    elif fpath.endswith('.csv'):
        df = pd.read_csv(fpath,header=0)
        # print('Dataframe: ', df)
        datalist = df[field].values.tolist()
    if root:
        datalist = ['%s/%s' % (root,item) for item in datalist]
    if ext:
        datalist = ['%s/%s' % (item,ext) for item in datalist]
    print(datalist)
    return datalist

#
from math import atan2


def sort_points_clockwise(points):
    # Find start point
    def z_key(point):
        x, y, z = point
        return z
    start_point = min(points, key=z_key)

    # Find center point
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]
    x_center = sum(x_coords) / len(x_coords)
    y_center = sum(y_coords) / len(y_coords)
    z_center = sum(z_coords) / len(z_coords)
    center_point = (x_center, y_center, z_center)

    # Calculate angles and sort points
    point_info = [(point, i) for i, point in enumerate(points)]
    def angle_key(point_info):
        point, i = point_info
        x, y, z = point
        angle = atan2(y - y_center, x - x_center)
        return angle

    sorted_points_info = sorted(point_info, key=angle_key)
    sorted_points = [point for point, i in sorted_points_info]
    original_indices = [i for point, i in sorted_points_info]
    sorted_points = np.array(sorted_points)
    # Reorder points so that the start point is first
    # low = np.argwhere(sorted_points == start_point)[0][0]
    start_index = np.argwhere(sorted_points == start_point)[0][0]
    # sorted_points = sorted_points[start_index:] + sorted_points[:start_index]
    original_indices = original_indices[start_index:] + original_indices[:start_index]

    return original_indices


def sort_points_anti_clockwise(points):
    # Find start point
    def z_key(point):
        x, y, z = point
        return z

    start_point = min(points, key=z_key)

    # Find center point
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]
    x_center = sum(x_coords) / len(x_coords)
    y_center = sum(y_coords) / len(y_coords)
    z_center = sum(z_coords) / len(z_coords)
    center_point = (x_center, y_center, z_center)

    # Calculate angles and sort points
    point_info = [(point, i) for i, point in enumerate(points)]

    def angle_key(point_info):
        point, i = point_info
        x, y, z = point
        angle = atan2(y - y_center, x - x_center)
        return -angle

    sorted_points_info = sorted(point_info, key=angle_key)
    sorted_points = [point for point, i in sorted_points_info]
    original_indices = [i for point, i in sorted_points_info]
    sorted_points = np.array(sorted_points)
    # Reorder points so that the start point is first
    # low = np.argwhere(sorted_points == start_point)[0][0]
    start_index = np.argwhere(sorted_points == start_point)[0][0]
    # sorted_points = sorted_points[start_index:] + sorted_points[:start_index]
    original_indices = original_indices[start_index:] + original_indices[:start_index]

    return original_indices

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res
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

if __name__ == '__main__':
  _Rim_ROOT ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/acetablum_points'
      # '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold/seperation_left_right/Polygons/rim_points'
      # '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/acetablum_points'
  # '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint/seperation_left_right/Polygons/rim_points'
  # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/acetablum_points'
  _CASE_LIST ='//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
  # '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
  # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'

  GT_distance_path ='//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/PolyDistances_20230214'
      # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/PolyDistances'
      # '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint/seperation_left_right/Polygons/PolyDistances'
  Auto_distance_path ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyDistances_20230214'
      # '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold/seperation_left_right/Polygons/PolyDistances'
      # '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyDistances
  replace_dict = {'N0018': '#2-1', 'N0024': '#2-2', 'N0047': '#2-3', 'N0056': '#2-4', 'N0074': '#2-5', 'N0076': '#2-6',
                  'N0091': '#2-7', 'N0094': '#2-8', 'N0107': '#2-9', 'N0108': '#2-10', 'N0116': '#2-11',
                  'N0132': '#2-12',
                  'N0133': '#2-13', 'N0140': '#2-14', 'N0144': '#2-15', 'N0171': '#2-16', 'N0180': '#2-17',
                  'N0187': '#2-18',
                  'k10387': '#1-1', 'k7510': '#1-2', 'k8559': '#1-3', 'k8574': '#1-4', 'k8699': '#1-5', 'k8748': '#1-6',
                  'k8772': '#1-7', 'k8895': '#1-8', 'k9020': '#1-9', 'k9089': '#1-10', 'k9162': '#1-11',
                  'k9193': '#1-12',
                  'k9204': '#1-13', 'k9622': '#1-14', 'k9831': '#1-15', 'k9861': '#1-16', 'k1565': '#1-17',
                  'k1585': '#1-18',
                  'k1631': '#1-19', 'k1657': '#1-20', 'k1665': '#1-21', 'k1677': '#1-22', 'k1712': '#1-23',
                  'k1756': '#1-24',
                  'k1796': '#1-25', 'k1802': '#1-26', 'k1870': '#1-27', 'k1873': '#1-28', 'k1647': '#1-29',
                  'k6940': '#1-30',
                  'k8041': '#1-31', 'k8454': '#1-32', 'k8795': '#1-33', 'k8892': '#1-34', 'k9086': '#1-35',
                  'k9339': '#1-36'}
  _TGT = Auto_distance_path
  _cases = read_datalist(_CASE_LIST)
  _cases_new_rule = np.array([replace_dict.get(item, item) for item in _cases])
  point =[]
  # _cases = ['k1870']
  df = pd.DataFrame()
  df['ID'] = []
  df['vessel/side'] = []
  df['GT/AUto'] = []

  E_df = pd.DataFrame()
  E_df['ID'] = []
  E_df['vessel/side'] = []
  E_df ['Error ; mm'] = []
  E_df ['Uncertainty']= []
  E_df['GT Distance; mm'] = []
  E_df['Predicted Distance; mm'] = []

 # Define the remaining 200 columns
  for i in range(200):
      df[f'index {(i) }'] = []
  for _organ1, _organ2 in zip(('artery', 'vein'), ('vein', 'artery')):
      _sides = ['left', 'right']
      for i, _side in enumerate(['right', 'left']):
          # _TGT_STRUCT = 'rim_%s' % _side
          _RIM_STRUCT = 'rim_%s' % _side
          _COMP_STRUCT = '%s_%s' % (_organ2, _side)
          # _COMP_STRUCT = '%s_%s' % (_organ2, _side)
          _SRC_STRUCT = '%s_%s' % (_organ1, _side)
          _opo_pelvis = 'pelvis_%s' % _sides[i]
          # tag = '-vessels_label_lr'
          tag = ''
          print(_COMP_STRUCT)
          print(_SRC_STRUCT)
          # _cases = ['k10387']
          for j,_case in enumerate(tqdm.tqdm(_cases)):
      # _SRC = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _SRC_STRUCT))
      # _TGT = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))

      # _SRC = os.path.join(_SEG_POLY_ROOT, '%s%s_%s_.vtk' % (_case,tag, _SRC_STRUCT))
      # vessel(vein/artery)_side
      # _SRC = os.path.join(_GT_POLY_ROOT, '%s%s_%s.vtk' % (_case, tag, _SRC_STRUCT))
      # # rim_side
      # _TGT = os.path.join(_Pelvis_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))
           _RIM = os.path.join(_Rim_ROOT, '%s_%s.vtp' % (_case, _RIM_STRUCT))

           _RIM_reader = vtk.vtkXMLPolyDataReader()

  # Set the file name for the reader
           _RIM_reader.SetFileName(_RIM)

  # Read the file
           _RIM_reader.Update()

  # Get the points from the reader's output
           points = _RIM_reader.GetOutput().GetPoints().GetData()
           points = VTK_Helper.vtk_to_numpy(points)
           if _side =='right':
             indices = sort_points_anti_clockwise(points)
           elif _side =='left':
             indices = sort_points_clockwise(points)
           # else:
           #    indices = change_point_order(points, if_reverse=False)
           print(max(indices),min(indices))
           GT_distance_csv = os.path.join(GT_distance_path,'%s_%s.csv' % (_case, _SRC_STRUCT))
           Auto_distance_csv = os.path.join(Auto_distance_path, '%s_%s.csv' % (_case, _SRC_STRUCT))
           GT_points_distance = pd.read_csv(GT_distance_csv).to_numpy()
           Auto_points_distance = pd.read_csv(Auto_distance_csv).to_numpy()
           # newarr = np.array_split(Point_mae_uncertainty, 200)

           GT_points_distance_order = [GT_points_distance [indice] [0] for indice in indices]
           Auto_points_distance_order = [Auto_points_distance [indice] [0] for indice in indices]
           total_error = [x - y for x, y in zip(GT_points_distance_order, Auto_points_distance_order)]


           # row = list([_case, 'GT',GT_points_distance_order])
           # df= df.append(row)
           index_dict = {f'index {j}': GT_points_distance_order[j] for j in range(len(GT_points_distance_order))}
           info_dict = {'ID': _case, 'vessel/side': _SRC_STRUCT,'GT/AUto': 'GT'}
           GT_DICT = Merge(info_dict, index_dict)
           df = df.append(GT_DICT ,ignore_index=True)
           # # df = df.append({'ID': _case, 'vessel/side': _SRC_STRUCT, 'GT/AUto': 'Auto',
           #                     f'index {(j)}': Auto_points_distance_order[j]}, ignore_index=True)
           index_dict = {f'index {j}': Auto_points_distance_order[j] for j in range(len(Auto_points_distance_order))}
           info_dict = {'ID': _case, 'vessel/side': _SRC_STRUCT, 'GT/AUto': 'Auto'}
           AUTO_DICT = Merge(info_dict, index_dict)
           df = df.append(AUTO_DICT, ignore_index=True)

           order = np.arange(1,201,1)
           fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 30))


      # Plot the first curve
           ax.plot(order, GT_points_distance_order, label='GT minimum distance',linewidth = 3)

      # Plot the second curve
           ax.plot(order, Auto_points_distance_order, label='Predicted minimum distance',linewidth = 3)
           ax.set_facecolor('w')
           plt.title('%s_%s' % (_case, _SRC_STRUCT), fontsize=45)

           plt.xlim(1, 200)

           plt.xticks([0,20,40,60,80,100,120,140,160,180,200])
           # plt.ylim(1, 100)
           ax.set_ylim(1, 100)


           # plt.yticks(np.arange(1, 100, 10))



           ax.set_ylabel('Rim points to closest vessel distance; mm', fontsize=45)
           ax.set_xlabel('Rim points index', fontsize=40)
           for label in ax.get_yticklabels():
             label.set_fontsize(40)
           for label in ax.get_xticklabels():
              label.set_fontsize(40)
           # plt.xticks(fontsize=40)
           # ax.set_yticks(fontsize=45)
           handles, labels = plt.gca().get_legend_handles_labels()
           ax.legend(loc='upper left',fontsize=40)
           # by_label = OrderedDict(zip(labels, handles))
           # plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc='upper right',
           #       fontsize=30)
           error = []
           Point_mae_uncertainty = []
           GT_diastance = []
           Predicted_distance = []
           for i in range(0, 100):
            # if GT_points_distance_order[i] <= 40:
                   error.append(Auto_points_distance_order[i] - GT_points_distance_order[i])
                   GT_diastance.append(GT_points_distance_order[i])
                   Predicted_distance.append(Auto_points_distance_order[i])
                   # Point_mae_uncertainty = Point_mae_uncertainty_order[i]
                   # Point_mae_uncertainty.append(Point_mae_uncertainty)
           # error = [Auto_points_distance_order[i]-GT_points_distance_order[i] for i in range(20, 100) if GT_points_distance_order[i] < 50]
           # Point_mae_uncertainty = [Point_mae_uncertainty_order for i in range(20, 100) if GT_points_distance_order[i] < 50][:][0]
           abs_error = np.abs(error)
           mae = np.mean(np.abs(total_error))
           mae_risky = np.mean(abs_error)
           colors = np.random.rand(len(abs_error))
      # plt.scatter(x, y, )
           norm = mcolors.Normalize(vmin=0, vmax=0.03)
           ax.grid(True)
           ax.set_aspect('equal', 'box')
           ax.set_facecolor('w')
           title = _cases_new_rule[j]
           plt.title(title + '_' + _SRC_STRUCT + ' distance distribution', fontsize=40, fontweight='bold')
           plt.grid(True)
           ax.vlines(x=0, color='r', ymin=0, ymax=100, linestyle='--')
           ax.vlines(x=100, color='r', ymin=0, ymax=100, linestyle='--')
           # ax.axhline(y=40, color='r', linestyle='--')
           # ax.fill_betweenx(y=np.linspace(0, 100, 100), x1=20, x2=100, where=((np.linspace(0, 100, 100) <= 40)),
           #             color='r',
           #             alpha=0.2)
           plt.text(110, 75, f'MAE Error = {mae:0.3f} mm', fontsize=40)
           plt.text(30, 10, f'MAE Error= {mae_risky:0.3f} mm', fontsize=40)
      # plt.xlim(0, 200)
      #
      # plt.xticks(np.arange(0, 10, 200))
      #
      # plt.ylim(0, 100)
      # plt.yticks([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
           plt.ylabel('Rim points to closest vessel distance; mm', fontsize=45, fontweight='bold')
           plt.xlabel('Rim point', fontsize=45, fontweight='bold')
           handles, labels = plt.gca().get_legend_handles_labels()

           plt.xticks(fontsize=45)
           plt.yticks(fontsize=45)

           ratio = 1

      # get x and y limits
           x_left, x_right = ax.get_xlim()
           y_low, y_high = ax.get_ylim()

      # set aspect ratio
           ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
           plt.tight_layout()
           plt.show()

           _TGT_FIG = _TGT + '/Rim_distatnce'
           os.makedirs(_TGT_FIG, exist_ok=True)
           fig.savefig(os.path.join(_TGT_FIG, title + '_' + _SRC_STRUCT + '_' + '_GT_Predicted_distance_dangerous.png'))
           plt.close()

  df.to_csv(os.path.join(Auto_distance_path, 'Osaka36_index_rimpoints_to_vessels_distance.csv'))

      # cmap = plt.get_cmap('jet')

           # for i in range(0,len(error)):
           #     # E_df['Uncertainty'] = E_df['Uncertainty'].round(16)
           #
           #     s= pd.Series(float(np.format_float_positional(uncertainty[i], trim='-')))
           #     info_dict = {'ID': _case, 'vessel/side': _SRC_STRUCT, 'Error ; mm': round(abs_error[i], 8),
           #                  'Uncertainty':float(np.format_float_positional(s.values[0],precision=15)),
           #                  'GT Distance; mm': GT_diastance[i],'Predicted Distance; mm': Predicted_distance[i]}
           #     E_df = E_df.append(info_dict, ignore_index=True)

               # pd.options.display.precision = len(str(Point_mae_uncertainty_order[0]).split(".")[-1])
               # E_df = E_df.applymap(lambda x: format(x, '.8f'))
               # normal_float=np.format_float_positional(uncertainty[i], trim='-')
               # print(float(uncertainty[i]))
               # E_df['Uncertainty'] = normal_float
               # pd.options.display.float_format = '{:,.10f}'.format
           # print(Point_mae_uncertainty_order[100])

           # sns.scatterplot(data=E_df, x='Error ; mm', y="Uncertainty",
           #             palette='bright',
           #            alpha=0.6, s=90, linewidths=5)
           # Point_mae_uncertainty_order = Point_mae_uncertainty_order*100

           #dangerous zone
           # abs_error = abs_error[40: 80]
           # Point_mae_uncertainty_order = Point_mae_uncertainty_order[40: 80]
           # error = [Auto_points_distance_order[i] - GT_points_distance_order[i] for i in range(20, 100) if
           #     GT_points_distance_order[i] < 50]
           # abs_error = np.abs(error)
           # Point_mae_uncertainty = [Point_mae_uncertainty_order for i in range(20, 100) if GT_points_distance_order[i] < 50]
      #      plt.scatter(x=abs_error, y=Point_mae_uncertainty, c=Point_mae_uncertainty, cmap='jet', s=80, norm=norm)
      #      reg = LinearRegression().fit(abs_error.reshape(-1, 1), Point_mae_uncertainty)
      #      r, p = stats.pearsonr(abs_error, Point_mae_uncertainty)
      # # 绘制回归曲线
      #      plt.plot(abs_error, reg.predict(abs_error.reshape(-1, 1)), color='red',linewidth=2, linestyle='dashed')
      #      ax.annotate(' Uncertainty = {:.3f} error + {:.3f}\n      ρ = {:.3f}   p = {:e} '.format(reg.coef_[0], reg.intercept_, r, p),
      #             xy=(2.5, np.min(reg.predict(abs_error.reshape(-1, 1))) - 0.0005),
      #             color='g', fontsize=35, fontweight='bold')
      # 添加回归方程
      #      eq = "y = {:.3f}x + {:.3f}".format(reg.coef_[0], reg.intercept_)
      #      plt.text(0.5, 0.9, eq, fontsize=10, transform=plt.gcf().transFigure)
           # plt.scatter(x=abs_error, y=Point_mae_uncertainty_order, c=Point_mae_uncertainty_order, cmap='jet',s=80)
           # t = Point_mae_uncertainty_order
           # map1 = ax.imshow(np.stack([t, t]), cmap='viridis')
           # plt.colorbar()
           # plt.scatter(x=abs_error, y=Point_mae_uncertainty_order)
      # Dataset = pd.concat([_Artery["Distance_error"], _Artery["Dice"]], axis=1).to_xarray()
      # sns.jointplot(x='Distance_error', y='Dice', data=_Data,
      #               kind="hex", color="r",  # 主图为六角箱图
      #               size=6, space=0.1,
      #               joint_kws=dict(gridsize=20, edgecolor='w'),  # 主图参数设置
      #               marginal_kws=dict(bins=20, color='g',
      #                                 hist_kws={'edgecolor': 'k'}))  # 边缘图设置
      #                   # annot_kws=dict(stat='r', fontsize=15))
      # plt.show()

           # Error = I_df['Error ; mm']
           # DC = I_df['Uncertainty']
           # X_test, y_pred, p_value, Coefficients = plot_linear_regression_line(X=Error, Y=DC)
           # a = (y_pred[1] - y_pred[0]) / (X_test[1] - X_test[0])
           # b = y_pred[1] - X_test[1] * a
           # r, p = stats.pearsonr(Error, DC)
           # ax.annotate(' Uncertainty = {:.3f} error + {:.3f}\n      ρ = {:.3f}   p value = {:.3f} '.format(a[0], b[0], r,
           #                                                                                   p_value[1]),
           #        xy=(2.5, np.min(y_pred) - 0.0005),
           #        color='g', fontsize=35, fontweight='bold')
           # plt.plot(X_test, y_pred, color='g', linewidth=2, linestyle='dashed')



           # ax.text(110, 75, f'MAE Error = {mae:0.3f} mm\nin total', fontsize=20)
           # ax.text(10, 10, f'MAE Error= {mae_100:0.3f} mm\nin danger', fontsize=20, color='r')
           # ax.vlines(x=100, color='r', ymin=0, ymax=100, linestyle='--')
      #      ratio = 1
      #
      # # get x and y limits
      #      x_left, x_right = ax.get_xlim()
      #      y_low, y_high = ax.get_ylim()
      #
      # # set aspect ratio
      #      ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
           # plt.tight_layout()
           # plt.show()
           # _TGT_FIG = _TGT + '/GT_Predicted_Points_distance_UNCERT'
           # os.makedirs(_TGT_FIG, exist_ok=True)
           # fig.savefig(os.path.join(_TGT_FIG, _case+ '_'+ _SRC_STRUCT + '_GT_Predicted_distance_uncert.png'))

  # E_df.to_csv(os.path.join(_TGT_FIG, 'Error VS Uncertainty.csv'),index=False)
