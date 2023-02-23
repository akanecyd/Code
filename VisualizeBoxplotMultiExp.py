import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_palette("Set2")
import matplotlib
import random
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import glob
import json

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)
replace_dict = {'N0018': '#2-1', 'N0024': '#2-2', 'N0047': '#2-3', 'N0056': '#2-4', 'N0074': '#2-5', 'N0076': '#2-6',
                'N0091': '#2-7', 'N0094': '#2-8', 'N0107': '#2-9', 'N0108': '#2-10', 'N0116': '#2-11', 'N0132': '#2-12',
                'N0133': '#2-13', 'N0140': '#2-14', 'N0144': '#2-15', 'N0171': '#2-16', 'N0180': '#2-17', 'N0187': '#2-18',
                'k10387': '#1-1', 'k7510': '#1-2', 'k8559': '#1-3', 'k8574': '#1-4', 'k8699': '#1-5', 'k8748': '#1-6',
                'k8772': '#1-7', 'k8895': '#1-8', 'k9020': '#1-9', 'k9089': '#1-10', 'k9162': '#1-11', 'k9193': '#1-12',
                'k9204': '#1-13', 'k9622': '#1-14', 'k9831': '#1-15', 'k9861': '#1-16', 'k1565': '#1-17', 'k1585': '#1-18',
                'k1631': '#1-19', 'k1657': '#1-20', 'k1665': '#1-21', 'k1677': '#1-22', 'k1712': '#1-23', 'k1756': '#1-24',
                'k1796': '#1-25', 'k1802': '#1-26', 'k1870': '#1-27', 'k1873': '#1-28', 'k1647': '#1-29', 'k6940': '#1-30',
                'k8041': '#1-31', 'k8454': '#1-32', 'k8795': '#1-33', 'k8892': '#1-34', 'k9086': '#1-35', 'k9339': '#1-36'}
def create_sub_img(root,name_list, ext='png'):
    imgs = []
    for _name in name_list:
        print(_name)
        img_file = glob.glob(os.path.join(root, '*%s*%s' % (_name, ext)))[0]
        im = plt.imread(img_file)
        imgs.append(im)
    return np.concatenate(imgs, axis=1)


def convert_pval_to_asterisk(p_val, th=0.05):
    if p_val > th:
        return 'n.s.'
    elif (p_val <= th):
        return '*'

def read_json (fpath):
    with open(fpath) as f:
        out_dict = json.load(f)
    return out_dict

class Configs(object):
    def __init__(self):
        self._ROOTS = {
            # '16 Iwasa cases':'//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/16_original_cropped/Evaluations/Accuracy',
            # '20 Osaka cases':'//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_womuscles_cropped/Evaluations/Accuracy',
            # '36 Osaka cases':'//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped/Evaluations/Accuracy',
            'Institute 1\n (N=36)': '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/Evaluations/Accuracy',
            # 'Institute 2\n (N=18) Only Test':'//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/21_vessels_bones_plain/Evaluations/Revised_Accuracy',
            'Institute 2\n (N=18)':'//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold/Evaluations/Recrop_Accuracy',
            # 'Institute 2\n Dilation (N=18) ': '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/21_vessels_bones_plain/Evaluations/Dilation_Accuracy',
            # 'Inst1 Test (N=16)\nWO Muscles': '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/crop_2.0/Evaluations/Accuracy',
            # 'Inst2 Test (N=17)\nWith Muscles': '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/crop_2.0/Evaluations/Accuracy',
            # 'Inst2 Test (N=17)\nWO Muscles': '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/crop_2.0/Evaluations/Accuracy',
            }
    # # ,
        # self._ROOTS = {
        #     'Institute2\n(N=17)':'//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/Evaluations/Accuracy'
        #     }
    # ,
            # 
        self._TAG = ''
        self._MEASURES = {
        #                     : {'name': 'ASD',
        # #                       'range': (0, 4.0),
        # #                       'cmap': 'jet',
        # #                       'ylabel': 'ASD [mm]'},
                          1: {'name': 'DC',
                              'range': (0.9, 1.0),
                              'cmap': 'jet_r',
                              'ylabel': 'DC'}}
        self._DIFF_MEASURES = {1: {'name': 'ASD',
                                   'range': (-0.5, 0.5),
                                   'cmap': 'jet'},
                               2: {'name': 'DC',
                                   'range': (-0.05, 0.05)},
                                   'cmap': 'jet_r'}
        self._STRUCT_PRESET = 'structures_json/Pelvis.json'
        self._PATIENT_LIST = None#'X:/mazen/Segmentation/Codes/datalists/nara/radiology/vessels/vessels_good_alignment_caseid_list.txt'
        # self._PATIENT_LIST = 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/Pass2/patient_list.txt'
        # self._DROP_PATIENTS = ['k10338', 'k10738','k6940', 
        #                        'k7449', 'k7699', 'k7759', 
        #                        'k7813', 'k8016', 'k8797', 
        #                        'k9804', 'k9869','k7142', 
        #                        'k7367','k8041', 'k8454', 
        #                        'k8572', 'k8748', 'k8795', 
        #                        'k9339', 'k10721']        
        # self._DROP_PATIENTS = ['k7142', 'k7367','k8041', 'k8454', 'k8572',
        #                        'k8748', 'k8795', 'k9339', 'k10721']
        # self._DROP_PATIENTS = ['k1565','k1585','k1631','k1636','k1647',
        #                        'k1657','k1665','k1677','k1680','k1712','k1756',
        #                        'k1796','k1802','k1807','k1828','k1864','k1870',
        #                        'k1873','k1964','k2006']
                        
        # self._DROP_PATIENTS = ['N0145']
        # self._DROP_STRUCTURES = None# only muscles
        # self._DROP_STRUCTURES = ['femur', 'pelvis', 'sacrum', 'adductor_muscles',
        #                          'biceps_femoris_muscle', 'gracilis_muscle',
        #                          'rectus_femoris_muscle', 'sartorius_muscle',
        #                          'semimembranosus_muscle', 'semitendinosus_muscle',
        #                          'tensor_fasciae_latae_muscle',
        #                          'vastus_lateralis_muscle_and_vastus_intermedius_muscle',
        #                          'vastus_medialis_muscle']# only hip muscles
        # self._DROP_STRUCTURES = ['femur', 'pelvis', 'sacrum', 'gluteus_maximus_muscle',
        #                          'gluteus_medius_muscle', 'gluteus_minimus_muscle',
        #                          'iliacus_muscle',
        #                          'obturator_externus_muscle', 'obturator_internus_muscle',
        #                          'pectineus_muscle', 'piriformis_muscle',
        #                          'psoas_major_muscle']# only hip muscles

        # self._DROP_STRUCTURES = ['femur', 'pelvis', 'sacrum']# bones
        self._ORDER = [0] #Only muscles
        # self._ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8,9] #Only hip muscles
        
        self._OUT_DIR = _TGT = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/Analysis_20230214'
        os.makedirs(self._OUT_DIR, exist_ok=True)

configs = Configs()
_STRUCT_DICT = read_json(configs._STRUCT_PRESET)['color']
_STRUCT_NAMES = [_STRUCT_DICT[c][1] for c in range(0,len(_STRUCT_DICT),1)]
print(_STRUCT_NAMES)
colors = {val[1]:np.array(val[2:5]) for val in _STRUCT_DICT}
print(_STRUCT_DICT)
_TMP_STRUCT_NAMES = _STRUCT_NAMES.copy()
_ADD_SUB_FIGURE = False

# if configs._DROP_STRUCTURES:
#     for _STRUCTURE in configs._DROP_STRUCTURES:
#         if _STRUCTURE in _TMP_STRUCT_NAMES:
#             _TMP_STRUCT_NAMES.remove(_STRUCTURE)

if configs._ORDER:
    print(_TMP_STRUCT_NAMES)
    print(configs._ORDER)
    _TMP_STRUCT_NAMES = [_TMP_STRUCT_NAMES[o] for o in configs._ORDER]
    colors = [list(colors.values())[o] for o in configs._ORDER]
print(colors)

exps = list(configs._ROOTS.keys())

# Boxplots
for _MEASURE in configs._MEASURES.values():
    _DFs_All = []
    for _ROOT_Dict in configs._ROOTS.items():
        print(_ROOT_Dict)
        _ROOT = _ROOT_Dict[1]
        _EXP = _ROOT_Dict[0]
        
        #Read the data
        _PATH = os.path.join(_ROOT, '*'+_MEASURE['name']+'*'+configs._TAG+'.csv')
        _FILE = glob.glob(_PATH)
        if _FILE:
            _FILE = _FILE[0]
            print(_FILE)
            _FILE_DATA = pd.read_csv(_FILE,usecols=[0,1,3,4])
            _FILE_DATA.replace(replace_dict, inplace=True)

            _FILE_DATA = _FILE_DATA.set_index('PatientID')
            if configs._PATIENT_LIST:
                print(configs._PATIENT_LIST)
                _IDs = np.genfromtxt(configs._PATIENT_LIST, dtype=str).tolist()
                print(_IDs)
                # _FILE_DATA = _FILE_DATA.reindex(_IDs)
                _FILE_DATA = _FILE_DATA.loc[_IDs,:]

            # _IDs = _FILE_DATA['PatientID'].tolist()
            # if configs._DROP_PATIENTS:
            #     for _PAT in configs._DROP_PATIENTS:
            #         if _PAT in _FILE_DATA.index.tolist():
            #             _FILE_DATA = _FILE_DATA.drop(_PAT)
            #             # _IDs.remove(_PAT)
            #
            # if configs._DROP_STRUCTURES:
            #     _FILE_DATA = _FILE_DATA.drop(configs._DROP_STRUCTURES, axis=1)
            
            _FILE_DATA = _FILE_DATA[[*_TMP_STRUCT_NAMES]]

            _DF_Melted = pd.melt(_FILE_DATA, var_name=['Structure'],
                                             value_name=_MEASURE['name'], 
                                             ignore_index=False)
            _DF_Melted['Exp'] = _EXP
            _DFs_All.append(_DF_Melted)
            print(_DF_Melted)
    _DFs_All = pd.concat(_DFs_All)
    print(_DFs_All)


    #Plot figures
    if _ADD_SUB_FIGURE:
        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(4,10))
        sub_fig = create_sub_img('./muscle_sub_figures', _TMP_STRUCT_NAMES)
        axs[0].imshow(sub_fig)
        axs[0].axis('off')
        main_axs = axs[1]
    else:
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(4,10))
        main_axs = axs
    palette = {'Institute 1\n (N=36)': 'white',
               'Institute 2\n (N=18)': 'aquamarine'}
    sns.boxplot(data=_DFs_All, x = 'Structure', y=_MEASURE['name'], hue='Exp', ax=main_axs,
                showmeans=True,showfliers = False,palette = palette,linewidth=2.5,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue",
                          "markersize":2.0 })
    for i, box in enumerate(axs.artists):
        box.set_edgecolor('black')
    rcolors = [i.get_facecolor() for i in axs.get_children() if isinstance(i, matplotlib.patches.PathPatch)]
    row_colors = list(map(lambda x: x[:-1], rcolors))[0:4]
    # g = sns.FacetGrid(data=_DFs_All, col='Structure', hue='Exp', showmeans=True, showfliers=False,
    #                   meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue",
    #                              "markersize": 2.0})
    # g.map(sns.boxplot, y=_MEASURE['name'], x='Exp')
    #
    # box_xcoords = [box.get_x() + box.get_width() / 2 for ax in g.axes.flatten() for box in ax.artists]

    sns.stripplot(data=_DFs_All, x = 'Structure', y=_MEASURE['name'],hue='Exp',color='k',size=10,
                  jitter=True, dodge=True, marker='.',edgecolor='k', alpha=0.8, ax=main_axs)
    _X_TICKS = main_axs.get_xticklabels()
    print(_X_TICKS)
    _X_TICKS = list(map(lambda x: x.get_text().replace('_muscle', '').title(), _X_TICKS))
    _X_TICKS = list(map(lambda x: x.replace('_', '\n'), _X_TICKS))
    print(_X_TICKS)
    plt.xticks(np.arange(0, len(_TMP_STRUCT_NAMES), 1),
               _X_TICKS,rotation=30,fontsize=16, horizontalalignment='right')
    plt.ylabel(_MEASURE['ylabel'],fontsize=16)
    max_val = np.max(_DFs_All[_MEASURE['name']])
    # axs[1].set_ylim([0, max_val + max_val*0.25])
    main_axs.set_ylim(_MEASURE['range'])
    # axs[1].set_yscale('log')
    
    # Add p-values
    # if len(configs._ROOTS.keys())==3:
    #     for _x, _STRUCT in enumerate(_TMP_STRUCT_NAMES):
    #         sub_df =  _DFs_All[_DFs_All['Structure']==_STRUCT]
    #         print(sub_df[sub_df['Exp']==exps[1]][_MEASURE['name']].values)
    #         print(sub_df[sub_df['Exp']==exps[2]][_MEASURE['name']].values)
    #
    #         [u_test_val, p_val] = ttest_rel(sub_df[sub_df['Exp']==exps[1]][_MEASURE['name']].values,
    #                                         sub_df[sub_df['Exp']==exps[2]][_MEASURE['name']].values,
    #                                         nan_policy='omit')
    #         print(p_val)
    #         # _y = np.max(sub_df[_MEASURE['name']].values)
    #         _y = _MEASURE['range'][1] - 0.1*_MEASURE['range'][1]
    #         if _y > _MEASURE['range'][1]:
    #             _y = _MEASURE['range'][1] - 0.1*_MEASURE['range'][1]
    #         # p_str = convert_pval_to_asterisk(p_val)
    #         # main_axs.text(_x, _y+_y*0.05, '%s' % (p_str),
    #         #         color='red' if p_str!='n.s.' else 'black')
    #         main_axs.text(_x-0.05, _y+0.05, 'P: %0.3f' % (p_val),
    #             color='red' if p_val<0.05 else 'black',fontsize=8)
    # Add outliers
    # _struct = ['vein', 'artery']
    # for _x, _STRUCT in enumerate(_struct):
    #     sub_df = _DFs_All[_DFs_All['Structure'] == _STRUCT]
    #     Inst1_df = sub_df[sub_df['Exp'] == exps[0]]
    #     outliers = Inst1_df[(Inst1_df[_MEASURE['name']] < 0.65) | (Inst1_df[_MEASURE['name']] == max(Inst1_df[_MEASURE['name']]))]
    #     IDs = outliers.index
    #     for i, outlier in enumerate(IDs):
    #         axs.annotate(outlier,   xy=(_x-0.2 , outliers.iloc[i, 1]+random.uniform(0, 0.01)),
    #                                 color='b', fontsize=15)
    #     Inst2_df = sub_df[sub_df['Exp'] == exps[1]]
    #     outliers = Inst2_df[(Inst2_df[_MEASURE['name']] < 0.65) | (Inst2_df[_MEASURE['name']] == max(Inst2_df[_MEASURE['name']]))]
    #     IDs = outliers.index
    #     for i, outlier in enumerate(IDs):
    #         axs.annotate(outlier, xy=(_x + 0.2, outliers.iloc[i, 1] + random.uniform(-0.01, 0.01)),
    #                      color='b', fontsize=15)

        # Find the maximum value
        # max_value = np.max(Inst_df.values)
        # Combine the outliers and the maximum value
        # outliers = np.append(outliers, max_value)



    cell_text = []
    for _EXP in configs._ROOTS.keys():
        sub_dat = _DFs_All[_DFs_All['Exp']==_EXP]
        if _MEASURE['name'] == 'ASD':
           row = ['%.3f±%.3f mm' % (sub_dat[sub_dat['Structure']==_STRUCT].mean(skipna=True),
                                        sub_dat[sub_dat['Structure']==_STRUCT].std(skipna=True)) 
                                        for _STRUCT in _TMP_STRUCT_NAMES]
        else:
            row = ['%.3f±%.3f' % (sub_dat[sub_dat['Structure'] == _STRUCT].mean(skipna=True),
                                  sub_dat[sub_dat['Structure'] == _STRUCT].std(skipna=True))
                   for _STRUCT in _TMP_STRUCT_NAMES]
        cell_text.append(row)
    print(cell_text)
    the_table = plt.table(cellText=cell_text,
                      rowLabels=list(configs._ROOTS.keys()),
                      rowColours=row_colors,
                      colColours=colors,
                      colLabels=_X_TICKS,
                      rowLoc='center',
                      loc='bottom',
                      colLoc='center')
    # the_table.auto_set_font_size(False)
    # the_table.set_fontsize(14)
    main_axs.set_xticklabels('')
    main_axs.set_xlabel('')
    the_table.scale(1, 3.5)

    # Add figure
    plt.legend([],[], frameon=False)
    plt.subplots_adjust(left=0.1, bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(configs._OUT_DIR, _MEASURE['name']+'_Osaka_Nara_Pelvis_MultiBoxplot.png'), dpi=300)
    plt.show()
    plt.close()


            