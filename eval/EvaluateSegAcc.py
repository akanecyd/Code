from utils import evaluate
import os
import numpy as np
import mhd
from joblib import Parallel, delayed
import pandas as pd
import glob
from utils.similarity_metrics import load_volume
# from utils.similarity_metrics import evaluate_labels


def combine_csvs(root, tag):
    files = sorted(glob.glob(os.path.join(root, '*{0}.csv'.format(tag))))
    csvs = []
    for file in files:
        csvs.append(pd.read_csv(file))
    print(csvs)
    csvs = pd.concat(csvs, axis=1).T
    return csvs

def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

def repl(x, rng_1, rng_2):
    if x is None: return x
    for (_from, _to) in zip(rng_1, rng_2):
        x = np.where(x == _from, _to, x) 
    return x

def crop(x, x_s, x_e, y_s, y_e):
    if x is None: return x
    x = np.asarray(x).swapaxes(_SLICE_AXIS, 0)
    x = x[:, x_s:x_e, y_s:y_e]
    x = x.swapaxes(0, _SLICE_AXIS)
    return x

def merge(x1,x2,axis=0):
    if x1 is None: return x1
    x = np.concatenate([x1, x2], axis=axis)
    return x

def divide(x, div, axis):
    '''
        Divides a batch into two parts by specific ratio
    attributes: 
        div: division ratio (0-1)
        axis: division axis (0 or 1)
    '''
    if x is None: return x,x
    _xs, _ys = x.shape[0], x.shape[1]
    if axis == 0:
        _div_ys = int(np.floor(_ys*div))
        x_1, x_2 = crop(x, 0, _xs, 0, _div_ys) ,\
                   crop(x, 0, _xs, _div_ys, _ys)
    elif axis == 1:
        _div_xs = int(np.floor(_xs*div))
        x_1, x_2 = crop(x, 0, _div_xs, 0, _ys),\
                   crop(x, _div_xs, _xs, 0, _ys)
    return [x_1, x_2]

_N_JOBS = 5
_ROOT_GT = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint'
# _ROOT_GT = 'X:/Chen/Vessel_data/OnlyVessels'
# _ROOT_GT = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/crop_2.0'
# _ROOT_SEG = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/muscle_preds/osaka_20'
# _ROOT_SEG = 'X:/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_20_wo_muscles_original20'
_ROOT_SEG = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold'
# _ROOT_SEG='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_womuscles_cropped'
# _ROOT_SEG='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/16_original_cropped'
# _ROOT_SEG = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000'
# _ROOT_SEG = 'X:/mazen/Segmentation/Codes/results/Nara/hip_vessels/20_wo_muscles_plain'
# _CASE_LIST = 'X:/mazen/Segmentation/Data/HipMusclesDataset/2.0.3/caseid_list_vessels_10.txt'
# _CASE_LIST = 'X:/mazen/Segmentation/Codes/datalists/nara/radiology/vessels/vessels_good_alignment_caseid_list.txt'
# _CASE_LIST = 'X:/mazen/Segmentation/Codes/datalists/osaka/vessels/caseid_list_16_original.txt'
# _CASE_LIST = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
# _CASE_LIST = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
# _CASE_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_hipjiont_75mm_ver3/caseid_list_vessels(with muscle)_20.txt'
# _CASE_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/only_16vessel_crop/caseid_list_vessels_16.txt'
# _CASE_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
_CASE_LIST = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
# _OUT_DIR = 'Z:/mazen/LowerLimbs/HipSegmetnation2/Analysis/Analysis_Revised_Hiasa'
# _CASE_LIST = _ROOT_GT + '/Patients_List.txt'
_OUT_DIR = _ROOT_SEG + '/Evaluations/Recrop_Accuracy'
os.makedirs(_OUT_DIR, exist_ok=True)

_SLICE_AXIS = 2

with open(_CASE_LIST, 'r') as ff:
    _CASES = ff.readlines()
_CASES = [_CASE.replace('\n', '') for _CASE in _CASES]
print(_CASES)

_LEFT_RIGHT_GT = {'status': False,
                  'div'   : 0.5,
                  'axis'  : 0,
                  'rng'   : [[list(range(2,22,1))+[22],
                              list(range(2,42,2))+[42]],
                             [list(range(2,22,1))+[22],
                              list(range(3,42,2))+[42]]
                            ]}

_LEFT_RIGHT_SEG ={'status': False,
                  'div'   : 0.5,
                  'axis'  : 0,
                  'rng'   : [[list(range(2,22,1))+[22],
                              list(range(2,42,2))+[42]],
                             [list(range(2,22,1))+[22],
                              list(range(3,42,2))+[42]]
                            ]}

def evaluate_case(_CASE, roi_num=None):
    # _GT_FNAME = _CASE + '_artery_label.mhd' #'original-mask-revised_2.mhd' _CASE + '*R%s*label_org.mhd' % roi_num
    _GT_FNAME = 'crop_vein_artery_bones_label.mhd' #'original-mask-revised_2.mhd' _CASE + '*R%s*label_org.mhd' % roi_num
    _SEG_FNAME =  _CASE + '-vessels_label.mhd' #  _CASE + '*R%s*label.mhd' % roi_num
    _TAG = ''
    
    print('Evaluating .... {0}'.format(_CASE))
    gt_path = glob.glob(os.path.join(_ROOT_GT,_CASE,_GT_FNAME))[0]
    _GT_LABEL, es_gt, offset = load_volume(gt_path, axis_order='xyz')
    print('GT Shape ', _CASE,  _GT_LABEL.shape)
    print('GT Spacing ', _CASE, es_gt)

    if _LEFT_RIGHT_GT['status']:
        _parts = divide(_GT_LABEL,
                        _LEFT_RIGHT_GT['div'],
                        _LEFT_RIGHT_GT['axis'])
        new_parts = []
        for i, (_part, _rng) in enumerate(zip(_parts, _LEFT_RIGHT_GT['rng'])):
            new_parts.append(repl(_part, sorted(_rng[0], reverse=True), 
                                         sorted(_rng[1], reverse=True)))
        _GT_LABEL = merge(new_parts[0], new_parts[1], axis=0)
        _GT_LABEL = _GT_LABEL.transpose(2,1,0).astype(np.uint8)
        mhd.write(os.path.join(_ROOT_SEG, _CASE, _CASE + '-muscle-label-LR-gt.mhd'),
                  _GT_LABEL, 
                  header={'ElementSpacing':es_gt,
                          'CompressedData':True})
        print('Ground-truth saved')
    print(os.path.join(_ROOT_SEG,_CASE,_SEG_FNAME))
    seg_path = glob.glob(os.path.join(_ROOT_SEG,_CASE,_SEG_FNAME))[0]
    _SEG_LABEL, es_seg, offset = load_volume(seg_path, axis_order='xyz')
    print('Seg Shape ',_CASE,  _SEG_LABEL.shape)
    print('Seg Spacing ',_CASE,  es_seg)

    if _LEFT_RIGHT_SEG['status']:
        _parts = divide(_SEG_LABEL,
                        _LEFT_RIGHT_SEG['div'],
                        _LEFT_RIGHT_SEG['axis'])
        new_parts = []
        for i, (_part, _rng) in enumerate(zip(_parts, _LEFT_RIGHT_SEG['rng'])):
            new_parts.append(repl(_part, sorted(_rng[0], reverse=True), 
                                         sorted(_rng[1], reverse=True)))
        _SEG_LABEL = merge(new_parts[0], new_parts[1], axis=0)
        _SEG_LABEL = _SEG_LABEL.transpose(2,1,0).astype(np.uint8)
        mhd.write(os.path.join(_ROOT_SEG, _CASE, _CASE + '-muscle-label-LR-gt.mhd'),
                  _GT_LABEL, 
                  header={'ElementSpacing':es_seg,
                          'CompressedData':True})
        print('Ground-truth saved')

    # Evaluate
    ASD, DC = evaluate(_SEG_LABEL,_GT_LABEL, es_gt, connectivity=3)
    print((_CASE, ASD, DC))

    with open(os.path.join(_OUT_DIR, '{0}_ASD{1}.csv'.format(_CASE, _TAG)),'w') as asd_f,\
         open(os.path.join(_OUT_DIR, '{0}_DC{1}.csv'.format(_CASE,_TAG)),'w') as dc_f:
            np.savetxt(asd_f, [_CASE, *ASD], delimiter=',', fmt='%s')
            np.savetxt(dc_f, [_CASE, *DC], delimiter=',', fmt='%s')

# Parallel(n_jobs=_N_JOBS)(delayed(evaluate_case)(_CASE) for _CASE in _CASES) 
for _CASE in _CASES:
    evaluate_case(_CASE)
for _METRIC in ['ASD', 'DC']:
    _TAG = ''
    csvs = combine_csvs(_OUT_DIR, _METRIC+'*'+_TAG)
    csvs.to_csv(os.path.join(_OUT_DIR,'{0}.csv'.format(_METRIC)), header=None)

