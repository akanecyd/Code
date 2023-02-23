from utils import mhd
import os
import numpy as np
import tqdm
from collections import OrderedDict
from joblib import Parallel, delayed

_MUSCLES_BONES_ROOT = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/21_vessels_bones_plain'
_VESSELS_ROOT = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint'
# _VESSELS_ROOT = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/only_16vessel_crop'
_TGT = _VESSELS_ROOT
os.makedirs(_TGT, exist_ok=True)
_PATIENT_LIST = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
# _PATIENT_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/OnlyVessels/caseid_list_vessels_16.txt'
_N_JOBS = 5
import os
import shutil





def combine_source_target(src, tgt, rng):
    print(rng)
    assert len(rng[0])==len(rng[1]),'Ranges should be equal'
    for (_from, _to) in zip(rng[0], rng[1]):
        tgt = np.where(src==_from, _to, tgt)
    return tgt

def replace_masked_value(img,mask, _from, _to):
    out_img = np.zeros_like(img)
    out_img = np.where(mask!=_from, _to, img)
    return out_img

with open(_PATIENT_LIST, 'r') as ff:
    _CASES = ff.readlines()
_CASES = [_CASE.replace('\n', '') for _CASE in _CASES]
print(_CASES)
# _CASES = ['k9339', 'k8795', 'k8892','k1647','k8041','k8454','k6940','k9086']
_CASES = ['N0180']

_FILES = OrderedDict()
_FILES[0] = {'ext': 'crop_dilated_pelvis_femur_label.mhd',
             'rng': [[4],
                     [4]],
             'root': _VESSELS_ROOT}
_FILES[1] = {'ext':  '-vessels_label.mhd',
             'rng': [[1, 2],
                     [1, 2]],
             'root': _MUSCLES_BONES_ROOT}

_OUT_LBL_NAME = 'crop_dilated_artery_pelvis_femur_label.mhd'
    # 'muscles-vessels_nerves_label.mhd'
def process_case(_CASE):
    try:
        os.makedirs(os.path.join(_TGT, _CASE), exist_ok=True)
        if not os.path.isfile(os.path.join(_TGT, _CASE, _OUT_LBL_NAME)):
            print('Processing %s' % (_CASE))
            # img, hdr = mhd.read(os.path.join(_IMG_ROOT,_CASE+_IMG_NAME))
            _DICT = _FILES[0]
            lbl, hdr = mhd.read(os.path.join(_DICT['root'], _CASE,_DICT['ext']))
            out_label = np.zeros_like(lbl,dtype=np.uint8)
            for i in _FILES.keys():
                _DICT = _FILES[i]
                if i==1 :
                    _DICT['ext']= _CASE + _DICT['ext']
                lbl, hdr = mhd.read(os.path.join(_DICT['root'], _CASE,_DICT['ext']))
                out_label = combine_source_target(lbl, out_label, _DICT['rng'])
            mhd.write(os.path.join(_TGT, _CASE, _OUT_LBL_NAME),
                      out_label,
                      header={'CompressedData': True,
                              'ElementSpacing': hdr['ElementSpacing'],
                              'Offset': hdr['Offset']})

    except Exception as e:
        print('Case %s not processed! Error: \n%s' % (_CASE,e))
# process_case(0])
Parallel(n_jobs=_N_JOBS)(delayed(process_case)(_CASE) for _CASE in tqdm.tqdm(_CASES))
# for _CASE in _CASES:
#     process_case(_CASE)
