import os
import glob

import numpy as np
import cc3d
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from joblib import Parallel, delayed
import tqdm
from utils import mhd

_N_JOBS = 5

# _ROOT = '//Conger/User/Databases/OsakaUniv/CT_database/merged_phases_mhd/Labels/20220411_Mazen_6layers/LR_Combined'
# _TGT = '//Conger/User/Databases/OsakaUniv/CT_database/merged_phases_mhd/Labels/20220411_Mazen_6layers/LR_Separated'
# _ROOT = 'X:/mazen/Segmentation/Codes/results/Nara/hip_muscles'
# _TGT = 'X:/mazen/Segmentation/Codes/results/Nara/hip_muscles/LR'
# _ROOT = 'X:/mazen/Segmentation/Codes/results/Osaka/hip_muscles/KCT_osaka_muscles_NMAR'
_ROOT = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold'
# _ROOT = 'X:/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_20_5fold_wo_muscles'
_TGT = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold/seperation_left_right'
# _TGT = '//Salmon/User/Chen/Vessel_data/revised_nerve/crop_hipjiont_75mm_ver3/seperation_left_right'
os.makedirs(_TGT, exist_ok=True)
# _IN_IMG_EXT  = 'muscles-vessels_nerves_label.mhd'
# _IN_IMG_EXT =  'k8892-vessels_label.mhd'


# _CASE_LIST = 'X:/mazen/Segmentation/Codes/results/Osaka/kct_idlist_1490.txt'
# _CASE_LIST = 'X:/mazen/Segmentation/Codes/datalists/osaka/unprocessed/caseid_list.txt'
# _CASE_LIST = 'X:/mazen/Segmentation/Codes/datalists/nara/radiology/caseid_list.txt'
# _CASE_LIST = 'X:/mazen/Segmentation/Codes/datalists/osaka/iwasa/iwasa_125.txt'
# _CASE_LIST = 'X:/mazen/Segmentation/Codes/datalists/osaka/NMAR/caseid_list.txt'
# _CASE_LIST = 'X:/mazen/Segmentation/Codes/datalists/osaka/vessels/caseid_list_vessels_20.txt'
_CASE_LIST = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
# _CASE_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
_EXC_LIST = None

# _keys = list(range(1, 26))
# _rng1, _rng2 = list(range(1, 51, 2)), \
#                list(range(2, 52, 2))
# _PROC_LEVELS = {_key: [val_1, val_2] for _key, val_1, val_2
#                 in zip(_keys, _rng1, _rng2)}
# print(_PROC_LEVELS)
# _UNPROC_LEVELS = {22:43}
# _TOTAL = 50

_keys = list(range(1, 5))
_rng1, _rng2 = list(range(1, 9, 2)), \
               list(range(2, 10, 2))
_PROC_LEVELS = {_key: [val_1, val_2] for _key, val_1, val_2
                in zip(_keys, _rng1, _rng2)}
print(_PROC_LEVELS)
_UNPROC_LEVELS = {0:0}
_TOTAL = 8
_CONN_LEVELS = [1]


def process_case(img, separator):
    # _IN_IMG_EXT = 'crop_vein_artery_bones_label.mhd'
    _IN_IMG_EXT = '{}-vessels_label.mhd'.format(img)
    print(_IN_IMG_EXT)
    _OUT_IMG_EXT =_IN_IMG_EXT.replace('.mhd', '_lr.mhd')
    # img_path = os.path.join(_ROOT,  _IN_IMG_EXT)
    img_path = os.path.join(_ROOT,img, _IN_IMG_EXT)
    # _fname = '%s_lr.mhd' % (os.path.splitext(os.path.basename(img_path))[0])
    # _fname = img+'_'+_OUT_IMG_EXT  # img+
    _fname =  _OUT_IMG_EXT
    # _TGT = os.path.join(_ROOT, img)
    # os.makedirs(os.path.join(_TGT,img), exist_ok=True)
    _out_path = os.path.join(_TGT, _fname)
    # _out_path = os.path.join(_TGT, _fname)
    print(_out_path)
    if True:  # not os.path.isfile(os.path.join(_out_path)):
        # try:
        separator.run(img_path, _out_path)
        # except Exception as e:
        #     print(e)


class Separator(object):
    '''
    A class for separating the labels in 
    a volume into left/right sides.
    '''

    def __init__(self, coord_th=256, area_th=150, big_island_th=0.75):
        self.coord_th = coord_th
        self.area_th = area_th
        self.big_island_th = big_island_th

    def _read_img(self, path):
        '''
        Read image
        '''
        _IMG, _HDR = mhd.read(path)
        _PROPS = {'es': _HDR['ElementSpacing'],
                  'dim': _HDR['DimSize'],
                  'offset': _HDR['Offset']}

        return _IMG, _PROPS

    # @staticmethod
    # def remove_noise_areas(arr, noise=0.05):
    #     max_area = np.max(arr[1:])
    #     idx_arr = [i+1 for i, x in enumerate (arr[1:]) if x/max_area > noise]
    #     return idx_arr

    # @classmethod
    # def _clean_label(self, labels):
    #     n = self.area_th
    #     area = np.bincount(labels.flat)
    #     if n==1:
    #         # 1: means except the area of 0, to find the max area
    #         labels = np.in1d(labels, np.argmax(area[1:])+1).reshape(labels.shape)
    #     elif n>1:
    #         labels = np.in1d(labels, area.argsort()[-n-1:-1][::-1]).reshape(labels.shape)
    #     elif n<1:
    #         indices = self.remove_noise_areas(area, noise=n)
    #         labels = np.in1d(labels, indices).reshape(labels.shape)
    #     N = len(np.bincount(labels.flat))
    #     return labels, N

    def _clean_label(self, lbl_img):
        # Measure properties of labeled image regions.
        props = regionprops(lbl_img)
        out_img = np.zeros_like(lbl_img)
        for i, prop in enumerate(props):
            area = prop.area
            # print('Cent %d is %.2f %.2f %.2f\n' % (i, cent[0],cent[1],cent[2] ))
            if area > self.area_th:
                out_img[lbl_img == (i + 1)] = 1
        lbl_img, N = cc3d.connected_components(out_img, return_N=True)
        return lbl_img, N

    def _process_big_island(self, label):
        pass

    def _process_single_comp(self, label):
        '''
        Process label image with a single component, assuming 2 exist
        '''
        distance = ndimage.distance_transform_edt(
            label)  # index of the closest background element is returned along the first axis of the result.
        local_maxi = peak_local_max(
            distance, indices=False,
            footprint=np.ones((15, 15, 15)),
            labels=label)
        #Find peaks in an image as coordinate list

        markers = ndimage.label(local_maxi)[0]
        ret = watershed(-distance, markers, mask=label)
        return ret

    def _divide_on_centroid(self, cent, idx, out_img, lbl_img, val):
        if cent[2] > self.coord_th:
            out_img[lbl_img == (idx + 1)] = _PROC_LEVELS[val][1]
            # print('%d component was set to %d, size %d' % (i+1, _PROC_LEVELS[_val][1], prop.area))
        else:
            out_img[lbl_img == (idx + 1)] = _PROC_LEVELS[val][0]
        return out_img

    def _process_multi_comp(self, lbl_img, out_img, val):
        '''
        Process label image with a single component, assuming 2 exist
        '''
        props = regionprops(lbl_img)
        #       If < div  --> right
        #       Else --> left
        # print('%s Components found for label %s.. in %s' % (len(props), _val, _img_name))

        # Process large islands
        total_area = np.count_nonzero(lbl_img > 0)
        for i, prop in enumerate(props, start=1):
            area = prop.area
            ratio = area / total_area
            if ratio > self.big_island_th:
                tmp_lbl_img = self._process_single_comp(lbl_img == i)
                lbl_img, N = cc3d.connected_components(tmp_lbl_img, return_N=True)
                props = regionprops(lbl_img)

        for i, prop in enumerate(props):
            cent = prop.centroid
            out_img = self._divide_on_centroid(cent, i, out_img, lbl_img, val)
            # print('Cent %d is %.2f %.2f %.2f\n' % (i, cent[0],cent[1],cent[2] ))

            # print('%d component was set to %d, size %d' % (i+1, _PROC_LEVELS[_val][0], prop.area))
        return out_img

    def _save_img(self, _img, _out_path, _es, _offset=[0, 0, 0]):
        '''
        Save image
        '''
        mhd.write(_out_path, _img,
                  header={'ElementSpacing': _es,
                          'Offset': _offset,
                          'CompressedData': True})

    def run(self, img_path, _out_path):
        # Read the volume
        print('Processing ', img_path)
        if not os.path.exists(_out_path):
            _img_name = os.path.basename(img_path)
            _img, _props = self._read_img(img_path)
            # Create processed/unprocessed labels arrays
            vals = np.setdiff1d(np.unique(_img), [0])
            #Find the set difference of two arrays. Return the unique values in ar1 that are not in ar2.

            print('Vals : ', vals)
            proc_vals = list(set(vals) - set(_UNPROC_LEVELS.keys()))
            print('Proc Vals : ', proc_vals)
            unproc_vals = [_val for _val in _UNPROC_LEVELS.keys() if _val in vals]
            print('Unproc Vals : ', unproc_vals)

            # Create out image
            out_img = np.zeros_like(_img)

            # Process each label
            for _val in proc_vals:
                #   Convert to binary volume
                tmp_img = _img == _val
                #   Perform connected component labeling
                lbl_img, N = cc3d.connected_components(tmp_img, return_N=True)
                # if N >1:
                #     lbl_img, N = self._clean_label(lbl_img)
                #     # self._save_img(lbl_img, _out_path, _props['es'], _offset=_props['offset'])
                #     # exit()
                #     if N > 2:
                #         total_area = np.count_nonzero(lbl_img>1)
                #         for i in range(2, N+1):
                #             tmp_area = np.count_nonzero(lbl_img==i)/total_area
                #             if tmp_area > self.big_island_th:
                #                 print('Big island detected (area %0.3f)! Dividing..' % (tmp_area))
                #                 tmp_lbl_img = self._process_single_comp(lbl_img==i)
                #                 lbl_img , N = cc3d.connected_components(tmp_lbl_img, return_N=True)

                #   Check every component within ranges [0]
                #       If single component, --> divide it
                #       Reperform connected component
                if N == 0:
                    print('No compnents found for label %s in %s' % (_val, _img_name))
                    pass
                # elif (N == 1): #and ((_val==1) or (_val==5))
                #     print('Single component found for label %s in %s' % (_val, _img_name))
                #     c=0
                #     while (N == 1) and (c<10):
                #         lbl_img = self._process_single_comp(lbl_img)
                #         lbl_img , N = cc3d.connected_components(lbl_img, return_N=True)
                #         c+= 1
                # else:
                #     with open('skipped_cases.log', 'a') as f:
                #         f.write('%s\n' % (img_path))
                #     continue
                #   Calculate the centroids
                out_img = self._process_multi_comp(lbl_img, out_img, _val)

            for val in unproc_vals:
                out_img = np.where(_img == val, _UNPROC_LEVELS[val], out_img)

            self._save_img(out_img, _out_path, _props['es'], _offset=_props['offset'])

            num_vals = len(np.setdiff1d(np.unique(out_img), [0]))
            if num_vals != _TOTAL:
                print('Missing labels:  %s\n' % (_out_path))
            print('Image %s saved' % (_out_path))
        else:
            pass
            # print('%s already exists' % (_out_path))


if __name__ == '__main__':
    with open(_CASE_LIST, 'r') as ff:
        _CASES = ff.readlines()
    _CASES = [_CASE.replace('\n', '') for _CASE in _CASES]
    # _CASES = ['N0018']
    if _EXC_LIST is not None:
        for _CASE in _EXC_LIST:
            _CASES.remove(_CASE)
    print(_CASES)

    separator = Separator(area_th=1, big_island_th=0.65)
    # Parallel(n_jobs=_N_JOBS)(delayed(process_case)(_CASE, separator) for _CASE in tqdm.tqdm(_CASES))
    for _case in _CASES:
        process_case(_case, separator)
