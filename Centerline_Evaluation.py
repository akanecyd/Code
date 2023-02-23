import numpy as np
import matplotlib.pyplot as plt
import boundingbox as bb
import os
import tqdm
import time
import eval_vessel
from skimage.morphology import skeletonize
from utils import mhd
import pandas as pd
import sklearn.metrics
GT_data_root = '//Salmon/User/Chen/Vessel_data/revised_nerve/crop_hipjiont_75mm_ver3'
AUTO_data_root ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold'
_PATIENT_LIST = '//Salmon/User/Chen/Vessel_data/revised_nerve/caseid_list_nerves.txt'
_Affected_csv = 'D:/temp/visualization/combined_distance_affected_1.csv'
GT_centerline_root ='//Salmon/User/Chen/Vessel_data/revised_nerve/crop_hipjiont_75mm_ver3/Centerline'
os.makedirs(GT_centerline_root, exist_ok=True)
AUTO_centerline_root = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold/Centerline'
os.makedirs(AUTO_centerline_root, exist_ok=True)
_color = './structures_json/vessels.json'
with open(_PATIENT_LIST) as f:
    case_IDs = f.read().splitlines()

for case_ID in tqdm.tqdm(case_IDs):
    GT_files = os.path.join(GT_data_root, case_ID, 'revised-vessels_nerves_label.mhd')
    Pred_files = os.path.join(AUTO_data_root, case_ID+'-vessels_nerves_label.mhd')
    GT_labels , hdr = mhd.read(GT_files)
    Auto_labels,_ = mhd.read(Pred_files)
    Sekelton_vol = np.zeros_like(GT_labels)
    Auto_sekelton_vol = np.zeros_like(GT_labels)
    # for i in [1, 2, 3]:
    #     pre_label = GT_labels == i
    #     pre_label_skeleton = skeletonize(pre_label)
    #     Sekelton_vol[pre_label_skeleton == 255] = i
    # mhd.write(os.path.join(GT_centerline_root,'{}-vessels_nerves_center_label.mhd'.format(case_ID)),Sekelton_vol,
    #           header={'CompressedData': True,
    #           'ElementSpacing': hdr['ElementSpacing'],
    #           'Offset': hdr['Offset']})

    for j in [1, 2, 3]:
        pre_label = Auto_labels == j
        pre_label_skeleton = skeletonize(pre_label)
        Auto_sekelton_vol[pre_label_skeleton == 255] = j
    mhd.write(os.path.join(AUTO_centerline_root,'{}-vessels_nerves_center_label.mhd'.format(case_ID)),Auto_sekelton_vol,
              header={'CompressedData': True,
              'ElementSpacing': hdr['ElementSpacing'],
              'Offset': hdr['Offset']})


#