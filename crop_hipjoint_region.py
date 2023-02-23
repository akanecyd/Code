from utils import mhd
from utils import ImageHelper
from utils.VideoHelper import VideoHelper
import glob
import os
import numpy as np
import tqdm
import pandas as pd
import json
from collections import OrderedDict
from joblib import Parallel, delayed


def findFiles(path):
    return glob.glob(path)


_MUSCLES_BONES_ROOT = '//Salmon/User/mazen/Segmentation/Data/HipMusclesDataset/2.0.5'
_VESSELS_ROOT = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/OnlyVessels'
# _hip_center = '//Salmon/User/Chen/Vessel_data/revised_nerve/20220511_HipCenter_Landmarks.csv'
_hip_center = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/20221110_HipCenter_Landmarks.csv'
_TGT = '//Salmon/User/Chen/Vessel_data/revised_nerve/crop_hipjiont_75mm_ver3'
_RESULT = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold'

AUTO_data_root ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_5fold'
AUTO_data_muscles_root ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold'
os.makedirs(_TGT, exist_ok=True)
_PATIENT_LIST = '//Salmon/User/Chen/Vessel_data/revised_nerve/caseid_list_nerves.txt'
with open(_PATIENT_LIST) as f:
    case_IDs = f.read().splitlines()
# case_IDs = ['k9339', 'k8795', 'k8892','k1647']
crop_list = [['Case_ID', 'hip_center', 'hip_max_z', 'hip_min_z']]
vessel_nerve_color = [[255, 0, 0], [0, 0, 255], [67, 211, 255]]
Predicted_color = [[196, 255, 196], [255, 0, 238], [147,20,255]]

with open('structures_json/muscle_vessel_nerve_color.json', 'r') as color_file:
    full_color = json.load(color_file)
full_color_list = np.array(full_color['color'])
full_color_list = np.array(full_color_list[:, 2:5], dtype=np.int)
full_color_list = full_color_list.tolist()

for i, case_ID in enumerate(tqdm.tqdm(case_IDs)):
    os.makedirs(os.path.join(_TGT, case_ID), exist_ok=True)
    Predicted_label, _ = mhd.read(os.path.join(_RESULT, case_ID + '-vessels_muscles_label.mhd'))
    skin_label, _ = mhd.read(os.path.join(_RESULT, case_ID + '-skin_label.mhd'))
    image_vol, hdr = mhd.read(os.path.join(_VESSELS_ROOT, case_ID, 'image.mhd'))
    hip_center = pd.read_csv(_hip_center, header=0, index_col=0)
    max_hip_center = max(np.round(hip_center.loc[case_ID.upper()]['femur_head_rt_z']),
                         np.round(hip_center.loc[case_ID.upper()]['femur_head_lt_z']))
    if i >= 13:
        max_hip_center = max_hip_center - hdr['Offset'][2]
    else:
        max_hip_center = max_hip_center - hdr['Offset'][2]
    _hip_max = np.int(max_hip_center + 75)
    _hip_min = np.int(max_hip_center - 75)
    crop_label = np.zeros_like(skin_label)
    crop_label[skin_label == 1] = 1
    mhd.write(os.path.join(_VESSELS_ROOT, case_ID, 'skin_label.mhd'),
              crop_label,
              header={'CompressedData': True,
                      'ElementSpacing': hdr['ElementSpacing'],
                      'Offset': hdr['Offset']})

    crop_list.append([case_ID, max_hip_center, _hip_max, _hip_min])
    # crop_out_label = muscle_bone_vol[_hip_min:_hip_max, :, :]
    mhd.write(os.path.join(_TGT, case_ID, 'label_ver2.mhd'),
              crop_out_label,
              header={'CompressedData': True,
                      'ElementSpacing': hdr['ElementSpacing'],
                      'Offset': [0, 0, np.array(hdr['Offset'][2]) + _hip_min]})
    mhd_paths = findFiles(os.path.join(_VESSELS_ROOT, case_ID, '*.mhd'))
    print(mhd_paths)
    for mhd_path in mhd_paths:
        mhd_name = os.path.basename(mhd_path)
        out_label, _ = mhd.read(mhd_path)
        print(mhd_name)
        crop_out_label = out_label[_hip_min:_hip_max, :, :]
        mhd.write(os.path.join(_TGT, case_ID, mhd_name),
                  crop_out_label,
                  header={'CompressedData': True,
                          'ElementSpacing': hdr['ElementSpacing'],
                          'Offset': [0, 0, np.array(hdr['Offset'][2]) + _hip_min]})
    crop_bones_label = Predicted_label[_hip_min:_hip_max, :, :]
    mhd.write(os.path.join(_TGT, case_ID, 'Predicted_label.mhd'),
              crop_bones_label,
              header={'CompressedData': True,
                      'ElementSpacing': hdr['ElementSpacing'],
                      'Offset': [0, 0, np.array(hdr['Offset'][2]) + _hip_min]})
    crop_skin_label = skin_label[_hip_min:_hip_max, :, :]
    crop_label = np.zeros_like(crop_skin_label)
    crop_label[crop_skin_label == 1] = 1
    # mhd.write(os.path.join(_TGT, case_ID, 'skin_label.mhd'),
    #          crop_label,
    #           header={'CompressedData': True,
    #                   'ElementSpacing': hdr['ElementSpacing'],
    #                   'Offset': [0, 0, np.array(hdr['Offset'][2]) + _hip_min]})

    # visulazation
    image_vol, _ = mhd.read(os.path.join(_TGT, case_ID, 'image.mhd'))
    vessel_nerve_label, _ = mhd.read(os.path.join(_TGT, case_ID, 'revised-vessels_nerves_label.mhd'))
    Predicted_nerve_label, _ = mhd.read(os.path.join(AUTO_data_root, case_ID + '-vessels_nerves_label.mhd'))
    Predicted_nerve_label_with_muscle, _ = mhd.read(os.path.join(AUTO_data_muscles_root, case_ID + '-muscles_vessels_nerves_label.mhd'))

    image_vol = ImageHelper.ImageHelper.normalize_hu(image_vol, dtype=np.uint8)
    # vessel_nerve_label = np.array(vessel_nerve_label, dtype=np.uint8)
    Predicted_nerve_label = np.array(Predicted_nerve_label, dtype=np.uint8)
    Predicted_nerve_label_with_muscle = np.array(Predicted_nerve_label_with_muscle, dtype=np.uint8)


    labeled_vol = ImageHelper.ImageHelper.label_images(images=image_vol,
                                                                   labels=vessel_nerve_label,
                                                                   colors=Predicted_color,
                                                                   thickness=1)


    labeled_vol_withMuscles = ImageHelper.ImageHelper.label_images(images=labeled_vol,
                                                       labels=Predicted_nerve_label_with_muscle,
                                                       colors=full_color_list,
                                                       thickness=1)
    labeled_vol_withoutMuscles = ImageHelper.ImageHelper.label_images(images=labeled_vol,
                                                       labels=Predicted_nerve_label,
                                                       colors=vessel_nerve_color,
                                                       thickness=1)
    movie_dir = os.path.join(_RESULT, 'movies')
    os.makedirs(movie_dir, exist_ok=True)
    VideoHelper.write_vol_to_video(vol=labeled_vol_withMuscles,
                                   case_name=case_ID,
                                   output_path=os.path.join(movie_dir,
                                                            "{}_Predicted_nerve_vessel_Muscle.mp4".format(case_ID)),
                                   if_reverse=True)

    VideoHelper.write_vol_to_video(vol=labeled_vol_withMuscles,
                                   case_name=case_ID,
                                   output_path=os.path.join(movie_dir,"{}_Predicted_nerve_vessel_Muscle.mp4".format(case_ID)),
                                   if_reverse=True)
    VideoHelper.write_vol_to_video(vol=labeled_vol_withoutMuscles,
                                   case_name=case_ID,
                                   output_path=os.path.join(movie_dir,
                                                            "{}_Predicted_nerve_vessel.mp4".format(case_ID)),
                                   if_reverse=True)

# my_df = pd.DataFrame(crop_list)
#
# my_df.to_csv(os.path.join(_TGT,'crop_hip_joint_osaka_20.csv'), index=False, header=False)
