from utils import mhd
from utils import ImageHelper
from utils.VideoHelper import VideoHelper
import glob
import os
import numpy as np
import tqdm
import scipy.ndimage as ndimage
import pandas as pd
import json
from collections import OrderedDict
from joblib import Parallel, delayed


def findFiles(path):
    return glob.glob(path)


_MUSCLES_BONES_ROOT = '//Salmon/User/mazen/Segmentation/Data/HipMusclesDataset/2.0.5'
_VESSELS_ROOT = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment'
# _hip_center = '//Salmon/User/Chen/Vessel_data/revised_nerve/20220511_HipCenter_Landmarks.csv'
_hip_center = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/20221110_HipCenter_Landmarks.csv'
_TGT = '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint'
_RESULT = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels'

AUTO_data_root ='//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/21_vessels_bones_plain'
AUTO_data_muscles_root ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/20_w_rev_nerves_muscles_5fold'
os.makedirs(_TGT, exist_ok=True)
_PATIENT_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
# '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
# '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
with open(_PATIENT_LIST) as f:
    case_IDs = f.read().splitlines()
# case_IDs = ['k9339', 'k8795', 'k8892','k1647']
crop_list = [['Case_ID', 'hip_center', 'hip_max_z', 'hip_min_z']]
vessel_nerve_color = [[255, 0, 0], [0, 0, 255], [67, 211, 255]]
vessel_color = [[153, 255, 255], [153, 255, 255], [255, 255, 51], [0, 0, 255]]
Predicted_vessel = [[196, 255, 196], [255, 0, 238]]
Predicted_color = [[153, 255, 255], [153, 255, 255], [0, 128, 255], [51, 255, 51]]
artery_color = [[0, 0, 255]]
dilated_color = [[67, 211, 255]]
predicted_artery_color =[[0, 252, 124]]
# case_IDs = ['N0047']

# with open('structures_json/muscle_vessel_nerve_color.json', 'r') as color_file:
#     full_color = json.load(color_file)
# full_color_list = np.array(full_color['color'])
# full_color_list = np.array(full_color_list[:, 2:5], dtype=np.int)
# full_color_list = full_color_list.tolist()

for i, case_ID in enumerate(tqdm.tqdm(case_IDs)):
    # os.makedirs(os.path.join(_TGT, case_ID), exist_ok=True)
    # plain_ct, hdr = mhd.read(os.path.join(_VESSELS_ROOT, case_ID, 'crop_plain_ct_image.mhd'))
    # enhanced_ct, _ = mhd.read(os.path.join(_VESSELS_ROOT, case_ID, 'crop_enhanced_ct_image.mhd'))
    # # plain_ct, _ = mhd.read(os.path.join(_VESSELS_ROOT, case_ID, 'plain_ct_image.mhd'))
    # plain_artery_label, _ = mhd.read(os.path.join(_VESSELS_ROOT, case_ID, 'registration_enhanced_artery_label.mhd'))
    # # Predicted_label, _ = mhd.read(os.path.join(_RESULT, case_ID + '-vessels_muscles_label.mhd'))
    # # skin_label, _ = mhd.read(os.path.join(_RESULT, case_ID + '-skin_label.mhd'))
    # # # image_vol, hdr = mhd.read(os.path.join(_VESSELS_ROOT, case_ID, 'image.mhd'))
    # hip_center = pd.read_csv(_hip_center, header=0, index_col=0)
    # a= hip_center.loc[case_ID.upper()]['femur_head_rt_z'][1]
    # max_hip_center = max(np.round(hip_center.loc[case_ID.upper()]['femur_head_rt_z'][1]),
    #                      np.round(hip_center.loc[case_ID.upper()]['femur_head_lt_z'][1]))
    # if i >= 13:
    #     max_hip_center = max_hip_center - hdr['Offset'][2]-50
    # else:
    #     max_hip_center = max_hip_center - hdr['Offset'][2]-50
    # _hip_max = np.int(max_hip_center + 75)
    # _hip_min = np.int(max_hip_center - 75)
    # crop_skin_label = np.zeros_like(skin_label)
    # crop_skin_label[skin_label == 1] = 1
    # mhd.write(os.path.join(_VESSELS_ROOT, case_ID, 'skin_label.mhd'),
    #           crop_skin_label,
    #           header={'CompressedData': True,
    #                   'ElementSpacing': hdr['ElementSpacing'],
    #                   'Offset': hdr['Offset']})
    # crop_skin = crop_skin_label[_hip_min:_hip_max, :, :]
    # mhd.write(os.path.join(_TGT, case_ID, 'crop_skin_label.mhd'),
    #           crop_skin,
    #           header={'CompressedData': True,
    #                   'ElementSpacing': hdr['ElementSpacing'],
    #                   'Offset': [0, 0, np.array(hdr['Offset'][2]) + _hip_min]})
    #
    # crop_list.append([case_ID, max_hip_center, _hip_max, _hip_min])
    # crop_ct = enhanced_ct[_hip_min:_hip_max, :, :]
    # mhd.write(os.path.join(_TGT, case_ID, 'crop_enhanced_ct_image.mhd'),
    #           crop_ct,
    #           header={'CompressedData': True,
    #                   'ElementSpacing': hdr['ElementSpacing'],
    #                   'Offset': [0, 0, np.array(hdr['Offset'][2]) + _hip_min]})

    # crop_artery_label = plain_artery_label[_hip_min:_hip_max, :, :]
    # mhd.write(os.path.join(_TGT, case_ID, 'crop_plain_artery_label.mhd'),
    #           crop_artery_label,
    #           header={'CompressedData': True,
    #                   'ElementSpacing': hdr['ElementSpacing'],
    #                   'Offset': [0, 0, np.array(hdr['Offset'][2]) + _hip_min]})

    # dialate 3d label
    # vessel_label, hdr = mhd.read(os.path.join(_TGT, case_ID, 'crop_vein_artery_bones_label.mhd'))
    # # Predicted_vessel_label, _ = mhd.read(os.path.join(_TGT, case_ID, 'vein_label.mhd'))
    # Predicted_vessel_label, _ = mhd.read(os.path.join(AUTO_data_root, case_ID, case_ID + '-vessels_label.mhd'))
    # vessels_bones_label = np.zeros_like(Predicted_vessel_label)
    # vessels_bones_label[Predicted_vessel_label == 1] = 1
    # vessels_bones_label[Predicted_vessel_label == 2] = 2
    # vessels_bones_label[vessel_label == 1] = 3
    # vessels_bones_label[vessel_label == 2] = 4
    # # struct1 = ndimage.generate_binary_structure(3, 3)
    # # se = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    # #                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    # #                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    # # # struct2 =[[[True],[True],[True]]]
    # # dilate_vessel_label = ndimage.binary_dilation(vessel_label, structure=se).astype(vessel_label.dtype)
    #
    # mhd.write(os.path.join(_TGT, case_ID, 'crop_vein_artery_bones_label.mhd'),
    #           vessels_bones_label,
    #           header={'CompressedData': True,
    #                   'ElementSpacing': hdr['ElementSpacing'],
    #                   'Offset': hdr['Offset']})




    # visulazation
    _GT_path = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases'
        # '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint'
        # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases'
    _Auto_path = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000'
        # '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold'
        # '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000'

    _TGT = _Auto_path
    image_vol, _ = mhd.read(os.path.join(_GT_path, case_ID, 'image.mhd'))
    vessel_label, _ = mhd.read(os.path.join(_GT_path, case_ID, 'vessels_pelvis_femur_label.mhd'))
    # vessel_labels = np.zeros_like(vessel_label)
    # # vessel_labels[vessel_label == 3] = 1
    # # vessel_labels[vessel_label == 4] = 2
    Predicted_vessel_label, _ = mhd.read(os.path.join(_Auto_path, case_ID, case_ID + '-vessels_label.mhd'))
    # Predicted_vessel_label, _ = mhd.read(os.path.join(_TGT, case_ID, 'crop_vein_artery_bones_label.mhd'))
    # Predicted_artery = np.zeros_like(Predicted_vessel_label)
    # Predicted_artery[Predicted_vessel_label == 3] = 1
    # Predicted_artery[Predicted_vessel_label == 4] = 2
    # Predicted_nerve_label_with_muscle, _ = mhd.read(os.path.join(AUTO_data_muscles_root, case_ID + '-muscles_vessels_nerves_label.mhd'))

    image_vol = ImageHelper.ImageHelper.normalize_hu(image_vol, dtype=np.uint8)
    vessel_label = np.array(vessel_label, dtype=np.uint8)
    # dilate_vessel_label = np.array(dilate_vessel_label, dtype=np.uint8)
    Predicted_artery = np.array(Predicted_vessel_label, dtype=np.uint8)
    # Predicted_nerve_label = np.array(Predicted_nerve_label, dtype=np.uint8)
    # Predicted_nerve_label_with_muscle = np.array(Predicted_nerve_label_with_muscle, dtype=np.uint8)
    movie_dir = os.path.join(_TGT, 'movies')
    os.makedirs(movie_dir, exist_ok=True)
    VideoHelper.write_vol_to_video(vol=image_vol,
                                   case_name=case_ID,
                                   output_path=os.path.join(movie_dir,
                                                            "{}_plain_ct.mp4".format(case_ID)),
                                   if_reverse=True)
    labeled_vol = ImageHelper.ImageHelper.label_images(images=image_vol,
                                                                   labels=vessel_label,
                                                                   colors=vessel_color,
                                                                   thickness=1)

    Predicted_labeled_vol = ImageHelper.ImageHelper.label_images(images=labeled_vol,
                                                                 labels=Predicted_artery,
                                                                 colors=Predicted_color,
                                                                 thickness=1)
    VideoHelper.write_vol_to_video(vol=Predicted_labeled_vol,
                                   case_name=case_ID,
                                   output_path=os.path.join(movie_dir,
                                                            "GT_Predicted_{}_labeled_ct.mp4".format(case_ID)),
                                   if_reverse=True)

    # labeled_vol = ImageHelper.ImageHelper.label_images(images=labeled_vol,
    #                                                                labels=dilate_vessel_label,
    #                                                                colors= dilated_color,
    #                                                                thickness=1)

    # labeled_vol_withMuscles = ImageHelper.ImageHelper.label_images(images=labeled_vol,
    #                                                    labels=Predicted_nerve_label_with_muscle,
    #                                                    colors=full_color_list,
    #                                                    thickness=1)
    # labeled_vol_withoutMuscles = ImageHelper.ImageHelper.label_images(images=labeled_vol,
    #                                                    labels=Predicted_nerve_label,
    #                                                    colors=vessel_nerve_color,
    #                                                    thickness=1)

    # VideoHelper.write_vol_to_video(vol=labeled_vol_withoutMuscles,
    #                                case_name=case_ID,
    #                                output_path=os.path.join(movie_dir,
    #                                                         "{}_Predicted_nerve_vessel.mp4".format(case_ID)),
    #                                if_reverse=True)
#
# # my_df = pd.DataFrame(crop_list)
# #
# # my_df.to_csv(os.path.join(_TGT,'crop_hip_joint_osaka_20.csv'), index=False, header=False)
