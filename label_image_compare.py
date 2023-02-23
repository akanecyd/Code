import cv2
import numpy as np
from utils.ImageHelper import ImageHelper
from utils.VideoHelper import VideoHelper
from PIL import Image
import matplotlib.pyplot as plt
import os
from utils import mhd


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder:{} --- ", path)


# data_root = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/crop_2.0'
data_root = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/20_wo_muscles_plain/crop'
case_IDs = ['N0018', 'N0024', 'N0031', 'N0047', 'N0056', 'N0074', 'N0076',
            'N0091', 'N0094', 'N0107', 'N0108', 'N0116', 'N0132', 'N0133', 'N0140',
            'N0144', 'N0145', 'N0152', 'N0171', 'N0180', 'N0187']
# case_IDs = ['k1657', 'k1756', 'k1802', 'k1873', 'k1565', 'k8892', 'k1631', 'k1870', 'k1647', 'k1677']
vessel_colors = [[255, 0, 0], [0, 0, 255]]
GT_color = [[0, 0, 255]]
predict_color = [[0, 255, 0]]
thickness = 1
for case_ID in case_IDs:

    out_dir = os.path.join(data_root, 'visualization',  case_ID)
    mkdir(out_dir)
    # load data
    image_vol, hdr = mhd.read(os.path.join(data_root, case_ID+'_crop_image.mhd'))
    # vessels_vol, _ = mhd.read(os.path.join(data_root, case_ID, 'vessels_label.mhd'))

    GT_label_vol, _ = mhd.read(os.path.join(data_root, case_ID+'_artery_label.mhd'))
    predict_label_vol, _ = mhd.read(os.path.join(data_root, case_ID+'_crop_vessels_label.mhd'))
    GT_artery = np.zeros_like(GT_label_vol)
    Predict_artery = np.zeros_like(predict_label_vol)
    GT_artery[GT_label_vol == 2] = 1
    Predict_artery[predict_label_vol == 2] = 1


    # image_vol = np.array(image_vol, dtype=np.uint8)
    GT_artery = np.array(GT_artery, dtype=np.uint8)
    Predict_artery = np.array(Predict_artery, dtype=np.uint8)
    # vessels_vol = np.array(vessels_vol, dtype=np.uint8)

    image_vol = ImageHelper.normalize_hu(image_vol, dtype=np.uint8)
    labeled_vol = ImageHelper.label_images(images=image_vol,
                                           labels=GT_artery,
                                           colors=GT_color,
                                           thickness=thickness)
    labeled_vol = ImageHelper.label_images(images=labeled_vol,
                                           labels=Predict_artery,
                                           colors=predict_color,
                                           thickness=thickness)
    print(labeled_vol.shape)
    for i in range(labeled_vol.shape[1]):
        image = labeled_vol[:, i, :, :]
        ratio = hdr['ElementSpacing'][2] / hdr['ElementSpacing'][1]
        image = cv2.resize(image, (512, int(526 * ratio)))
        image = cv2.flip(image, 0)
        cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format('%03d' % i)), image)

