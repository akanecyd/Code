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


def resize_flip_img(img_vol, image, image_ratio):
    img = cv2.resize(image, (512, int(img_vol.shape[0] * image_ratio)))
    img = cv2.flip(img, 0)
    return img


# data_root = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/crop_2.0'
data_root = '//Salmon/User/mazen/Segmentation/Data/HipMusclesDataset/2.0.3'
distance_root = '//Salmon/User/mazen/Segmentation/Data/HipMusclesDataset/2.0.3'
# case_IDs = ['N0018', 'N0024', 'N0031', 'N0047', 'N0056', 'N0074', 'N0076',
#             'N0091', 'N0094', 'N0107', 'N0108', 'N0116', 'N0132', 'N0133', 'N0140',
#             'N0144', 'N0145', 'N0152', 'N0171', 'N0180', 'N0187']
# case_IDs = ['k1657', 'k1756', 'k1802', 'k1873', 'k1565', 'k8892', 'k1631', 'k1870', 'k1647', 'k1677']
with open(os.path.join(data_root, 'caseid_list_50.txt')) as f:
    case_IDs = f.read().splitlines()


for case_ID in case_IDs:

    out_dir = os.path.join(data_root, 'Visualizations', 'Muscles_distance', case_ID)
    mkdir(out_dir)
    # load data
    image_vol, hdr = mhd.read(os.path.join(data_root, case_ID, 'image.mhd'))
    distance_map_vol, _ = mhd.read(os.path.join(distance_root, case_ID, 'muscle_bone_distance_map.mhd'))


    # GT_label_vol, _ = mhd.read(os.path.join(data_root, case_ID+'_artery_label.mhd'))
    # predict_label_vol, _ = mhd.read(os.path.join(data_root, case_ID+'_crop_vessels_label.mhd'))
    # GT_artery = np.zeros_like(GT_label_vol)
    # Predict_artery = np.zeros_like(predict_label_vol)
    # GT_artery[GT_label_vol == 2] = 1
    # Predict_artery[predict_label_vol == 2] = 1


    # image_vol = np.array(image_vol, dtype=np.uint8)
    # GT_artery = np.array(GT_artery, dtype=np.uint8)
    # Predict_artery = np.array(Predict_artery, dtype=np.uint8)
    image_vol = ImageHelper.normalize_hu(image_vol, dtype=np.uint8)
    fig = plt.figure()




    for i in range(200, 300, 10):
        distance_map_vol = np.array(distance_map_vol, dtype=np.uint8)
        # distance_map_vol = cv2.IMREAD_GRAYSCALE (distance_map_vol)
        # im_color = cv2.applyColorMap(distance_map_vol, cv2.COLORMAP_JET)
        ct_image = image_vol[:, i, :]
        ratio = hdr['ElementSpacing'][2] / hdr['ElementSpacing'][1]
        ct_image = resize_flip_img(img_vol=image_vol, image=ct_image, image_ratio=ratio)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        plt.imshow(ct_image, cmap='gray')
        plt.axis('off')
        plt.show()

        ax2 = fig.add_subplot(1, 2, 2)

        # show plots

        distance_image = distance_map_vol[:, i, :]
        distance_image = resize_flip_img(img_vol=image_vol, image=distance_image, image_ratio=ratio)

        im_ratio = distance_image.shape[0] / distance_image.shape[1]
        img = ax2.imshow(distance_image, cmap='jet')
        ax2.axis('off')
        fig.subplots_adjust(right=0.9)
        position = fig.add_axes([0.95, 0.22, 0.015, .58])  # 位置[左,下,右,上]
        cb = plt.colorbar(img, cax=position, fraction=0.040 * im_ratio)
        colorbarfontdict = {"size": 10, "color": "k", 'family': 'Times New Roman'}
        cb.ax.set_title('Distance\n(Pixels)', fontdict=colorbarfontdict, pad=8)
        cb.ax.tick_params(labelsize=11, direction='in')
        cb.ax.set_yticklabels(['0', '5', '10', '15', '20', '25', '30', '40'], family='Times New Roman')
        # cb.set_label('Distance', fontdict=font)  # 设置colorbar的标签字体及其大小
        # cb = fig.colorbar(aximg, ax=ax2, label="Distance", orientation="vertical")
        # ax2.clim(0, np.max(distance_map_vol))

        fig.show()
        fig.savefig(os.path.join(out_dir, '{}_distance_map.png'.format('%03d' % i)), dpi=100)
        # overlay_image
        im_color = distance_image/np.max(distance_image)*255
        im_color = np.array(im_color, dtype=np.uint8)
        im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_JET)
        ct_image = cv2.cvtColor(ct_image, cv2.COLOR_GRAY2RGB)
        # cv2.imshow('image', ct_image)
        added_image = cv2.addWeighted(ct_image, 0.4, im_color, 0.3, 0)
        cv2.imwrite(os.path.join(out_dir, '{}_overlay_distance.png'.format('%03d' % i)), added_image)



