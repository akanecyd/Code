from utils.ImageHelper import ImageHelper
from utils.VideoHelper import VideoHelper
from utils import mhd
import numpy as np
import cv2
import os
import glob2

data_root = 'D:/temp/patch_test'
case_ID = 'k1647'
vessel_colors = [[255, 0, 0], [0, 0, 255]]
thickness = 1

image_dir = os.path.join(data_root, case_ID, 'patch_images')
label_dir = os.path.join(data_root, case_ID, 'patch_labels')
out_dir = os.path.join(data_root, case_ID, 'visul_patches')

image_names = glob2.glob(os.path.join(image_dir, '*.mhd'))
label_names = glob2.glob(os.path.join(label_dir, '*.mhd'))

for i in range(len(image_names)):
    image_vol, hdr = mhd.read(image_names[i])
    print(image_names[i])
    label_vol, _ = mhd.read(label_names[i])
    print(label_names[i])
    label_vol = np.array(label_vol, dtype=np.uint8)
    image_vol = ImageHelper.normalize_hu(image_vol, dtype=np.uint8)
    outdir = out_dir
    labeled_patch_vol = ImageHelper.label_images(images=image_vol,
                                                 labels=label_vol,
                                                 colors=vessel_colors,
                                                 thickness=thickness)
    # cv2.imwrite(os.path.join(outdir, "{}_labeled_vol.png".format('%03d' % i)), labeled_patch_vol[:,:,64,:])
    VideoHelper.write_vol_to_video(vol=labeled_patch_vol,
                                   case_name='k1647',
                                   output_path=os.path.join(outdir, "{}.mp4".format(image_names[i][-33:-4])),
                                   if_reverse=True)
