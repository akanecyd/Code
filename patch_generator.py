import numpy as np
from utils.ImageHelper import ImageHelper
from utils.VideoHelper import VideoHelper
import matplotlib.pyplot as plt
import os
from utils import mhd


def volume_patch_gen(images,
                     labels,
                     vessels_skeleton,
                     patch_size,
                     random_size,
                     out_dir,
                     colorlist,
                     head_in):

    PS = (patch_size - 1) // 2

    npad = ((PS, PS), (PS, PS), (PS, PS))

    # Pad whole image with zeros to allow for patches at edges

    image_pad = np.pad(images, pad_width=npad, mode='reflect')
    label_pad = np.pad(labels, pad_width=npad, mode='reflect')
    recover_pad_image_vol = np.zeros_like(image_pad)
    recover_pad_label_vol = np.zeros_like(label_pad)

    pad_label_vol = np.array(label_pad, dtype=np.uint8)
    pad_image_vol = ImageHelper.normalize_hu(image_pad, dtype=np.uint8)
    labeled_pad_vol = ImageHelper.label_images(images=pad_image_vol,
                                               labels=pad_label_vol,
                                               colors=colorlist,
                                               thickness=1)

    VideoHelper.write_vol_to_video(vol=labeled_pad_vol,
                                   case_name='k1802',
                                   output_path=os.path.join(out_dir, "pad_labeled_images.mp4"),
                                   if_reverse=True)

    # extract vessel skeleton coordinate
    vessels_center_location = np.argwhere(vessels_skeleton > 0)
    center_coordinate = deleteDuplicatedElementFromList(vessels_center_location.tolist())
    # sample random numbers
    random_coordinate = np.random.choice(len(center_coordinate), size=random_size, replace=False, p=None)

    # Cycle over all voxels
    crop_list = []
    for i in range(0, len(random_coordinate)):
        random_number = random_coordinate[i]
        x, y, z = center_coordinate[random_number]
        x_pad = x + PS
        y_pad = y + PS
        z_pad = z + PS

        tmp_patch_image = image_pad[x_pad - PS:x_pad + PS + 2,
                                    y_pad - PS:y_pad + PS + 2,
                                    z_pad - PS:z_pad + PS + 2]
        tmp_patch_label = label_pad[x_pad - PS:x_pad + PS + 2,
                                    y_pad - PS:y_pad + PS + 2,
                                    z_pad - PS:z_pad + PS + 2]

        # save random patches
        mhd_path = os.path.join(out_dir, 'patches_mhd')
        mkdir(mhd_path)
        mhd.write(mhd_path + '/' +
                  'random_patch_' + '%03d' % random_number + '_image.mhd',
                  tmp_patch_image, head_in)
        mhd.write(mhd_path + '/' +
                  'random_patch_' + '%03d' % random_number + '_label.mhd',
                  tmp_patch_label, head_in)
        # visual each labeled patch
        vis_path = os.path.join(out_dir, 'patch_videos')
        mkdir(vis_path)
        patch_label_vol = np.array(tmp_patch_label, dtype=np.uint8)
        patch_image_vol = ImageHelper.normalize_hu(tmp_patch_image, dtype=np.uint8)
        labeled_pad_vol = ImageHelper.label_images(images=patch_image_vol,
                                                   labels=patch_label_vol,
                                                   colors=colorlist,
                                                   thickness=1)
        # VideoHelper.write_vol_to_video(vol=labeled_pad_vol,
        #                                case_name='k1802',
        #                                output_path=os.path.join(vis_path,
        #                                                         'random_' + '%03d' % random_number + '_patch.mp4'),
        #                                if_reverse=True)
        # quick check of each patch
        rows = 8
        cols = 8
        axes = []
        fig = plt.figure()

        for j in range(rows * cols):
            sub_image = labeled_pad_vol[j, :, :, :]
            axes.append(fig.add_subplot(rows, cols, j+1))
            plt.axis('off')
            plt.imshow(sub_image)
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join(vis_path, 'random_' + '%03d' % random_number + '_patch.png'))

        # unpatch recover shape
        recover_pad_image_vol[x_pad - PS:x_pad + PS + 2, y_pad - PS:y_pad + PS + 2,
        z_pad - PS:z_pad + PS + 2] = tmp_patch_image
        recover_pad_label_vol[x_pad - PS:x_pad + PS + 2, y_pad - PS:y_pad + PS + 2,
        z_pad - PS:z_pad + PS + 2] = tmp_patch_label

    mhd.write(mhd_path + '/' + 'recover_pad_image.mhd', recover_pad_image_vol, head_in)
    mhd.write(mhd_path + '/' + 'recover_pad_label.mhd', recover_pad_label_vol, head_in)
    # crop the padded volume
    sample_image_vol = recover_pad_image_vol[PS:PS + images.shape[0],
                                             PS:PS + images.shape[1],
                                             PS:PS + images.shape[2]]
    sample_label_vol = recover_pad_label_vol[PS:PS + images.shape[0],
                                             PS:PS + images.shape[1],
                                             PS:PS + images.shape[2]]
    mhd.write(mhd_path + '/' + 'recover_image.mhd', sample_image_vol, head_in)
    mhd.write(mhd_path + '/' + 'recover_label.mhd', sample_label_vol, head_in)









def deleteDuplicatedElementFromList(list):
    resultList = []
    for item in list:
        if not item in resultList:
            resultList.append(item)
    return resultList


def vessels_center_coordinate(vein_center, artery_center):
    vein_center_location = np.argwhere(vein_center > 0)
    artery_center_location = np.argwhere(artery_center > 0)
    vessels_center_location = np.append(vein_center_location, artery_center_location, axis=0).tolist()
    return vessels_center_location


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder:{} --- ", path)




#
#
if __name__ == '__main__':
    data_root = '//Salmon/User/Chen/Vessel_data/vessels_with_muscles'
    case_ID = 'k1802'
    vessel_colors = [[255, 0, 0], [0, 0, 255]]
    thickness = 1
    out_dir = os.path.join('//Salmon/User/Chen/Result/patch_test', case_ID+'_random_100')
    mkdir(out_dir)
    # load data
    image_vol, hdr = mhd.read(os.path.join(data_root, case_ID, 'image.mhd'))
    label_vol, _ = mhd.read(os.path.join(data_root, case_ID, 'vessels_label.mhd'))
    vein_skel_vol, _ = mhd.read(os.path.join(data_root, case_ID, 'vein_skel.mhd'))
    artery_skel_vol, _ = mhd.read(os.path.join(data_root, case_ID, 'artery_skel.mhd'))
    vessels_skel_vol, _ = mhd.read(os.path.join(data_root, case_ID, 'vessels_skel.mhd'))

    # patch generate
    head_in = {'CompressedData': True, 'ElementSpacing': hdr['ElementSpacing']}
    volume_patch_gen(images=image_vol, labels=label_vol, vessels_skeleton=vessels_skel_vol,
                     out_dir=out_dir, colorlist=vessel_colors,
                     patch_size=64, random_size=100, head_in=head_in)

    # write original video
    label_vol = np.array(label_vol, dtype=np.uint8)
    image_vol = ImageHelper.normalize_hu(image_vol, dtype=np.uint8)
    labeled_patch_vol = ImageHelper.label_images(images=image_vol,
                                                 labels=label_vol,
                                                 colors=vessel_colors,
                                                 thickness=thickness)

    VideoHelper.write_vol_to_video(vol=labeled_patch_vol,
                                   case_name=case_ID,
                                   output_path=os.path.join(out_dir, "labeled_original_images.mp4"),
                                   if_reverse=True)




    # print(image_patch_vol.shape)
    # print(label_patch_vol.shape)
    #
    # print(np.min(image_patch_vol), np.max(image_patch_vol))
    # print(np.min(label_patch_vol), np.max(label_patch_vol))
