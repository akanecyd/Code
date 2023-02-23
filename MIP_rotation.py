import os.path

import SimpleITK as sitk
import numpy as np
from utils.ImageHelper import ImageHelper
from utils import mhd

case_IDs = ['N0133']

ct_root = 'D:/NARA_ENHANCE/MIP_bones_removal'
for case_ID in case_IDs:
    print('____processing:{}____'.format(case_ID))
    # [image, hdr] = mhd.read(os.path.join(ct_root, '{}_MIP_enhanced_image.mhd'.format(case_ID)))
    # head_in = {'CompressedData': True, 'ElementSpacing': hdr['ElementSpacing']}
    # image = ImageHelper.normalize_hu(image)
    # mhd.write(os.path.join(ct_root, '{}_normal_enhanced_ct.mhd'.format(case_ID)), image, head_in)
    # image = sitk.ReadImage(os.path.join(ct_root, '{}_normal_enhanced_ct.mhd'.format(case_ID)))
    image = sitk.ReadImage(os.path.join(ct_root, '{}_MIP_enhanced_image.mhd'.format(case_ID)))
    projection = {'sum': sitk.SumProjection,
                  'mean': sitk.MeanProjection,
                  'std': sitk.StandardDeviationProjection,
                  'min': sitk.MinimumProjection,
                  'max': sitk.MaximumProjection}
    ptype = 'max'
    paxis = 0

    rotation_axis = [0, 0, 1]
    rotation_angles = np.linspace(0.0, 2 * np.pi, int(360.0 / 20.0))
    rotation_center = image.TransformContinuousIndexToPhysicalPoint([(index - 1) / 2.0 for index in image.GetSize()])
    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_center)

    # Compute bounding box of rotating volume and the resampling grid structure

    image_indexes = list(zip([0, 0, 0], [sz - 1 for sz in image.GetSize()]))
    image_bounds = []
    for i in image_indexes[0]:
        for j in image_indexes[1]:
            for k in image_indexes[2]:
                image_bounds.append(image.TransformIndexToPhysicalPoint([i, j, k]))

    all_points = []
    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle)
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)
    # resampling grid will be isotropic so no matter which direction we project to
    # the images we save will always be isotropic (required for image formats that
    # assume isotropy - jpg,png,tiff...)
    new_spc = [np.min(image.GetSpacing())] * 3
    new_sz = [int(sz / spc + 0.5) for spc, sz in zip(new_spc, max_bounds - min_bounds)]

    proj_images = []
    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle)
        resampled_image = sitk.Resample(image1=image,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkLinear,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                                        defaultPixelValue=-150,
                                        # HU unit for air in CT, possibly set to 0 in other cases
                                        outputPixelType=image.GetPixelID())
        proj_image = projection[ptype](resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis] = 0
        proj_images.append(sitk.Extract(proj_image, extract_size))

    # Stack all images into fuax-volume for display
    sitk.Show(sitk.JoinSeries(proj_images))
