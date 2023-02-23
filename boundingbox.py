# -*- coding: utf-8 -*-
import numpy as np

def bbox(img):
    """Compute bounding box for given ndarray.

    Args:
        img (ndarray): Input image.
    Returns:
        (np.array, np.array): Bounding box of input image. (bbox_min, bbox_max)
    """
    if np.count_nonzero(img) == 0:
        raise ValueError('Input image is empty.')
    dim = img.ndim
    bb = np.array([np.where(np.any(img, axis=tuple([i for i in range(dim) if i != d])))[0][[0,-1]] for d in range(dim)])
    return bb[:,0],bb[:,1]

def crop(image, bbox, margin=0):
    """Crop image using given bounding box.

    Args:
        img (ndarray): Input image.
        bbox (np.array, np.array): Input bounding box.
        margin (int): The size of margin.
    Returns:
        np.ndarray: Cropped image.
    """
    bmin = bbox[0]
    bmax = bbox[1]
    bmin = np.maximum(0,bmin-margin)
    bmax = np.minimum(np.array(image.shape),bmax+(margin+1))
    return image[[slice(bmin[i],bmax[i]) for i in range(len(bmin))]]

def trim(image, margin=0):
    """Trim image.

    This function is equivalent to ``crop(image,bbox(image),margin)``

    Args:
        img (ndarray): Input image.
        margin (int): The size of margin.
    Returns:
        np.ndarray: Trimmed image.
    """
    bb = bbox(image)
    return crop(image, bb, margin)

def uncrop(image,original_shape,bbox,margin=0,constant_values=0):
    '''Revert cropping

    Args:
        image (ndarray): Input cropped image.
        original_shape (array__like): Original shape before cropping.
        bbox (np.array, np.array): Bounding box used for cropping.
        margin (int): Margin used for cropping.
        constant_values (int or array_like): Passed to np.pad
    Returns:
        np.ndarray: Uncropped image.
    '''
    before = np.maximum(bbox[0]-margin,0)
    after = np.maximum(np.array(original_shape)-bbox[1]-margin-1,0)
    pad_width = np.array((before,after)).T
    return np.pad(image,pad_width,'constant',constant_values=constant_values)


def bbox_mask_image(image, margin=0):
    ''' Create bounding box mask image

    '''
    bbimage = np.zeros(image.shape, dtype=np.uint8)
    bb = bbox(image)
    bbmin = np.clip(bb[0]-margin,0,None)
    bbmax = np.clip(bb[1]+1+margin,None,bbimage.shape)
    bbimage[bbmin[0]:bbmax[0],bbmin[1]:bbmax[1],bbmin[2]:bbmax[2]] = 1
    return bbimage


