import os
import numpy as np
import cv2
from utils import mhd
from medpy.metric.binary import dc
from skimage import measure
from scipy.ndimage import morphology
import sklearn
import sklearn.neighbors
from utils.similarity_metrics import assd
import tqdm
def load_image(filename):

    _, ext = os.path.splitext( os.path.basename(filename) )

    if ext in ('.mha', '.mhd'):
        [img, img_header] = mhd.read(filename)
        spacing = img_header['ElementSpacing']
        img.flags.writeable = True
        if img.ndim == 3:
            img = np.transpose(img, (1,2,0))

    elif ext in ('.png', '.jpg', '.bmp'):
        img = cv2.imread(filename)
        spacing = None

    else:
        raise NotImplementedError()

    return img, spacing

def save_image(filename, image, spacing=None):

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    _, ext = os.path.splitext( os.path.basename(filename) )

    if ext in ('.mha', '.mhd'):
        header = {}
        if spacing is not None:
            header['ElementSpacing'] = spacing
        if image.ndim == 2:
            header['TransformMatrix'] = '1 0 0 1'
            header['Offset'] = '0 0'
            header['CenterOfRotation'] = '0 0'
        elif image.ndim == 3:
            image = image.transpose((2,0,1))
            header['TransformMatrix'] = '1 0 0 0 1 0 0 0 1'
            header['Offset'] = '0 0 0'
            header['CenterOfRotation'] = '0 0 0'
        else:
            raise NotImplementedError()
        mhd.write(filename, image, header)

    elif ext in ('.png', '.jpg', '.bmp'):
        cv2.imwrite(filename, image)

    else:
        raise NotImplementedError()

def bbox(img):
    if np.count_nonzero(img) == 0:
        raise ValueError('Input image is empty.')
    dim = img.ndim
    bb = np.array([np.where(np.any(img, axis=tuple([i for i in range(dim) if i != d])))[0][[0,-1]] for d in range(dim)])
    return bb[:,0],bb[:,1]

def crop(image, bbox, margin=0, do_return_new_bb=False):
    bmin = bbox[0]
    bmax = bbox[1]
    bmin = np.maximum(0,bmin-margin)
    bmax = np.minimum(np.array(image.shape),bmax+(margin+1))
    dim = len(bmin)
    slice_str = ','.join(['bmin[{0}]:bmax[{0}]'.format(i) for i in range(dim)])
    cropped = eval('image[{0}].copy()'.format(slice_str))
    if do_return_new_bb:
        return cropped, (bmin,bmax)
    return cropped

def calculate_surface_distances(a, b, spacing=None, connectivity=None, keep_border=False):
    '''Extract border voxels of a and b, and return both a->b and b->a surface distances'''
    if spacing is None:
        spacing = np.ones(a.ndim)
    pts_a = np.column_stack(np.where(extract_contour(a,connectivity,keep_border))) * np.array(spacing)
    pts_b = np.column_stack(np.where(extract_contour(b,connectivity,keep_border))) * np.array(spacing)
    tree = sklearn.neighbors.KDTree(pts_b)
    dist_a2b,_ = tree.query(pts_a)
    tree = sklearn.neighbors.KDTree(pts_a)
    dist_b2a,_ = tree.query(pts_b)
    return dist_a2b, dist_b2a

def extract_contour(mask, connectivity=None, keep_border=False):
    if connectivity is None:
        connectivity = mask.ndim
    conn = morphology.generate_binary_structure(mask.ndim, connectivity)
    return np.bitwise_xor(mask, morphology.binary_erosion(mask, conn, border_value=0 if keep_border else 1))

def average_symmetric_surface_distance(dist_a, dist_b):
    if dist_a is None or dist_b is None:
        return np.nan
    return (np.sum(dist_a)+np.sum(dist_b))/(dist_a.size+dist_b.size)

def assd_custom(a,b, voxel_spacing=None, connectivity=1,keep_border=0):
    region_bb = a | b
    count_of_region_bb = np.count_nonzero(region_bb)
    if count_of_region_bb != 0:
        bb = bbox(a | b)
        a = crop(a,bb,1)
        b = crop(b,bb,1)
    dist_a, dist_b = calculate_surface_distances(a, b, voxel_spacing, connectivity, keep_border)
    return average_symmetric_surface_distance(dist_a, dist_b)

def evaluate(label1,label2, spacing, connectivity=1,lbl_range=[1,22],keep_border=0):
    '''
    label1: predict label
    label2: original label
    '''
    max_label1, max_label2 = np.max(label1), np.max(label2)
    max_label = int(max(max_label1, max_label2))
    min_label = 1
    # min_label, max_label = lbl_range[0], lbl_range[1]
    # _header = ['Case']
    ASDs = []
    DCs = []
    for i in range(min_label,max_label+1):
        a = np.asarray(label1 == i).astype(int)
        b = np.asarray(label2 == i).astype(int)
        # ASDs.append(assd(a,b, voxel_spacing=spacing, connectivity=connectivity))
        if (np.count_nonzero(a)!=0) and (np.count_nonzero(b)!=0):
            ASDs.append(assd_custom(a,b, voxel_spacing=spacing, connectivity=connectivity))
            DCs.append(dc(a,b))
        else: 
            ASDs.append(np.nan)
            DCs.append(0)
    return ASDs,DCs


def remove_noise_areas(arr, noise=0.05):
    max_area = np.max(arr[1:])
    idx_arr = [i+1 for i, x in enumerate (arr[1:]) if x/max_area > noise]
    return idx_arr


def largest_CC(image, n=1):
    labels = measure.label(image, connectivity=1, background=0)
    area = np.bincount(labels.flat)
    print(len(area))
    if ((len(area)>1) & (n==1)):
        # 1: means except the area of 0, to find the max area
        labels = np.in1d(labels, np.argmax(area[1:])+1).reshape(image.shape)
        return labels 
    elif ((len(area)>1) & (n>1)):
        labels = np.in1d(labels, area.argsort()[-n-1:-1][::-1]).reshape(image.shape)
        return labels
    elif ((len(area)>1) & (n<1)):
        indices = remove_noise_areas(area, noise=n)
        labels = np.in1d(labels, indices).reshape(image.shape)
        return labels
    else:
        return np.zeros(labels.shape,np.bool)

def refine_labels(labels, n=1):
    refined = np.zeros_like(labels)
    for i in range(1,np.max(labels)+1):
        cc = largest_CC(labels==i, n=n)
        refined[cc] = i
    return refined

def refine_labels_var(labels, ns):
    refined = np.zeros_like(labels)
    for i, n in zip(range(1,np.max(labels)+1), ns):
        cc = largest_CC(labels==i, n=n)
        refined[cc] = i
    return refined