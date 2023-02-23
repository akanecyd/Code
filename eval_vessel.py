import scipy.ndimage as ndimage
import numpy as np


def dilate(image, n):
    st = ndimage.morphology.generate_binary_structure(3, 2)
    return ndimage.binary_dilation(image, structure=st, iterations=n)


def evaluate(truth, result, margin=2, d_truth=None):
    '''
    TP=1,FP=2,FN=3
    '''
    if d_truth is None:
        d_truth = dilate(truth, margin)
    d_result = dilate(result, margin)
    truth_tp = truth & d_result
    r_truth = 3 * (truth > 0)  # initialize with FN
    r_truth[truth_tp > 0] = 1  # TP

    r_result = 2 * (result > 0)  # initialize with FP
    result_tp = d_truth & result
    r_result[result_tp > 0] = 1  # TP

    return r_truth.astype(np.uint8), r_result.astype(np.uint8)
