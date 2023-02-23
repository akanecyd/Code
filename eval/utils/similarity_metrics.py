# -*- coding: utf-8 -*-
'''Implementations of similarity measurements for evaluation of semantic image segmentation.

Example:
    similarity.evaluate_labels(['JI','Dice','ASD'], predicted_labels, true_labels)

References:
    T. Heimann et al., "Comparison and Evaluation of Methods for Liver Segmentation From CT Datasets," in IEEE Transactions on Medical Imaging, vol. 28, no. 8, pp. 1251-1265, Aug. 2009.
    M. S. Fasihi and W. B. Mikhael, "Overview of Current Biomedical Image Segmentation Methods," 2016 International Conference on Computational Science and Computational Intelligence (CSCI), Las Vegas, NV, 2016, pp. 803-808.
'''
import sys
import os
import numpy as np
from scipy.ndimage import morphology
import sklearn.neighbors
import tqdm
from inspect import signature

from medpy.metric import binary

_thismodule = sys.modules[__name__]

# IO functions
def load_volume(filename, axis_order='xyz'):
    import SimpleITK as sitk
    itkimage = sitk.ReadImage(filename)
    volume = sitk.GetArrayFromImage(itkimage)
    spacing = np.array(list(itkimage.GetSpacing()))
    origin = np.array(list(itkimage.GetOrigin()))

    if axis_order == 'xyz':
        volume = np.transpose(volume, (2, 1, 0))
    elif axis_order == 'zyx':
        spacing = spacing[::-1]
        origin = origin[::-1]
    else:
        raise ValueError('unexpected axis order')

    return volume, spacing, origin

#Overlap
def calculate_confusion_matrix(prediction, truth):
    if prediction is None or truth is None:
        return None, None, None, None
    p = prediction
    inv_p = ~p
    t = truth
    inv_t = ~t
    tp = np.count_nonzero(p & t)
    fp = np.count_nonzero(p & inv_t)
    fn = np.count_nonzero(inv_p & t)
    tn = np.count_nonzero(inv_p & inv_t)
    return tp, fp, fn, tn

def jaccard_index(tp, fp, fn ,tn):
    if tp is None or fp is None or fn is None:
        return np.nan
    '''Jaccard Index'''
    tp_fp_fn = tp+fp+fn
    inv_tp_fp_fn=0.0
    if tp_fp_fn > 0:
        inv_tp_fp_fn = 1.0 / tp_fp_fn
    return tp * inv_tp_fp_fn
    return tp / (tp+fp+fn)
def JI(tp, fp, fn ,tn):
    '''Alias of :func:`jaccard_index`'''
    return jaccard_index(tp, fp, fn ,tn)
def VO(tp, fp, fn ,tn):
    '''Volume Overlap : alias of :func:`jaccard_index`'''
    return jaccard_index(tp, fp, fn ,tn)
def jc(result,reference):
    '''Jaccard coefficient(/index)'''
    return binary.jc(result,reference)
def volumetric_overlap_error(tp, fp, fn, tn):
    if tp is None or fp is None or fn is None:
        return np.nan
    return 1 - jaccard_index(tp, fp, fn ,tn)
def VOE(tp, fp, fn ,tn):
    '''Alias of :func:`volumetric_overlap_error`'''
    return volumetric_overlap_error(tp, fp, fn ,tn)
def jce(result,reference):
    '''Jaccard coefficient(/index) error'''
    return 1 - jce(result,reference)
def relative_volume_difference(tp, fp, fn ,tn):
    if tp is None or fp is None or fn is None:
        return np.nan
    vol_truth = tp+fn
    vol_pred = tp+fp
    return (vol_pred - vol_truth) / vol_truth
def VolDiff(tp, fp, fn ,tn):
    '''Alias of :func:`relative_volume_difference`'''
    return relative_volume_difference(tp, fp, fn ,tn)
def dice_coeff(tp, fp, fn ,tn):
    #if tp is None or fp is None or fn is None:
    #    return np.nan
    '''Dice coefficient'''
    #return 2*tp / (2*tp+fp+fn)
    return f_measure(tp, fp, fn ,tn)
def Dice(tp, fp, fn ,tn):
    '''Alias of :func:`dice_coeff`'''
    return dice_coeff(tp, fp, fn ,tn)
def dc(result,reference):
    '''Dice coefficient'''
    return binary.dc(result,reference)
def dce(result,reference):
    '''Dice coefficient error'''
    return 1 - dc(result,reference)
def prec(result,reference):
    '''Precison'''
    return binary.precision(result,reference)
def recll(result,reference):
    '''Recall'''
    return binary.recall(result,reference)
def snstvty(result,reference):
    '''Sensitivity'''
    return binary.sensitivity(result,reference)
def spcfcty(result,reference):
    '''Specificity'''
    return binary.specificity(result,reference)
def truepositiverate(result,reference):
    '''True positive rate'''
    return binary.true_positive_rate(result,reference)
def truenegativerate(result,reference):
    '''True negative rate'''
    return binary.true_negative_rate(result,reference)
def ravd(result,reference):
    '''Relative absolute volume difference'''
    return binary.ravd(result,reference)
def recall(tp, fp, fn ,tn):
    if tp is None or fn is None:
        return np.nan
    tp_fn = tp+fn
    inv_tp_fn=0.0
    if tp_fn > 0:
        inv_tp_fn = 1.0 / tp_fn
    return tp * inv_tp_fn
    return tp / (tp+fn)
def sensitivity(tp, fp, fn ,tn):
    '''Alias of :func:`recall`'''
    return recall(tp, fp, fn ,tn)
def TPR(tp, fp, fn ,tn):
    '''True Positive Rate : alias of :func:`recall`'''
    return recall(tp, fp, fn ,tn)
def TPF(tp, fp, fn ,tn):
    '''True Positive Fraction : alias of :func:`recall`'''
    return recall(tp, fp, fn ,tn)
def FPR(tp, fp, fn ,tn):
    if tp is None or fp is None or fn is None:
        return np.nan
    tp_fn = tp+fn
    inv_tp_fn=0.0
    if tp_fn > 0:
        inv_tp_fn = 1.0 / tp_fn
    '''False Positive Rate'''
    return fp * inv_tp_fn
    return fp / (tp+fn)
def precision(tp, fp, fn ,tn):
    if tp is None or fp is None:
        return np.nan
    tp_fp = tp+fp
    inv_tp_fp=0.0
    if tp_fp > 0:
        inv_tp_fp = 1.0 / tp_fp
    return tp * inv_tp_fp
    return tp / (tp+fp)
def specificity(tp, fp, fn ,tn):
    if fp is None or tn is None:
        return np.nan
    tn_fp = tn+fp
    inv_tn_fp=0.0
    if tn_fp > 0:
        inv_tn_fp = 1.0 / tn_fp
    return tn * inv_tn_fp
    return tn / (tn+fp)
def f_measure(tp, fp, fn ,tn):
    if tp is None or fp is None or fn is None:
        return np.nan
    tp_fp_fn = 2.0*tp+fp+fn
    inv_tp_fp_fn=0.0
    if tp_fp_fn > 0:
        inv_tp_fp_fn = 1.0 / tp_fp_fn
    return 2*tp * (1.0 * inv_tp_fp_fn)
    return 2*tp / (2*tp+fp+fn)
def false_positive_fraction(tp, fp, fn ,tn):
    if tp is None or fp is None or fn is None:
        return np.nan
    '''aka False Discovery Rate'''
    tp_fp = tp+fp
    inv_tp_fp=0.0
    if tp_fp > 0:
        inv_tp_fp = 1.0 / tp_fp
    return fp * inv_tp_fp
    return fp / (tp+fp)
def FPF(tp, fp, fn ,tn):
    '''Alias of :func:`false_positive_fraction`'''
    return false_positive_fraction(tp, fp, fn ,tn)
def accuracy(tp, fp, fn ,tn):
    if tp is None or fp is None or fn is None or tn is None:
        return np.nan
    tp_fp_fn_tn = tp+fp+fn+tn
    inv_tp_fp_fn_tn=0.0
    if tp_fp_fn_tn > 0:
        inv_tp_fp_fn_tn = 1.0 / tp_fp_fn_tn
    return (tp+tn) * inv_tp_fp_fn_tn
    return (tp+tn) / (tp+fp+fn+tn)
def TP(tp, fp, fn ,tn):
    if tp is None:
        return np.nan
    return tp
def FP(tp, fp, fn ,tn):
    if fp is None:
        return np.nan
    return fp
def FN(tp, fp, fn ,tn):
    if fn is None:
        return np.nan
    return fn
def TN(tp, fp, fn ,tn):
    if tn is None:
        return np.nan
    return tn

#Surface Distance
def average_symmetric_surface_distance(dist_a, dist_b):
    if dist_a is None or dist_b is None:
        return np.nan
    return (np.sum(dist_a)+np.sum(dist_b))/(dist_a.size+dist_b.size)
def ASD(dist_a, dist_b):
    ''' Alias of :func:`average_symmetric_surface_distance`'''
    return average_symmetric_surface_distance(dist_a, dist_b)
def average_prediction_to_truth_surface_distance(dist_pred, dist_truth):
    if dist_pred is None:
        return np.nan
    return np.mean(dist_pred)
def AS2RSD(dist_pred , dist_truth):
    '''Alias of :func:`average_prediction_to_truth_surface_distance`'''
    return average_prediction_to_truth_surface_distance(dist_pred, dist_truth)
def average_truth_to_prediction_surface_distance(dist_pred, dist_truth):
    return np.mean(dist_truth)
def AR2SSD(dist_pred , dist_truth):
    '''Alias of :func:`average_truth_to_prediction_surface_distance`'''
    return average_truth_to_prediction_surface_distance(dist_pred, dist_truth)

def root_mean_square_symmetric_surface_distance(dist_a, dist_b):
    if dist_a is None or dist_b is None:
        return np.nan
    return np.sqrt((np.sum(dist_a**2)+np.sum(dist_b**2))/(dist_a.size+dist_b.size))
def RMSD(dist_a, dist_b):
    ''' Alias of :func:`root_mean_square_symmetric_surface_distance`'''
    return root_mean_square_symmetric_surface_distance(dist_a, dist_b)

def maximum_symmetric_surface_distance(dist_a, dist_b):
    if dist_a is None or dist_b is None:
        return np.nan
    return max(np.max(dist_a),np.max(dist_b))
def MSD(dist_a, dist_b):
    ''' Alias of :func:`maximum_symmetric_surface_distance`'''
    return maximum_symmetric_surface_distance(dist_a, dist_b)
def maximum_prediction_to_truth_surface_distance(dist_pred, dist_truth):
    if dist_pred is None:
        return np.nan
    return np.max(dist_pred)
def MS2RSD(dist_pred , dist_truth):
    '''Alias of :func:`maximum_prediction_to_truth_surface_distance`'''
    return maximum_prediction_to_truth_surface_distance(dist_pred, dist_truth)
def maximum_truth_to_prediction_surface_distance(dist_pred, dist_truth):
    if dist_truth is None:
        return np.nan
    return np.max(dist_truth)
def MR2SSD(dist_pred , dist_truth):
    '''Alias of :func:`maximum_truth_to_prediction_surface_distance`'''
    return maximum_truth_to_prediction_surface_distance(dist_pred, dist_truth)

def hd(result,reference,voxel_spacing=None,conectivity=1):
    '''Hausdorff Distance'''
    return binary.hd(result,reference,voxel_spacing,conectivity)
def asd(result,reference,voxel_spacing=None,conectivity=1):
    '''Average surface distance'''
    return binary.asd(result,reference,voxel_spacing,conectivity)
def assd(result,reference,voxel_spacing=None,connectivity=1):
    '''Average symmetric surface distance.'''
    return binary.assd(result,reference,voxel_spacing,connectivity)
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
    
def extract_contour(mask, connectivity=None, keep_border=False):
    if connectivity is None:
        connectivity = mask.ndim
    conn = morphology.generate_binary_structure(mask.ndim, connectivity)
    return np.bitwise_xor(mask, morphology.binary_erosion(mask, conn, border_value=0 if keep_border else 1))

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

def evaluate_labels_medpy(metrics, prediction_labels, true_labels, spacing=None, n_classes=None, connectivity=1, keep_border=None, background_classes=[0], do_remove_background_classess_already=False, verbose=False):
    '''Evaluate segmentation accuracy using given metrics for each class in the label image'''
    do_calc_n_classes_here=False
    if n_classes is None:
        n_classes = max(np.max(prediction_labels),np.max(true_labels)) + 1
        do_calc_n_classes_here = True
    #
    if not do_calc_n_classes_here:
        if do_remove_background_classess_already:
            background_classes=[]
    #
    class_labels = [i for i in range(n_classes) if i not in background_classes]
    if not do_calc_n_classes_here:
        if do_remove_background_classess_already:
            class_labels_temp = class_labels
            class_labels = [i+1 for i in range(0,len(class_labels_temp))]
    functions = [getattr(_thismodule,metric) for metric in metrics]
    n_func_args = [len(signature(func).parameters) for func in functions]
    overlap_metric_exists = sum([n==4 for n in n_func_args]) > 0
    dist_metric_exists = sum([n==2 for n in n_func_args]) > 0
    result = []
    for i in (tqdm.tqdm if verbose else iter)(class_labels):
        pred_bin = prediction_labels==i
        true_bin = true_labels==i
        #
        count_of_pred_bin = np.count_nonzero(pred_bin)
        count_of_true_bin = np.count_nonzero(true_bin)
        #
        args = [[pred_bin,true_bin,spacing,connectivity] if n == 4 else [pred_bin, true_bin] for n in n_func_args]
        values = [f(*a) for f,a in zip(functions,args)]
        result.append(values)
        #
    #
    return result, n_classes


def evaluate_labels(metrics, prediction_labels, true_labels, spacing=None, n_classes=None, connectivity=None, keep_border=None, background_classes=[0], do_remove_background_classess_already=False, verbose=False):
    '''Evaluate segmentation accuracy using given metrics for each class in the label image'''
    do_calc_n_classes_here=False
    if n_classes is None:
        n_classes = max(np.max(prediction_labels),np.max(true_labels)) + 1
        do_calc_n_classes_here = True
    #
    if not do_calc_n_classes_here:
        if do_remove_background_classess_already:
            background_classes=[]
    #
    class_labels = [i for i in range(n_classes) if i not in background_classes]
    if not do_calc_n_classes_here:
        if do_remove_background_classess_already:
            class_labels_temp = class_labels
            class_labels = [i+1 for i in range(0,len(class_labels_temp))]
    functions = [getattr(_thismodule,metric) for metric in metrics]
    print(functions)
    n_func_args = [len(signature(func).parameters) for func in functions]
    overlap_metric_exists = sum([n==4 for n in n_func_args]) > 0
    dist_metric_exists = sum([n==2 for n in n_func_args]) > 0
    print(dist_metric_exists)
    result = []
    for i in (tqdm.tqdm if verbose else iter)(class_labels):
        pred_bin = prediction_labels==i
        print(type(pred_bin))
        true_bin = true_labels==i
        print(type(true_bin))
        #
        region_bb = pred_bin | true_bin
        count_of_region_bb = np.count_nonzero(region_bb)
        
        if False:#True:
            count_of_pred_bin = np.count_nonzero(pred_bin)
            count_of_true_bin = np.count_nonzero(true_bin)
            if i == 0:
                print()
            print(("count_of_region_bb : "), count_of_region_bb)
            print(("count_of_pred_bin : "), count_of_pred_bin)
            print(("count_of_true_bin : "), count_of_true_bin)
            print()
        if count_of_region_bb != 0:
            bb = bbox(pred_bin | true_bin)
            pred_bin = crop(pred_bin,bb,1)
            true_bin = crop(true_bin,bb,1)
        #
        count_of_pred_bin = np.count_nonzero(pred_bin)
        count_of_true_bin = np.count_nonzero(true_bin)
        #
        if overlap_metric_exists:
            tp, fp, fn, tn = calculate_confusion_matrix(pred_bin, true_bin)
        if dist_metric_exists:
            dist_pred = None
            dist_true = None
            if count_of_pred_bin !=0 and count_of_true_bin != 0:#or
                dist_pred, dist_true = calculate_surface_distances(pred_bin, true_bin, spacing, connectivity, keep_border)
        args = [[tp,fp,fn,tn] if n == 4 else [dist_pred, dist_true] for n in n_func_args]
        print(args)
        print(functions)
        values = [f(*a) for f,a in zip(functions,args)]
        result.append(values)
    return result, n_classes

#
def _get_filelist_contents_list(FilelistFile):
    fin = open(str(FilelistFile),'r')
    dst = []
    for line in fin:
        contents = line[:-1]#.split('\t')
        dst.append(contents)
    fin.close()
    return dst

def _add_slash_check_empty(input_path,split_str=("/")):
    if input_path == str(""):
        return input_path

    if input_path[len(input_path)-1:] == str(split_str):
        return input_path

    return input_path + split_str

def _replace_list_str(str_list, target_str=("\\"), to_str=("/")):
    return [ s.replace(target_str,to_str) for s in str_list ]

def _composite_path(filename):
    dirname, basefile = os.path.split(filename)
    basename, ext = os.path.splitext(basefile)
    return dirname, basename, ext

def _composite_path_ver2(filename):
    dirname, basename, ext = _composite_path(filename)
    return _add_slash_check_empty(dirname), basename, ext

def _make_dir_and_path_by_rootpath_and_filename(output_file):
    out_path,out_filename,out_ext = _composite_path_ver2(output_file)
    #
    out_filename_path = out_path + out_filename
    os.makedirs(out_filename_path,exist_ok=True)
    out_filename_path = _add_slash_check_empty(out_filename_path)
    return out_filename_path, out_path, out_filename, out_ext

class _FileWriter():
    def __init__(self, path):
        self.file = open(path, 'w')

    #def finalize(self):
    def __del__(self):
        self.file.close()

    def writes(self, list_object):
        for i in range (0, len(list_object)):
            self.file.write(("%.6e" % list_object[i]))
            if i == (len(list_object) - 1):
                self.file.write(("\n"))
            else:
                self.file.write(("\t"))
    
    def writes10(self, list_object):
        for i in range (0, len(list_object)):
            self.file.write(("%.10e" % list_object[i]))
            if i == (len(list_object) - 1):
                self.file.write(("\n"))
            else:
                self.file.write(("\t"))
    
    def writesStr(self, list_object):
        for i in range (0, len(list_object)):
            self.file.write(("s" % list_object[i]))
            if i == (len(list_object) - 1):
                self.file.write(("\n"))
            else:
                self.file.write(("\t"))

    def writesInt(self, list_object):
        for i in range (0, len(list_object)):
            self.file.write(("%d" % list_object[i]))
            if i == (len(list_object) - 1):
                self.file.write(("\n"))
            else:
                self.file.write(("\t"))

    def write(self, value_object, add_str = ("\n")):
        self.file.write(("%.6e" % value_object))
        self.file.write(add_str)

    def write10(self, value_object, add_str = ("\n")):
        self.file.write(("%.10e" % value_object))
        self.file.write(add_str)

    def writeInt(self, value_object, add_str = ("\n")):
        self.file.write(("%d" % value_object))
        self.file.write(add_str)

    def writeStr(self, value_object, add_str = ("\n")):
        self.file.write(("%s" % value_object))
        self.file.write(add_str)


def _main(sys_argv):
    original_file = ("")
    ground_truth_label_file = ("")
    predict_label_file = ("")
    #
    original_root_path = ("")
    ground_truth_label_root_path = ("")
    predict_label_root_path = ("")
    #
    metric_list = [('Dice'),('JI'),('precision'),('recall'),('specificity'),('ASD')]
    #
    case_phase_id_filelist_file = ("")
    output_file = ("")
    #
    do_use_resize_image = True
    is_abdominal_image = True
    #
    predict_label_fname = ("predict_label")
    ground_truth_label_fname = ("label")
    if is_abdominal_image:
        ground_truth_label_fname = ("LvSpPcLRKdGbwAAOIVCHrtStmPVLRLngSPVSMVLRRV_mask")
    #
    resize_str = ("_resize_512")
    if do_use_resize_image:
        ground_truth_label_fname = ground_truth_label_fname + resize_str
    #
    image_ext = (".mha")
    #
    if True:
        if len(sys_argv)==5:
            predict_label_root_path = str(sys_argv[1])
            ground_truth_label_root_path = str(sys_argv[2])
            case_phase_id_filelist_file = str(sys_argv[3])
            output_file = str(sys_argv[4])
        else:
            print(("1 : predict_label_root_path[str]"), predict_label_root_path)
            print(("2 : ground_truth_label_root_path[str]"), ground_truth_label_root_path)
            print(("3 : case_phase_id_filelist_file[.txt]"), case_phase_id_filelist_file)
            print(("4 : output_file[.txt]"), output_file)
            return
    else:
        predict_label_root_path = ("")
        ground_truth_label_root_path = ("")
        case_phase_id_filelist_file = ("")
        output_file = ("")
        #
        if True:
            pass
    #
    case_phase_id_list = None
    if os.path.exists(case_phase_id_filelist_file):
        case_phase_id_list = _get_filelist_contents_list(case_phase_id_filelist_file)
        if True:
            case_phase_id_list = _replace_list_str(case_phase_id_list, target_str=("\\"), to_str=("/"))
    #
    if case_phase_id_list is None:
        return
    #
    len_of_case_phase_id_list = len(case_phase_id_list)
    #
    estimated_labels = []
    groundtruth_labels = []
    for i in range(0,len_of_case_phase_id_list):
        case_phase_id = case_phase_id_list[i]
        predict_label_file = os.path.join(predict_label_root_path, case_phase_id, predict_label_fname + image_ext)
        ground_truth_label_file = os.path.join(ground_truth_label_root_path, case_phase_id, ground_truth_label_fname + image_ext)
        #
        estimated_labels.append(predict_label_file)
        groundtruth_labels.append(ground_truth_label_file)
    #
    process(estimated_labels=estimated_labels, groundtruth_labels=groundtruth_labels, output_file=output_file, case_phase_id_list=case_phase_id_list,class_name_list=None, metric_list = metric_list,xlsx_ext=(".xlsx"),do_show=True)


def process_core(est_file, gt_file,metric_list=[("Dice")],class_name_list=None, patient=None,est_target_label=None,gt_target_label=None,do_show=False,do_use_medpy=False):
    est = None
    gt = None
    spacing = None
    origin = None
    est_spacing = None
    est_origin = None
    if os.path.exists(est_file):
        est, est_spacing, est_origin = load_volume(est_file)
    else:
        if do_show:
            if True:
                print(("est_file"),est_file)
    if spacing is None:
        spacing = est_spacing
    if origin is None:
        origin = est_origin
    gt_spacing = None
    gt_origin = None
    if os.path.exists(gt_file):
        gt, gt_spacing, gt_origin  = load_volume(gt_file)
    else:
        if do_show:
            if True:
                print(("gt_file"),gt_file)
    #
    if spacing is None:
        spacing = gt_spacing
    if origin is None:
        origin = gt_origin
    #
    result, n_classes = process_core_of_core(est=est, gt=gt, spacing=spacing, origin=origin, metric_list=metric_list,class_name_list=class_name_list,est_target_label=est_target_label,gt_target_label=gt_target_label,do_show=do_show,do_use_medpy=do_use_medpy)
    return result, n_classes

def process_core_of_core(est, gt, spacing, origin,metric_list=[("Dice")],class_name_list=None,est_target_label=None,gt_target_label=None,do_show=False,do_use_medpy=False):
    import scipy.ndimage
    #
    len_of_class_name_list = None
    if class_name_list is not None:
        len_of_class_name_list = len(class_name_list)
    #
    est_multi = None
    if est is not None:
        if est_target_label is not None:
            est_multi = est
            est = np.zeros_like(est_multi)
            if isinstance(est_target_label, (tuple,list)):
                len_of_est_target_label = len(est_target_label)
                for i in range(0,len_of_est_target_label):
                    i_p1 = i + 1
                    tl = est_target_label[i]
                    est_target_label_region = est_multi == est_target_label
                    est[est_target_label_region] = i_p1#1

            else:
                est_target_label_region = est_multi == est_target_label
                est[est_target_label_region] = 1
    #
    gt_multi = None
    if gt is not None:
        if gt_target_label is not None:
            gt_multi = gt
            gt = np.zeros_like(gt_multi)
            if isinstance(gt_target_label, (tuple,list)):
                len_of_gt_target_label = len(gt_target_label)
                for i in range(0,len_of_gt_target_label):
                    i_p1 = i + 1
                    tl = gt_target_label[i]
                    gt_target_label_region = gt_multi == gt_target_label
                    gt[gt_target_label_region] = i_p1#1

            else:
                gt_target_label_region = gt_multi == gt_target_label
                gt[gt_target_label_region] = 1
    #
    result = None
    n_classes = None
    if est is not None and gt is not None and spacing is not None:
        zoom = [g/float(e) for g, e in zip(gt.shape, est.shape)]
        est = scipy.ndimage.zoom(est, zoom, order=0)
        est = est.astype(gt.dtype)
        if do_use_medpy:
            result, n_classes = evaluate_labels_medpy(metric_list, est, gt, spacing=spacing,n_classes=len_of_class_name_list,do_remove_background_classess_already=True)
        else:
            result, n_classes = evaluate_labels(metric_list, est, gt, spacing=spacing,n_classes=len_of_class_name_list,do_remove_background_classess_already=True)
        if False:#True:
            print(result)
            print(n_classes)
        #
    #
    return result, n_classes


def process(estimated_labels, groundtruth_labels, output_file, case_phase_id_list=None,class_name_list=None, metric_list = [('Dice'),('ASD')],xlsx_ext=(".xlsx"),correct_fname=("correct"), mistake_fname=("mistake"),do_show=True, est_target_label=None, gt_target_label=None):
    import glob
    import tqdm
    import os
    import scipy.ndimage

    if False:#True:#False:#True:#False:
        if case_phase_id_list is not None:
            print(case_phase_id_list)
        if estimated_labels is not None:
            print(estimated_labels)
        if groundtruth_labels is not None:
            print(groundtruth_labels)
        if True:
            print()
    #
    out_filename_path, out_path, out_filename, out_ext = _make_dir_and_path_by_rootpath_and_filename(output_file)
    out_correct_file = out_filename_path + out_filename + ("_") + correct_fname + out_ext
    out_mistake_file = out_filename_path + out_filename + ("_") + mistake_fname + out_ext
    out_correct_writer = _FileWriter(out_correct_file)
    out_mistake_writer = _FileWriter(out_mistake_file)
    #
    len_of_class_name_list = None
    if class_name_list is not None:
        len_of_class_name_list = len(class_name_list)

    results = []
    n_classes = None
    counter = 0
    for p, est_file, gt_file in tqdm.tqdm(zip(case_phase_id_list, estimated_labels, groundtruth_labels)):
        result = None
        if True:
            result, n_classes = process_core(est_file=est_file, gt_file=gt_file,metric_list=metric_list,class_name_list=class_name_list, patient=p,est_target_label=est_target_label,gt_target_label=gt_target_label,do_show=do_show)
            if result is not None:
                out_correct_writer.writeStr(p)
            else:
                out_mistake_writer.writeStr(p)
        else:
            est = None
            gt = None
            spacing = None
            origin = None
            if os.path.exists(est_file):
                est, _, _ = load_volume(est_file)
            else:
                if do_show:
                    if True:
                        print(("est_file"),est_file)
            if os.path.exists(gt_file):
                gt, spacing, origin  = load_volume(gt_file)
            else:
                if do_show:
                    if True:
                        print(("gt_file"),gt_file)
            #
            est_multi = None
            if est is not None:
                if est_target_label is not None:
                    est_multi = est
                    est = np.zeros_like(est_multi)
                    if isinstance(est_target_label, (tuple,list)):
                        len_of_est_target_label = len(est_target_label)
                        for i in range(0,len_of_est_target_label):
                            i_p1 = i + 1
                            tl = est_target_label[i]
                            est_target_label_region = est_multi == est_target_label
                            est[est_target_label_region] = i_p1#1

                    else:
                        est_target_label_region = est_multi == est_target_label
                        est[est_target_label_region] = 1
            #
            gt_multi = None
            if gt is not None:
                if gt_target_label is not None:
                    gt_multi = gt
                    gt = np.zeros_like(gt_multi)
                    if isinstance(gt_target_label, (tuple,list)):
                        len_of_gt_target_label = len(gt_target_label)
                        for i in range(0,len_of_gt_target_label):
                            i_p1 = i + 1
                            tl = gt_target_label[i]
                            gt_target_label_region = gt_multi == gt_target_label
                            gt[gt_target_label_region] = i_p1#1

                    else:
                        gt_target_label_region = gt_multi == gt_target_label
                        gt[gt_target_label_region] = 1
            #
            #result = None
            if est is not None and gt is not None and spacing is not None:
                zoom = [g/float(e) for g, e in zip(gt.shape, est.shape)]
                est = scipy.ndimage.zoom(est, zoom, order=0)
                est = est.astype(gt.dtype)
                result, n_classes = evaluate_labels(metric_list, est, gt, spacing=spacing,n_classes=len_of_class_name_list,do_remove_background_classess_already=True)
                if False:#True:
                    print(result)
                    print(n_classes)
                out_correct_writer.writeStr(p)
            else:
                if do_show:
                    if True:
                        if p is not None:
                            print(p)
                        if est is not None:
                            print(("est"),type(est))
                        if gt is not None:
                            print(("gt"),type(gt))
                        if spacing is not None:
                            print(("spc"),type(spacing))

                out_mistake_writer.writeStr(p)
        #
        results.append(result)
        counter = counter + 1
    #
    if True:
        del out_correct_writer
        del out_mistake_writer
    #
    if True:
        print()
        print(type(results))
        print(len(results))
        print()
    #
    if n_classes is not None:
        if class_name_list is None:
            n_classes_minus_one = n_classes - 1
            class_name_list = [ str(i) for i in range(0,n_classes_minus_one) ]
    #
    dirname, basename, ext = _composite_path_ver2(output_file)
    out_file = dirname + basename + xlsx_ext
    #
    results_array = None
    if results is not None:
        results_array = np.array(results)
        #
        if results_array is not None:
            if True:
                print(("results_array"))
                print(type(results_array))
                print(results_array.ndim)
                print(results_array.dtype)
                print(results_array.shape)
    #
    if results_array is not None:
        if results_array.ndim==3:
            transpose_tuple = (2, 0, 1)
            #(4,17,5)
            results_array = np.transpose(results_array, transpose_tuple)
        elif results_array.ndim==1:
            if True:
                print(("error, results_array.ndim==1, please check."))
            results_array_is_None = results_array == None
            if results_array_is_None.all():
                print(("results_array.all()"))
            elif results_array_is_None.any():
                print(("results_array.any()"))
        else:
            print(("results_array can't transpose."))
        #
        if results_array is not None:
            if True:
                print(("results_array(transposed)"))
                print(type(results_array))
                print(results_array.ndim)
                print(results_array.dtype)
                print(results_array.shape) #(Met,Pat,Org) # (10,20,22)
    #len_of_results_array = len(results_array)
    #
    import xlsxwriter
    workbook = xlsxwriter.Workbook(out_file)
    #
    #
    for i in range(len(metric_list)):
        metric = metric_list[i]
        metric_str = ("_") + metric
        out_metric_file = dirname + basename + metric_str  + ext
        out_metric_writer = _FileWriter(out_metric_file)
        worksheet = workbook.add_worksheet(metric)

        if class_name_list is not None:
            for j in range(len(class_name_list)):
                worksheet.write(j+1, 0, class_name_list[j])

        if case_phase_id_list is not None:
            for j in range(len(case_phase_id_list)):
                worksheet.write(0, j+1, case_phase_id_list[j])

        if class_name_list is not None and case_phase_id_list is not None:
            for j in range(len(class_name_list)):
                for k in range(len(case_phase_id_list)):
                    data = results[k][j][i]
                    is_nan_data = np.isnan(data)
                    is_inf_data = np.isinf(data)
                    j_1 = j+1
                    k_1 = k+1
                    if is_nan_data or is_inf_data:
                        if is_nan_data:
                            data = ("nan")
                        elif is_inf_data:
                            data = ("inf")
                        worksheet.write(j_1, k_1, data)
                    else:
                        worksheet.write(j_1, k_1, float(data))
                    #
                #
        metric_array = results_array[i]
        #
        if metric_array is not None:
            len_of_metric_array = len(metric_array)
            for j in range(0,len_of_metric_array):
                patient_metric_array = metric_array[j]
                out_metric_writer.writes(patient_metric_array)
        #
        if True:
            del out_metric_writer
    #
    workbook.close()

    

if __name__ == '__main__':
    _main(sys.argv)
    
