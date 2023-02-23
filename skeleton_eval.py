import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
from skimage.morphology import skeletonize

import boundingbox as bb
import eval_vessel
from utils import mhd
from utils.ImageHelper import ImageHelper
from utils.VideoHelper import VideoHelper

pr_value_dir = '//Salmon/User/Chen/Result/skeleton_test_wo_muscles'

case_IDs = ['k1657', 'k1756', 'k1802', 'k1873', 'k1565', 'k8892', 'k1631', 'k1870', 'k1647', 'k1677']
ACUPR_vein = []
ACUPR_artery = []
for case_ID in tqdm.tqdm(reversed(case_IDs), desc='calculate', total=len(case_IDs)):
    with open(os.path.join(pr_value_dir, case_ID, 'vein_PR_values.csv')) as f:
        pr_values = csv.DictReader(f)
        precision = [r['precisions'] for r in pr_values]
    with open(os.path.join(pr_value_dir, case_ID, 'vein_PR_values.csv')) as f:
        pr_values = csv.DictReader(f)
        recall = [row['recall'] for row in pr_values]
        organ_auc = sklearn.metrics.auc(np.array(recall, dtype=float), np.array(precision, dtype=float))
        ACUPR_vein.append(organ_auc)

for case_ID in tqdm.tqdm(reversed(case_IDs), desc='calculate', total=len(case_IDs)):
    with open(os.path.join(pr_value_dir, case_ID, 'artery_PR_values.csv')) as f:
        pr_values = csv.DictReader(f)
        precision = [r['precisions'] for r in pr_values]
    with open(os.path.join(pr_value_dir, case_ID, 'artery_PR_values.csv')) as f:
        pr_values = csv.DictReader(f)
        recall = [row['recall'] for row in pr_values]
        organ_auc = sklearn.metrics.auc(np.array(recall, dtype=float), np.array(precision, dtype=float))
        ACUPR_artery.append(organ_auc)

df = pd.DataFrame({'case_ID': case_IDs,
                  'vein': reversed(ACUPR_vein),
                'artery': reversed(ACUPR_artery)})
df.to_csv(os.path.join(pr_value_dir, 'ACUpr_values.csv'), index= False)






    # for thresh in tqdm.tqdm(reversed(threshs), desc='calculate', total=len(threshs)):
    #     label = cropped_result > thresh
    #     if np.array_equal(prev, label):  # thresholding result is identical to the previous result
    #         values.append(values[-1])  # reuse the previous values
    #         continue
    #     prev = label
    #     thin_result = bt3.thinning(label)
    #     r_truth, r_result = eval_vessel.evaluate(thin_truth, thin_result, margin=2, d_truth=d_truth)
    #     tp, fn = np.sum(r_truth == 1), np.sum(r_truth == 3)
    #     if (tp + fn) == 0:
    #         continue
    #     tpr = tp / (tp + fn)
    #     tp, fp = np.sum(r_result == 1), np.sum(r_result == 2)
    #     if (tp + fp) == 0:
    #         continue
    #     prec = tp / (tp + fp)
    #     values.append((tpr, prec))
