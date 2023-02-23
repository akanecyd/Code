import numpy as np
import matplotlib.pyplot as plt
import boundingbox as bb
import os
import tqdm
import time
import eval_vessel
from skimage.morphology import skeletonize
from utils import mhd
import pandas as pd
import sklearn.metrics


def get_pr_values(truth_skeleton,
                  pre_probability):
    pre_probability = pre_probability / 1000
    threshs = np.arange(0.0, 1.0, 0.001)
    threshs = np.append(threshs, [0.0, 1.0])
    # threshs = np.append(threshs, np.arange(0, 1.0, 0.005))

    values = []
    for thresh in tqdm.tqdm(threshs, desc='calculate', total=len(threshs)):
        # for timeit.timeit('"-".join( for thresh in threshs)', number=len(threshs)):
        pre_label = pre_probability > thresh
        pre_label = pre_label == 1
        pre_label_skeleton = skeletonize(pre_label)

        r_truth, r_result = eval_vessel.evaluate(truth_skeleton, pre_label_skeleton, margin=2)

        tp, fn = np.sum(r_truth == 1), np.sum(r_truth == 3)
        if (tp + fn) == 0:
            tpr = 1
        else:
            tpr = tp / (tp + fn)
        tp, fp = np.sum(r_result == 1), np.sum(r_result == 2)
        if (tp + fp) == 0:
            prec = 1
        else:
            prec = tp / (tp + fp)
        print('___threshold:{}___tpr:{}___prec:{}'.format(thresh, tpr, prec))
        values.append((tpr, prec))

    values.append((1, 0))
    precision = [v[1] for v in values]
    recall = [v[0] for v in values]
    sorted_i = np.argsort(recall)
    recalls = [recall[i] for i in sorted_i]
    precisions = [precision[i] for i in sorted_i]

    return recalls, precisions


if __name__ == '__main__':
    def mkdir(path):
        folder = os.path.exists(path)

        if not folder:
            os.makedirs(path)
            print("---  new folder:{} --- ", path)

# truth_dir = '//SALMON/User/Chen/Vessel_data/recover_original_interpolated'
# predict_dir = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_with_muscles_original20'
truth_dir = predict_dir = '//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/wo_muscles_plain/crop_2.0'

# case_IDs = ['k1657', 'k1756', 'k1802', 'k1873', 'k1565', 'k8892', 'k1631', 'k1870', 'k1647', 'k1677']
# case_IDs = ['K10387', 'K7510', 'K8041', 'K8454', 'K8559', 'K8574',
#              'K8699', 'K8748', 'K8772', 'K8795', 'K8895', 'K9020', 'K9086',
#              'K9089', 'K9162', 'K9193', 'K9204', 'K9622', 'K9831', 'K9861']
case_IDs = ['N0018', 'N0024', 'N0031', 'N0047', 'N0056', 'N0074', 'N0076',
            'N0091', 'N0094', 'N0107', 'N0108', 'N0116', 'N0132', 'N0133', 'N0140',
            'N0144', 'N0145', 'N0152', 'N0171', 'N0180', 'N0187']

# class_list = ['vein', 'artery']
class_list = ['artery']
n_class = 1

for case_ID in case_IDs:
    print('___process_{}___'.format(case_ID))
    out_dir = os.path.join('//Salmon/User/Chen/Result/Nara_skeleton_20220613', case_ID)
    mkdir(out_dir)
    # load data
    label_vol, hdr = mhd.read(os.path.join(truth_dir, '{}_artery_label.mhd'.format(case_ID)))
    head_in = {'CompressedData': True, 'ElementSpacing': hdr['ElementSpacing']}

    for i in range(1, n_class + 1):  # n_class + 1
        organ_label = np.zeros_like(label_vol)
        organ_label[label_vol == i+1] = 1
        # crop vessel region
        bbox = bb.bbox(organ_label)
        cropped_truth = bb.crop(organ_label, bbox, margin=1)
        cropped_truth = cropped_truth == 1
        organ_name = class_list[i - 1]
        mhd.write(os.path.join(out_dir, case_ID + '_{}_crop_label.mhd'.format(organ_name)), cropped_truth, head_in)
        print('____extract_{}_{}_skeleton____'.format(case_ID, organ_name))
        # extract vessel skeleton
        organ_skeleton = skeletonize(cropped_truth)
        mhd.write(os.path.join(out_dir, case_ID + '_{}_skeleton.mhd'.format(organ_name)), organ_skeleton, head_in)
        # load probability map and crop by the bounding box
        organ_probability, _ = mhd.read(
             # os.path.join(predict_dir, case_ID + '-{}_vessels_prob.mhd'.format(i + 22)))
             os.path.join(predict_dir, case_ID + '_crop_artery_prob.mhd'))
        cropped_probability = bb.crop(organ_probability, bbox, margin=1)
        mhd.write(os.path.join(out_dir, case_ID + '_{}_probability.mhd'.format(organ_name)), cropped_probability,
                  head_in)
        #  PR values calculation
        start = time.time()
        recalls, precisions = get_pr_values(organ_skeleton, cropped_probability)
        end = time.time()
        print('__{}__{}___0.001_threshold_time:{}s'.format(case_ID, organ_name, '%.4f' % (end-start)))
        # print(timeit.Timer("get_pr_values(organ_skeleton, cropped_probability)",
        #                    setup='from __main__ import get_pr_values').timeit())
        # ACUPR calculation
        organ_auc = sklearn.metrics.auc(recalls, precisions)
        print('___acu:{}___'.format(organ_auc))
        # plot PR curve
        plt.plot(recalls, precisions, linewidth=2)
        plt.gca().set_aspect('equal')
        plt.xlim([0, 1])
        plt.xlabel('Recall')
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.title('{}_ACUPR:{}'.format(organ_name, '%.4f' % organ_auc), loc='right')
        plt.savefig(os.path.join(out_dir, '{}_pr_curve.png'.format(organ_name)))
        plt.show()

        # with open(os.path.join(out_dir, '{}_values.json'.format(case_ID)), 'w') as f:
        #     json.dump({'recall': recalls, 'precision': precisions}, f, indent=2)
        df = pd.DataFrame({'recall': recalls,
                           'precisions': precisions})
        df.to_csv(os.path.join(out_dir, '{}_PR_values.csv'.format(organ_name)), index=False)

    # with open(os.path.join(out_dir, '{}_PR_values.csv'.format(organ_name)), 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(recalls)
    #     writer.writerow(precisions)
