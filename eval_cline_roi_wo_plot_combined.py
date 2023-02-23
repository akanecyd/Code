from pylab import rcParams
rcParams['figure.figsize'] = 8,8
from pylab import rcParams
rcParams['figure.figsize'] = 15,10
rcParams['font.family'] = 'serif'
rcParams["font.size"] = 25
import mhd
import boundingbox as bb
import binary_thinning3d as bt3
import numpy as np
import tqdm
import eval_vessel
import sklearn.metrics
import os
import json
roi_dir = r'S:\abdomen\exp\2018\20180703_manual_segmentation\roi'
idlist = [e for e in os.listdir(r'\\SALMON\User\y-suzuki\exp\20170520_LineFilter\output') if e.startswith('Osaka')]
margin = 1
#names = ['alex3_5_24', 'alex3_5_32', 'alex3_5_48', 'unet_48', 'unet_64', 'hessian']
#names = ['alex3_5_32', 'alex3_5_48', 'alex3_5_64','unet_48', 'unet_64', 'unet_80', 'hessian']
#names=  ['hessian']
#names = ['unet96']
# names = ['alex80','alex96']
# names = ['unet_d1_64', 'unet_d1_80']
names = ['alex96','unet80','hessian']
result_dir_base = r'S:\abdomen\exp\2018\20180724_vesselness_exp\exp1\patch_exp\output'

def pad_to(arr, shape):
    p = tuple((0, t-s) for s,t in zip(arr.shape, shape))
    return np.pad(arr, p, mode='constant')

#result_dir = r'\\SALMON\User\y-suzuki\abdomen\exp\2018\20180724_vesselness_exp\exp1\with_da\output\unet'
#result_dir = r'S:\abdomen\exp\2018\20180724_vesselness_exp\exp1\output\alex3_5_p48'
for name in names:
    print(name)
    num = 100
    result_dir = os.path.join(result_dir_base,name)
    outdir = 'combined_roi'
    os.makedirs(outdir,exist_ok=True)
    for roi_no in [1+1,1+3]:
        aucs = []
        d_truths = []
        cropped_results = []
        thin_truths = []

        for ID in idlist:
        #    truth,h = mhd.read(r'\\SALMON\User\ono\2017\data\{}\LeftKidney_bboxartery.mha'.format(ID))
            truth,h = mhd.read(os.path.join(roi_dir,'{}_manual.mha'.format(ID)))
            bbox = bb.bbox(truth==roi_no)
            cropped_truth = bb.crop(truth, bbox, margin=margin)
            cropped_truth = cropped_truth == 1
            thin_truth = bt3.thinning(np.ascontiguousarray(cropped_truth))
            d_truth = eval_vessel.dilate(thin_truth,2)
    
            result,h = mhd.read(os.path.join(result_dir,'{}.mha'.format(ID)))
            cropped_result = bb.crop(result, bbox, margin=margin)

            thin_truth = np.pad(thin_truth, 1, mode='constant')
            d_truth = np.pad(d_truth, 1, mode='constant')
            cropped_result = np.pad(cropped_result, 1, mode='constant')
            thin_truths.append(thin_truth)
            d_truths.append(d_truth)
            cropped_results.append(cropped_result)

        max_shape = np.array([d.shape for d in d_truths]).max(axis=0)
        d_truths = [pad_to(d, max_shape) for d in d_truths]
        d_truth = np.concatenate(d_truths, axis=0)
        cropped_results = [pad_to(c, max_shape) for c in cropped_results]
        cropped_result = np.concatenate(cropped_results, axis=0)
        thin_truths = [pad_to(t, max_shape) for t in thin_truths]
        thin_truth = np.concatenate(thin_truths, axis=0)


        s,e = np.min(cropped_result), np.max(cropped_result)
        threshs = s+((e-s)/num)*np.arange(num)

        values = []
        prev = cropped_result
        for thresh in tqdm.tqdm(reversed(threshs),desc='calculate', total=len(threshs)):
            label = cropped_result >= thresh
#                if np.array_equal(prev,label): # thresholding result is identical to the previous result
#                    values.append(values[-1]) # reuse the previous values
#                    continue
            prev = label
            thin_result = bt3.thinning(label)
            r_truth, r_result = eval_vessel.evaluate(thin_truth, thin_result,margin=2,d_truth=d_truth)
            tp,fn = np.sum(r_truth==1), np.sum(r_truth==3)
            if (tp+fn)==0:
                tpr = 1
            else:
                tpr = tp/(tp+fn)
            tp,fp = np.sum(r_result==1), np.sum(r_result==2)
            if (tp+fp)==0:
                prec = 1
            else:
                prec = tp/(tp+fp)
            values.append((tpr,prec))

        values.append((1,0))
        precision = [v[1] for v in values]
        recall = [v[0] for v in values]
        sorted_i = np.argsort(recall)
        recall = [recall[i] for i in sorted_i]
        precision = [precision[i] for i in sorted_i]

   
        with open('{}/{}_{}.json'.format(outdir,roi_no,name),'w') as f:
            json.dump({'recall':recall,'precision':precision},f,indent=2)
        #        f.write(json.dump())
        
    #    r_auc = np.reshape(np.array(aucs),(-1,3))
        #np.savetxt('{}/{}_aucs.csv'.format(outdir,roi_no),np.array(aucs),delimiter=',')
        #print(np.mean(r_auc,axis=0)*100)
        #import mhd
        #mhd.write('values_roi{}/aucs.mha'.format(roi_no),r_auc)
