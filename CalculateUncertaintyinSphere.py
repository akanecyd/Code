import os
import numpy as np
import pandas as pd
from utils import mhd
from skimage.morphology import ball
# from utils.io import read_datalist
import tqdm
import logging
# logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')

def get_center_coords(point, es, offset):
    return [
            int(np.abs((point[0]-offset[0])/es[0])),
            int(np.abs((point[1]-offset[1])/es[1])),
            int(np.abs((point[2]-offset[2])/es[2])),
     ]

def recenter_roi(roi, sphere, center, r):
    roi[sphere[0]+ center[0]-r,
        sphere[1]+ center[1]-r,
        sphere[2]+ center[2]-r] = 1
    return roi

def get_mean_uncert(roi, vol):
    return np.mean(vol[roi==1])

logging.info('Initialization...')
# with open('//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt') as f:
#     cases = f.read().splitlines()
with open('//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt') as f:
    cases = f.read().splitlines()
# cases = read_datalist()
sides = ['right', 'left']
organs = ['vein', 'artery']
r = 3
sphere = np.where(ball(r)==1)
logging.info(f'Sphere with radius {r} created...')

TGT = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyDistUncertainties_R3'
os.makedirs(TGT, exist_ok=True)

pbar = tqdm.tqdm(cases)
logging.info(f'Started processing...')
for _case in pbar:
    logging.info(f'Processing case {_case}...')
    uncert_f = f'//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/{_case}/{_case}-vessels_uncert.mhd'
    vol, hdr = mhd.read(uncert_f)
    for organ in organs:
        for side in sides:
            logging.info(f' Processing {side} {organ}')
            point_list_f = f'//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/PolyDistances_Vessel/{_case}_{organ}_{side}.csv'
            point_list = pd.read_csv(point_list_f, sep=",", header=None).to_numpy()
            
            _point_uncerts = []
            for _point in tqdm.tqdm(point_list, desc="Points"):
                _center = get_center_coords(_point, hdr['ElementSpacing'], hdr['Offset'])
                _ROI = np.transpose(np.zeros_like(vol),[2,1,0])
                _ROI = recenter_roi(_ROI, sphere, _center, r)
                _ROI = np.transpose(_ROI, [2,1,0])
                mean_uncert = get_mean_uncert(_ROI, vol)
                _point_uncerts.append(mean_uncert)
            out_file = os.path.join(TGT,f'{_case}_{organ}_{side}.csv')
            pd.DataFrame(_point_uncerts).to_csv(out_file, header=False, index=False)   
            logging.info(f"File {out_file} created...")

    logging.info(f'Case {_case} done...')



