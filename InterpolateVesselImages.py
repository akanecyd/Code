import os
import numpy as np

from utils import mhd
import tqdm

from scipy.interpolate import RegularGridInterpolator as interp

_ROOT = 'C:/Users/cheny/Desktop/vincent/vincent_to_mhd/vein_artery_nerve/crop_CT_label'
_CASES = ['K8559', 'K8795', 'K8895', 'K9020', 'K9086', 'K9162', 'K9193', 'K9204']
# _CASES = ['K8574', 'K8699']
_EXT_IMG_IN = 'crop_original_image.mhd'
_EXT_IMG_OUT = 'image.mhd'
_EXT_LBL_IN = 'crop_label_image.mhd'
_EXT_LBL_OUT = 'vessels_nerves_label.mhd'
_OUT_ES = 1

def resize_img(img,in_es,out_es, method='linear', cval = -1000):
    #Shapes
    in_sz = img.shape

    zi, yi, xi = np.arange(0, in_es[2]*in_sz[0], in_es[2]),\
                 np.arange(0, in_es[1]*in_sz[1], in_es[1]),\
                 np.arange(0, in_es[0]*in_sz[2], in_es[0])  
    print(xi.shape)
    print(yi.shape)
    print(zi.shape)
    #Linspaces
    zo, yo, xo = np.arange(0, in_es[2]*in_sz[0], out_es[2]),\
                 np.arange(0, in_es[1]*in_sz[1], out_es[1]),\
                 np.arange(0, in_es[0]*in_sz[2], out_es[0]) 
    print(zo.shape)
    print(xo.shape)
    print(yo.shape)                 

    #Grids
    zo_o ,xo_o, yo_o  = np.meshgrid(zo,xo,yo,
                                  sparse=False, 
                                  indexing='xy')

    print(xo_o.shape)
    print(yo_o.shape)
    print(zo_o.shape)

    #Interpolator
    interpolater = interp((zi, xi, yi), 
                            img, method=method, 
                            fill_value=cval,
                            bounds_error=False)
    img_out = interpolater ((zo_o,xo_o, yo_o))

    img_out = np.transpose(img_out, [1,0,2])
    print(img_out.shape)
    return img_out


for _case in tqdm.tqdm(_CASES):
    # os.makedirs(os.path.join(_ROOT,_case,'tmp'), exist_ok=True)
    case_pth = os.path.join(_ROOT,_case, _EXT_LBL_IN)
    img_in, hdr = mhd.read(case_pth)
    img_in = np.where(img_in> 3, 0, img_in)
    print(case_pth)
    print(img_in.shape)
    in_es = hdr['ElementSpacing']
    out_es = [in_es[0], in_es[1],_OUT_ES]
    
    img_out = resize_img(img_in, in_es, out_es, method='nearest').astype('uint8')[:-1,:,:]
    print(img_out.shape)
    out_case_pth = os.path.join(_ROOT,_case, _EXT_LBL_OUT)
    mhd.write(out_case_pth, img_out, header={'ElementSpacing': out_es, 
                                             'CompressedData':True})
                                             # 'Offset': offset})

    case_pth = os.path.join(_ROOT,_case, _EXT_IMG_IN)
    img_in, hdr = mhd.read(case_pth)
    print(case_pth)
    print(img_in.shape)
    in_es = hdr['ElementSpacing']
    out_es = [in_es[0], in_es[1],_OUT_ES]
    # offset = hdr['Offset']
    img_out = resize_img(img_in, in_es, out_es, method='linear').astype('int16')[:-1,:,:]
    print(img_out.shape)
    out_case_pth = os.path.join(_ROOT,_case, _EXT_IMG_OUT)
    mhd.write(out_case_pth, img_out, header={'ElementSpacing': out_es, 
                                             'CompressedData':True})
                                             # 'Offset': offset})


