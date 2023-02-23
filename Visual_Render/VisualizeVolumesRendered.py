import os
import glob
import pandas as pd
import numpy as np
import shutil
import vis_utils
import vtk
from joblib import Parallel, delayed
import tqdm

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

def get_cd(path):
    img_name = os.path.basename(path)
    if 'R1' in path:
        cd = 750
    elif ('R2' in path) or ('R3' in path):
        cd = 1100
    elif ('R4' in path) or ('R5' in path):
        cd = 1000
    elif ('R6' in path) or ('R7' in path):    
        cd = 500
    else:
        cd = 1200
    return cd

def read_datalist(fpath, field='fileID'):
    datalist = []
    if fpath.endswith('.txt'):
        datalist = np.genfromtxt(fpath, dtype=str)
    elif fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        print('Dataframe: ', df)
        datalist = df[field].values.tolist()
    return datalist

# Path to the .mha file
# _ROOT = None
# _ROOT = 'E:/Projects/Lower Limbs/Codes/PostureNormalization/ResultsTransformix'
# _ROOT = 'Z:/otake/Collaboration/KyotoUniv_PlasticSurgery/20210517_PreliminaryExperiment_dataset/MHD_2'
# _ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2'
# _ROOT = 'Z:/otake/Collaboration/KyotoUniv_PlasticSurgery/20210517_PreliminaryExperiment_dataset/MHD'
_LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/KCT_osaka_muscles'
# _LABEL_ROOT = 'E:/Projects/Lower Limbs/Data/100_Osaka_Bones/Auto Labels/Step 2'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/KyotoPlastic/10-fold_NMAR_5000'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/2_osaka_roi/femur'
# _LABEL_ROOT = 'Z:/otake/Collaboration/KyotoUniv_PlasticSurgery/20210517_PreliminaryExperiment_dataset/MHD'
# _LABEL_ROOT = 'Z:/chen/crop_CT_label+skin'
# _LABEL_ROOT = 'E:/Projects/Lower Limbs/NII/Pelvis/Cropped_Revised_Labels'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/chen_nara_plain'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/2_osaka_roi/20210828_merged'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/2_osaka_roi/20210831_femur_hip_roi/Merged/Resized ROI'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/chen_nara_paired'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/10_osaka_vessels_only'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/10_osaka_vessels'
# _ROOT = 'Z:/otake/Collaboration/KyotoUniv_PlasticSurgery/20210517_PreliminaryExperiment_dataset/MHD'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/2_osaka'
# _LABEL_ROOT = 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/Pass2'#
#_FILE_LIST = 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/Patients_List.txt'
# _FILE_LIST = 'Z:/uemura/NaraMed_ArteryCT/patient_list.txt'
# _FILE_LIST = 'Z:/otake/Collaboration/KyotoUniv_PlasticSurgery/20210517_PreliminaryExperiment_dataset/patients_list.txt'
_FILE_LIST = None
# _FILE_LIST = 'Z:/chen/crop_CT_label+skin/Patients_List.txt'#'Z:/chen/crop_CT_label+skin/patients_list.txt'
# _FILE_LIST = 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/Patients_List.txt' # 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/Patients_List.txt'#'Z:\chen\crop_CT_label+skin/patients_list.txt'
# _FILE_LIST = 'Z:/uemura/NaraMed_ArteryCT/patient_list.txt' # 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/Patients_List.txt'#'Z:\chen\crop_CT_label+skin/patients_list.txt'
# _FILE_LIST = 'Z:/mazen/LowerLimbs/NII/20210916_pelvis_uncertainty_strat_sample_ids.csv'
# _FILE_LIST = 'Z:/intern/nakajima/Patients_List_o.txt'
# _ROI_ROOT = 'Z:/uemura/Iwasa_muscle_data/ROIs' #'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/ROIs'
_ROI_ROOT = None #'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/ROIs'
# _OUT_DIR = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/10_osaka_vessels/Visualizations'
# _OUT_DIR = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/2_osaka_roi/merged_label/Visualizations'
# _OUT_DIR =  'Z:/mazen/LowerLimbs/NII/20210903_Cropped_Label_Files_Visualizations' #'Z:/uemura/Ando_data/Ando_CT_MHD/Visualizations'
_OUT_DIR =  _LABEL_ROOT + '/Visualizations'#'E:/Projects/Lower Limbs/NII/Pelvis/Visualizations'
# _OUT_DIR = 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/ROIs/Visualizations_ROI'
# _OUT_DIR = 'Z:/mazen/LowerLimbs/HipSegmetnation2/results/2_osaka_roi/20210831_femur_hip_roi/Visualizations'
_LABEL_EXT = '-muscles-label.mhd' #% (_TAG) # '-vessel-label.mhd'
_IMAGE_EXT = '_skin_moved*.mhd' #% (_TAG) # '-vessel-label.mhd'
_SKIN_EXT = '_cropped_label.mhd' 
_TGTS = ['label'] #,'label' 'label_foot' 'ct_bone', 'ct_bone', 'ct_muscle_bone' 'label_vessel', 
os.makedirs(_OUT_DIR, exist_ok=True)
shutil.copy(__file__, os.path.join(_OUT_DIR, 'Visualization_Script.py')) 
_N_JOBS = 4
_X, _Y = 500, 500
# cd = 900  # Pelvis: 700, Hip all: 1250  ROI: 1000 Lower all: 1900
# angles = {'RL': [0,90],
#           'LR': [0,270],
#           'AP': [0,180],
#           'IS': [-90,-90],
#           'SI': [90,0],
#           'PA': [180,-180]}
angles = {'PA': [180,180],
          'AP': [0,180] }


bb_colors = [
             [0,0,0],
             [1,0,0],
             [0,1,0],
             [0,0,1],
             [1,1,0],
             [1,0,1],
             [0,1,1]
            ]

# imagePath = 'Z:/otake/Collaboration/KyotoUniv_PlasticSurgery/20210517_PreliminaryExperiment_dataset/MHD/KPS0001_image.mhd'
# imagePath = '//Scallop/User/uemura/Iwasa_muscle_data/interpolated_mhd/K6940_interpolated.mhd'
# skinPath = './samples/17035395872_1210171051807706_4-skin-label.mhd'
clipping = None # Clipping option

def visualize_image(image, tgt,roi_num=None):
    # print( image + _LABEL_EXT)
    if 'label' in tgt:
        imagePath=image
        case_id = os.path.dirname(imagePath).split('\\')[-1]
        # imagePath = os.path.join(_LABEL_ROOT,image, 'cropped','R%s_' % roi_num + _LABEL_EXT) #_LABEL_ROOT,image,_LABEL_EXT
        # imagePath = glob.glob(os.path.join(_LABEL_ROOT,'%s*R%s*%s' % (image,roi_num,_LABEL_EXT)))[0]
        _INTERP = 'nearest'
        _SKIN = None
        vu = [0,0,1]
        cam_config = {'ambient': 0.5,
                      'diffuse': 0.5,
                      'specular': 0.2,
                      'spec_power': 10.0}
        if 'R6' in imagePath:
            print('%s Flipped' % (imagePath))
            _FLIP = True
        else:
            _FLIP = False
    else:
        imagePath=image
        case_id = os.path.dirname(imagePath).split('\\')[-1]
        # imagePath = os.path.join(_ROOT,'%s%s' % (image,_IMAGE_EXT)) #, image+_IMAGE_EXT
        # imagePath = os.path.join(_ROOT,image, 'cropped_simple', 'R%s%s' % (roi_num,_IMAGE_EXT)) #, image+_IMAGE_EXT
        # imagePath = os.path.join(_ROOT, image, image + '_R%s_512_512_img.mhd' % (roi_num))
        # imagePath = glob.glob(os.path.join(_ROOT, image, '%s*R%s*%s' % (image,roi_num,_IMAGE_EXT)))[0]
        _INTERP = 'linear'
        _SKIN = None#True
        vu = [0,0,1]
        cam_config = {'ambient': 0.12,
                      'diffuse': 1.0,
                      'specular': 0.0,
                      'spec_power': 10.0}
        if 'R7' in imagePath:
            _FLIP = True
        else:
            _FLIP = False

    cd = get_cd(imagePath)   
    color_dict, opacity_dict, gopacity_dict = vis_utils.get_color_configs(tgt)
    _CASE = os.path.basename(imagePath).split('_')[0]
    _EXT = os.path.basename(imagePath).split('.')[-1]
    # print('Case: %s Ext: %s Target: %s' % (_CASE, _EXT, tgt))
    # bbPath = glob.glob(os.path.join(_ROI_ROOT, _CASE, '*.acsv'))
    bbPath = None

    #Create reader
    if _EXT in ['mha', 'mhd']:
        reader = vis_utils.get_meta_reader(imagePath)
        intc =  None
    elif _EXT in ['gz']:
        reader = vis_utils.get_nifty_reader(imagePath)
        intc = reader.GetRescaleIntercept()
    elif _EXT in ['nrrd']:
        reader = vis_utils.get_nrrd_reader(imagePath)
        intc =  None
    else:
        raise NotImplementedError
    # castFilter = vis_utils.cast_reader(reader)
    if intc:
        if intc != 0.0:
            imgMath = vtk.vtkImageMathematics()
            imgMath.SetInputConnection(reader.GetOutputPort())
            imgMath.SetOperationToAddConstant()
            imgMath.SetConstantC(intc)
            imgMath.Update()
            imageData = imgMath.GetOutput()
    else:
        imageData = reader.GetOutput()

    #Create transfer functions
    color_trans_func = vis_utils.get_color_trans_func(color_dict)
    opacity_trans_func = vis_utils.get_scalar_funct(opacity_dict.keys(),
                                                    scalar=opacity_dict.values())
    gradient_opacity_func = vis_utils.get_scalar_funct(gopacity_dict.keys(),
                                                    scalar=gopacity_dict.values())
    prop = vis_utils.get_volume_property(color_trans_func,
                                        opacity_trans_func,
                                        gradient_opacity_func,
                                        interp=_INTERP,
                                        **cam_config)
    # Render
    c1 = vis_utils.get_label_center(imageData,label=16)
    # print(c)
    # c1 = imageData.GetCenter()
    # print(c1)
    

    # Renderers
    for _dir, angle in angles.items(): 
        # print('Processing %s Angle: %d' % (_CASE, angle))
        if _SKIN:
            skinPath = os.path.join(_LABEL_ROOT, image + _SKIN_EXT)
            print(skinPath)
            skin_reader = vis_utils.get_meta_reader(skinPath)
            skinData = skin_reader.GetOutput()
            mask = vtk.vtkImageMask()
            mask.SetImageInputData(imageData)
            # print('Mask Image Input Set')
            if _FLIP_SKIN:
                flip_1 = vtk.vtkImageFlip()
                flip_1.SetInputData(skinData)
                flip_1.SetFilteredAxis(2)
                flip_1.Update()        
                flip_2 = vtk.vtkImageFlip()
                flip_2.SetInputData(flip_1.GetOutput())
                flip_2.SetFilteredAxis(1)
                flip_2.Update()        
                mask.SetMaskInputData(flip_2.GetOutput())
            else:
                caster = vtk.vtkImageCast()
                caster.SetInputData(skinData)
                caster.SetOutputScalarTypeToUnsignedChar()
                mask.SetMaskInputData(caster.GetOutput())
            mask.SetMaskedOutputValue(-1024)
            # mask.NotMaskOn()
            mask.Update()

            mapper = vis_utils.get_volume_mapper(mask.GetOutput())
        else:
            if _FLIP:    
                flip_1 = vtk.vtkImageFlip()
                flip_1.SetInputData(imageData)
                flip_1.SetFilteredAxis(0)
                flip_1.Update()        
                # flip_2 = vtk.vtkImageFlip()
                # flip_2.SetInputData(flip_1.GetOutput())
                # flip_2.SetFilteredAxis(0)
                # flip_2.Update()        
                imageData=flip_1.GetOutput()
            mapper = vis_utils.get_volume_mapper(imageData)

        # Add actors
        actor = vis_utils.get_volume_actor(mapper, prop)
        # actor.SetOrientation(0,0,0)
        # c1 = actor.GetCenter()
        # c1 = [c1[0], c1[1], c1[2]-100]
        _txt = os.path.basename(imagePath) #case_id + '_'+ 
        text_actor = vis_utils.get_text_actor(_txt,loc=[20, _Y-20]) #image + '_' + 
        if clipping:
            plane_clip = vis_utils.get_plane_clip(imageData, angle=angle)
            mapper.AddClippingPlane(plane_clip)
        renderer = vis_utils.createDummyRenderer(c1, cd, angle, vu, tgt)
        renderer.AddActor(actor)  
        renderer.AddActor(text_actor)
        if bbPath:
            for i,bb in enumerate(bbPath):
                cube_actor = vis_utils.get_bb_actor(bb, bb_colors[i])
                renderer.AddActor(cube_actor)
        # Show the windows
        renWin= vis_utils.get_renderer_window(renderer,_x=_X, _y=_Y) #)_x=700, _y=900
        w2if = vis_utils.get_window_to_image_filter(renWin)
        # vis_utils.run_renderer_window(renWin) # For debugging
        # out_image_path = os.path.join(_OUT_DIR,image+'-'+ _TAG +'_R%s-%d_%s.png' % (roi_num,angle, tgt)) #image+'-'+ 
        # out_image_path = os.path.join(_OUT_DIR,
                #  os.path.basename(imagePath).replace('.mhd', '_%s_%s_R%s.png' % (tgt,angle, roi_num))) #+'-'+os.path.basename(imagePath)
        out_image_path = os.path.join(_OUT_DIR,
                os.path.basename(imagePath).replace('.mhd', '_%s_%s.png' % ( tgt,_dir))) #+'-'+os.path.basename(imagePath) case_id + '_'+
        # out_image_path = './test.png'
        vis_utils.write_image(out_image_path, w2if)
        renderer.Clear()
        # print('%s Angle %d %d... done' % (os.path.basename(imagePath), angle[0], angle[1]))

# visualize_image('KPS0001', 'ct_bone')
# visualize_image('KPS0001', 'ct_muscle_bone')

if __name__ == '__main__':
    if _FILE_LIST:
        img_list = read_datalist(_FILE_LIST)
        # img_list = [_CASE.lower() for _CASE in img_list]
    else:
        img_list = glob.glob(os.path.join(_LABEL_ROOT,'*'+_LABEL_EXT ))
    print(*img_list, sep='\n')
    # for roi_num in range(1,8):
    pbar = tqdm.tqdm(img_list)
    Parallel(n_jobs = _N_JOBS)(delayed(visualize_image)(img, tgt) for img in pbar for tgt in _TGTS)
    # visualize_image(img_list[0], _TGTS[0])
