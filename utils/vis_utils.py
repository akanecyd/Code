import os
import numpy as np
import csv
import vtk
from vtk.util import numpy_support

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

def get_color_configs(tgt):
    """
    Get color/opacity confiuration dictionaries 
    for volume rendering
    """   
    if tgt == 'ct_bone':
        filenameColorfile    = "./TFs/CT_bone_color_tf.txt"
        filenameOpacityfile  = "./TFs/CT_bone_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/CT_bone_gopacity_tf.txt"
    if tgt == 'ct_bone_pyvr':
        filenameColorfile    = "./TFs/CT_bone_color_tf_pyvr.txt"
        filenameOpacityfile  = "./TFs/CT_bone_opacity_tf_pyvr.txt"
        filenameGOpacityfile = "./TFs/CT_bone_gopacity_tf.txt"
    elif tgt == 'ct_bone_implant':
        filenameColorfile    = "./TFs/CT_bone_implant_color_tf.txt"
        filenameOpacityfile  = "./TFs/CT_bone_implant_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/CT_bone_gopacity_tf.txt"
    elif tgt == 'ct_muscle_bone':
        filenameColorfile    = "./TFs/CT_muscle_bone_color_tf_Otake.txt"
        filenameOpacityfile  = "./TFs/CT_muscle_bone_opacity_tf_Otake.txt"
        filenameGOpacityfile = "./TFs/CT_muscle_bone_gopacity_tf_Otake.txt"
    elif tgt == 'ct_muscle_bone_implant':
        filenameColorfile    = "./TFs/CT_muscle_bone_implant_color_tf.txt"
        filenameOpacityfile  = "./TFs/CT_muscle_bone_implant_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/CT_muscle_bone_gopacity_tf_Otake.txt"
    elif tgt == 'mr_muscle_bone':
        filenameColorfile    = "./TFs/MR_muscle_bone_color_tf.txt"
        filenameOpacityfile  = "./TFs/MR_muscle_bone_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/MR_muscle_bone_gopacity_tf.txt"
    elif tgt == 'ct_bone_main_muscle_sub':
        filenameColorfile    = "./TFs/CT_muscle_bone_color_tf_Otake.txt"
        filenameOpacityfile  = "./TFs/CT_bone_main_muscle_sub_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/CT_muscle_bone_gopacity_tf_Otake.txt"
    elif tgt == 'label_muscle_hip':
        filenameColorfile    = "./TFs/Labels_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_muscle_hip_mri':
        filenameColorfile    = "./TFs/Labels_mri_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_bone_hip':
        filenameColorfile    = "./TFs/Labels_bone_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_bone_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_muscle_hip_rib':
        filenameColorfile    = "./TFs/Labels_color_hip_rib_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_muscle_hip_rib_bones':
        filenameColorfile    = "./TFs/Labels_color_hip_rib_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_bone_only_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_muscl_hip_cross_vessel':
        filenameColorfile    = "./TFs/Labels_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_muscle_cross_vessel_trans_tf"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_muscle_lr_hip':
        filenameColorfile    = "./TFs/Labels_HipThighSkin_LR_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_HipThighSkin_LR_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_muscles_skin_gopacity_tf.txt"
    elif tgt == 'label_muscle_lr_hip_bright':
        filenameColorfile    = "./TFs/Labels_HipThighSkin_LR_bright_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_HipThighSkin_LR_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_muscles_skin_gopacity_tf.txt"
    elif tgt == 'label_muscle_bone_lr_hip_bright':
        filenameColorfile    = "./TFs/Labels_HipThighBoneMuscleSkin_LR_bright_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_HipThighBoneMuscleSkin_LR_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_muscles_skin_gopacity_tf.txt"
    elif tgt == 'label_pelvis':
        filenameColorfile    = "./TFs/Labels_Pelvis_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_pelvis_only':
        filenameColorfile    = "./TFs/Labels_Pelvis_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_pelvis_only_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_skin_only':
        filenameColorfile    = "./TFs/Labels_Skin_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_single_label_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_single_label_gopacity_tf.txt"        
    elif tgt == 'label_skin':
        filenameColorfile    = "./TFs/Labels_Skin_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_trans_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"        
    elif tgt == 'label_skin_phantom':
        filenameColorfile    = "./TFs/Labels_Skin_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_trans_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"        
    elif tgt == 'label_head':
        filenameColorfile    = "./TFs/Labels_Head_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_head_muscles':
        filenameColorfile    = "./TFs/Labels_Head_Muscles_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_head_muscles_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_lr':
        filenameColorfile    = "./TFs/Labels_HipThigh_LR_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"         
    elif tgt == 'label_vessel':
        filenameColorfile    = "./TFs/Labels_VesselNerve_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_artery':
        filenameColorfile    = "./TFs/Labels_VesselNerve_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"        
    elif tgt == 'label_muscle_vessel':
        filenameColorfile    = "./TFs/Labels_VesselNerveMuscles_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        # filenameOpacityfile  = "./TFs/VesselNerveMuscles_Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"
    elif tgt == 'label_muscle_artery':
        filenameColorfile    = "./TFs/Labels_VesselNerveMuscles_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_ArteryMuscles_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"                               
    elif tgt == 'label_muscle_vein':
        filenameColorfile    = "./TFs/Labels_VesselNerveMuscles_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_VeinMuscles_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"                               
    elif tgt == 'label_muscle_nerve':
        filenameColorfile    = "./TFs/Labels_VesselNerveMuscles_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_NerveMuscles_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"                               
    elif tgt == 'label_vessel_artery':
        filenameColorfile    = "./TFs/Labels_VesselNerve_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_ArteryVessels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"                               
    elif tgt == 'label_vessel_vein':
        filenameColorfile    = "./TFs/Labels_VesselNerve_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_VeinVessels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"                               
    elif tgt == 'label_sub_vessel_muscle':
        filenameColorfile    = "./TFs/Labels_SubVesselMuscles_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_SubVesselMuscles_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"    
    elif tgt == 'label_vessel_nerve':
        filenameColorfile    = "./TFs/Labels_VesselNerve_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_NerveVessels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"    
    elif tgt == 'label_error_map':
        filenameColorfile    = "./TFs/Labels_ErrorMap_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_ErrorMap_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"    
    elif tgt == 'label_4_muscles':
        filenameColorfile    = "./TFs/Labels_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_Revised4Muscles_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"    
    elif tgt == 'label_4_muscles_GMinOnly':
        filenameColorfile    = "./TFs/Labels_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_Revised4Muscles_GMinOnly_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'label_foot_compartment':
        filenameColorfile    = "./TFs/Foot/Labels_foot_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'label_foot_indiv':
        filenameColorfile    = "./TFs/Foot/Labels_foot_indiv_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'label_foot_sub':
        filenameColorfile    = "./TFs/Foot/Labels_foot_color_tf.txt"
        filenameOpacityfile  = "./TFs/Foot/Labels_foot_bone_only_opacity_sub_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'label_foot_bone':
        filenameColorfile    = "./TFs/Foot/Labels_foot_bone_color_tf.txt"
        filenameOpacityfile  = "./TFs/Foot/Labels_foot_bone_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'label_foot_muscle':
        filenameColorfile    = "./TFs/Foot/Labels_foot_color_tf.txt"
        filenameOpacityfile  = "./TFs/Foot/Labels_foot_muscle_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'label_hip_bones_lr':
        filenameColorfile    = "./TFs/Labels_HipBonesLR_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'label_hip_muscles_bones_lr':
        filenameColorfile    = "./TFs/Labels_HipThigh_LR_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt"  
    elif tgt == 'uncertainty':
        filenameColorfile    = "./TFs/Uncertainty_color_tf.txt"
        filenameOpacityfile  = "./TFs/Uncertainty_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Uncertainty_gopacity_tf.txt"  
    elif tgt == 'distance':
        filenameColorfile    = "./TFs/Distance_color_tf.txt"
        filenameOpacityfile  = "./TFs/Distance_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Distance_gopacity_tf.txt"  
    elif tgt == 'label_muscle_bone_vessel':
        filenameColorfile    = "./TFs/Labels_MuscleBoneVessel_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_opacity_MuscleBoneVessel_color_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt" 
    elif tgt == 'single_label':
        filenameColorfile    = "./TFs/Labels_single_label_color_tf.txt"
        filenameOpacityfile  = "./TFs/Labels_single_label_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_single_label_gopacity_tf.txt" 
    elif tgt == 'muscle_ratio':
        filenameColorfile    = "./TFs/Muscle_ratio_red2_color_tf.txt"
        filenameOpacityfile  = "./TFs/Muscle_ratio_opacity_tf.txt"
        filenameGOpacityfile = "./TFs/Muscle_ratio_gopacity_tf.txt" 
    elif tgt == 'hitachi_all':
        filenameColorfile    = "./TFs/Labels_hitachi_all_color_tf.txt"
        filenameOpacityfile  = "./TFs/Opacity_hitachi_all_color_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt" 
    elif tgt == 'hitachi_bone':
        filenameColorfile    = "./TFs/Labels_hitachi_bone_color_tf.txt"
        filenameOpacityfile  = "./TFs/Opacity_hitachi_bone_color_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt" 
    elif tgt == 'hitachi_muscle':
        filenameColorfile    = "./TFs/Labels_hitachi_muscle_color_tf.txt"
        filenameOpacityfile  = "./TFs/Opacity_hitachi_muscle_color_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt" 
    elif tgt == 'hitachi_chest_bone':
        filenameColorfile    = "./TFs/Labels_hitachi_chest_bone_color_tf.txt"
        filenameOpacityfile  = "./TFs/Opacity_hitachi_chest_bone_color_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt" 
    elif tgt == 'hitachi_chest_muscle':
        filenameColorfile    = "./TFs/Labels_hitachi_chest_muscle_color_tf.txt"
        filenameOpacityfile  = "./TFs/Opacity_hitachi_chest_muscle_color_tf.txt"
        filenameGOpacityfile = "./TFs/Labels_gopacity_tf.txt" 
    else:
        raise NotImplementedError
    color = get_color_from_csv(filenameColorfile)
    opacity = get_scalar_from_csv(filenameOpacityfile)
    gopacity = get_scalar_from_csv(filenameGOpacityfile)
    
    return color, opacity,gopacity

def get_meta_reader(fname):
    """
    Read a meta image
    """        
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader

def get_nifty_reader(fname):
    """
    Read a nifty image
    """        
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader    

def get_nrrd_reader(fname):
    """
    Read a NRRD image
    """        
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader    

def get_poly_reader(fname):
    '''
    Read polygon data saved in VTK format
    '''
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader

def get_plane_clip(dat, angle=0):
    _origin = l2n(dat.GetOrigin())
    _spacing = l2n(dat.GetSpacing())
    _dims = l2n(dat.GetDimensions())

    planeClip = vtk.vtkPlane()
    if angle in [0, 360] :
        planeClip.SetNormal(0.0, -1.0, 0.0)
        _center = n2l(_origin+_spacing*(_dims/2.0))
    elif angle == 90:
        planeClip.SetNormal(-1.0, 0.0, 0.0)
        _center = n2l(_origin+_spacing*(_dims/2.0))
    elif angle == 180:
        planeClip.SetNormal(0.0, 1.0, 0.0)
        _center = n2l(_origin+_spacing*(_dims/3.0))
    elif angle == 270:
        planeClip.SetNormal(1.0, 0.0, 0.0)
        _center = n2l(_origin+_spacing*(_dims/3.0))
    else:
        raise NotImplementedError        
    planeClip.SetOrigin(_center)

    return planeClip

def cast_reader(reader):
    """
    Casts an image reader to unsigned short (label) output.
    """        
    castFilter = vtk.vtkImageCast()
    castFilter.SetInputConnection(reader.GetOutputPort())
    castFilter.SetOutputScalarTypeToUnsignedShort()
    castFilter.Update()
    return castFilter

def get_rendered_window(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns window.
    """    
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(True)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.SetMultiSamples(0)
    renderWindow.SetAlphaBitPlanes(0)
    renderWindow.Render()
    return renderWindow

def get_window_to_image_filter(renderWindow):
    """
    Takes rendered window and returns window-to-image filter.
    """        
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()
    return windowToImageFilter

def write_image(fname,f):
    """
    Takes filter and writes the content to a file.
    """    
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname)
    writer.SetInputData(f.GetOutput())
    writer.Write()

def set_camera(camera,c, cd=2150, angle=[0.0, 0.0],vu=[0,0,1]):
    """
    Takes renderer and outputs a camera.
    """ 
    if len(angle)==1:
        angle = [0.0] +  angle
    camera.SetViewUp(*vu)
    camera.SetPosition(c[0], c[1]+cd, c[2])
    camera.SetClippingRange(100,5000)
    camera.SetFocalPoint(c[0], c[1], c[2])
    # camera.ComputeViewPlaneNormal()
    # camera.SetFocalPoint(c[0], c[1], c[2])
                #camera.Azimuth(90.0)
    camera.Elevation(angle[0])
    camera.Azimuth(angle[1])
    return camera

def set_light(light, camera, c, angle):
    camera_distance = camera.GetDistance()
    camera_fp = camera.GetFocalPoint()
    light.SetIntensity(3)
    if np.abs(angle[1]) == 180:
        light.SetPosition(c[0]+camera_distance, c[1]+2*camera_distance, c[2]+camera_distance)
    elif np.abs(angle[1]) == 90:
        light.SetPosition(c[0]+camera_distance, c[1]+2*camera_distance, c[2]+camera_distance)
    elif np.abs(angle[1]) == 270:
        light.SetPosition(c[0]-camera_distance, c[1]+2*camera_distance, c[2]+camera_distance)    
    elif angle[1] == 0 or angle[1] == 360:
        light.SetPosition(c[0]+camera_distance, c[1]-2*camera_distance, c[2]+camera_distance)    
    light.SetFocalPoint(*camera_fp)
    light.SetDiffuseColor(1, 1, 1)  
    light.PositionalOn()
    # light.SetLightTypeToCameraLight()
    return light

def createDummyRenderer(c ,cd, angle, vu, col=[0.0, 0.0, 0.0]):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(col)
    cam = renderer.MakeCamera()
    cam = set_camera(cam,c, cd, angle, vu)
    # light_2 = set_light(light, c, angle+90)
    renderer.SetActiveCamera(cam)
    renderer.ResetCamera()
    # renderer.LightFollowCameraOff()
    # renderer.SetUseDepthPeeling(True)
    # renderer.SetMaximumNumberOfPeels(100)
    # renderer.SetOcclusionRatio(0.0)
    # renderer.ResetCameraClippingRange()
    # renderer.PreserveDepthBufferOn()
    # renderer.AddLight(light_2)
    return renderer

def add_light_to_label_renderer(renderer, c, angle):
    light = vtk.vtkLight()
    cam = renderer.GetActiveCamera()
    out_light = set_light(light,cam, c, angle)
    renderer.AddLight(out_light)
    return renderer

def get_color_from_csv(fpath):
    fid = open(fpath, "r")
    reader = csv.reader(fid)
    dictRGB = {}
    for line in reader:
        dictRGB[float(line[0])] = [float(line[1]),
                                   float(line[2]),
                                   float(line[3])]
    fid.close()
    return dictRGB

def get_color_trans_func(color_dict):
    """
    Takes a color dictionary and return a transfer function.
    """       
    assert isinstance(color_dict, dict), 'Color is not a dictionary!'
    funcColor = vtk.vtkColorTransferFunction()
    funcColor.RemoveAllPoints()
    # funcColor.AllowDuplicateScalarsOn()
    for idx in color_dict.keys():
        funcColor.AddRGBPoint(idx, 
                            color_dict[idx][0],
                            color_dict[idx][1],
                            color_dict[idx][2], 0.5, 1.0)
    # funcColor.SetColorSpace(2)
    funcColor.Build()                            
    return funcColor

def get_scalar_from_csv(fpath):
    fid = open(fpath, "r")
    reader = csv.reader(fid)
    dictScalar = {}
    for line in reader:
        dictScalar[float(line[0])] = float(line[1])
    fid.close()
    return dictScalar

def get_scalar_funct(idxs, scalar=0.2):
    """
    Takes color dictionary an returns a scalar function, e.g. opacity.
    """   
    funcScalar = vtk.vtkPiecewiseFunction()
    # funcScalar.AllowDuplicateScalarsOn()
    if isinstance(scalar, (float, int)):
        for idx in idxs:
            funcScalar.AddPoint(idx, scalar if idx!=0 else 0.0)
    elif len(scalar) == len(idxs):
        for idx, val in zip(idxs, scalar):
            funcScalar.AddPoint(idx, val, 0.5, 1.0)
    return funcScalar

def get_label_center(img, label = 1):
    img_arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    dims = img.GetDimensions()[::-1]
    es   = img.GetSpacing()[::-1]
    offset = img.GetOrigin()
    lbl_img = img_arr.reshape(dims, order="C")== label
    _x, _y, _z = np.argwhere(lbl_img==1).sum(0)/lbl_img.sum()
    label_c = [_x+30, _y, _z]#[::-1]-75
    # label_c = [256, 256, _z]#[::-1]
    c = np.multiply(es, label_c)[::-1]
    if offset:
        c += offset
    return c

def get_volume_property(color_trans_func,
                        opac_trans_func,
                        grad_opac_func,
                        shade = None,
                        interp = 'linear',
                        ambient = 0.1,
                        diffuse=1.0,
                        specular=0.0,
                        spec_power=1.0):
    """
    Takes transfer functions and returns volume proprty object.
    """                         
    propVolume = vtk.vtkVolumeProperty()
    # propVolume.ShadeOff()
    propVolume.SetColor(color_trans_func)
    propVolume.SetScalarOpacity(opac_trans_func)
    propVolume.SetGradientOpacity(grad_opac_func)
    # propVolume.SetIndependentComponents(True)
    if interp == 'linear':
        propVolume.SetInterpolationTypeToLinear()
    if interp == 'nearest':
        propVolume.SetInterpolationTypeToNearest()    
    propVolume.ShadeOn()
    propVolume.SetAmbient(ambient) #0.4
    propVolume.SetDiffuse(diffuse)
    propVolume.SetSpecular(specular) #0.17
    propVolume.SetSpecularPower(spec_power)
    propVolume.SetScalarOpacityUnitDistance(.5)
    return propVolume

def get_volume_mapper(dat, gpu=True):
    # funcRayCast = vtk.vtkFixedPointVolumeRayCastMapper()
    # funcRayCast.SetCompositeMethodToClassifyFirst()
    if gpu:
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        # mapper.SetBlendModeToComposite()
        # mapper.SetSampleDistance(0.01)
        # mapper.SetUseJittering(0)
        
        # mapper.SetAutoAdjustSampleDistances(0)
    else:
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()    
    # mapper = vtk.vtkSmartVolumeMapper()
    # mapper.SetRequestedRenderMode(0)
    # mapper.SetVolumeRayCastFunction(funcRayCast)
    mapper.SetInputData(dat)
    return mapper
def get_volume_mapper_from_connection(src, gpu=True):
    # funcRayCast = vtk.vtkFixedPointVolumeRayCastMapper()
    # funcRayCast.SetCompositeMethodToClassifyFirst()
    if gpu:
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        # mapper.SetBlendModeToComposite()
        # mapper.SetSampleDistance(0.01)
        # mapper.SetUseJittering(0)
        
        # mapper.SetAutoAdjustSampleDistances(0)
    else:
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()    
    # mapper = vtk.vtkSmartVolumeMapper()
    # mapper.SetRequestedRenderMode(0)
    # mapper.SetVolumeRayCastFunction(funcRayCast)
    mapper.SetInputConnection(src.GetOutputPort())
    mapper.Update()
    return mapper

def get_volume_actor(mapper, prop):
    actor = vtk.vtkVolume()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    actor.Update()
    return actor

def set_image_threshold(img_reader, val_in, val_out=0):
    # threshold
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(img_reader)
    threshold.ThresholdBetween(val_in,val_in)
    # threshold.ThresholdByLower(val_in)
    # threshold.ReplaceOutOn()
    threshold.SetOutValue(val_out)
    threshold.Update()
    return threshold.GetOutput()

def get_surface_actor(imgreader, label=1):
    # Extractor = vtk.vtkDiscreteMarchingCubes()
    Extractor = vtk.vtkDiscreteFlyingEdges3D()
    Extractor.SetInputData(imgreader)
    Extractor.SetValue(0,label)
    Extractor.Update()

    surface = get_poly_smoother(Extractor)
    surface = get_poly_decimator(surface)
    surface = get_surface_normals(surface)
    surface_actor = get_poly_actor(surface)
    return surface_actor

def get_surface_normals(polyreader, angle=60.0):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polyreader.GetOutput())
    return normals

def get_poly_decimator(polyreader, ratio=0.99, largest=False):
    redpoly = vtk.vtkQuadricDecimation()
    redpoly.SetInputData(polyreader.GetOutput())
    redpoly.SetTargetReduction(ratio)
    redpoly.VolumePreservationOn()
    redpoly.Update()
    if largest:
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(redpoly.GetOutput())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        return conn
    return redpoly

def get_poly_smoother(polyreader, iter_val = 15,feature_angle=120,pass_band=0.001):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polyreader.GetOutput())
    smoother.SetNumberOfIterations(iter_val)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother

def get_poly_renderer(bg = None, wsize=(500, 500), off_screen=True, gradient_bg = False):
        '''
        Function to create a renderer window.
        Inputs:
            bg   : background color (default white)
            wsize:  window size
        Outputs:
            renderer  : renderer object
            renWindow : renderer window
        '''                              
        renderer = vtk.vtkRenderer()
        renWindow = vtk.vtkRenderWindow()
        renWindow.AddRenderer(renderer)
        renWindow.SetSize(*wsize)
        if bg is not None:
            renderer.SetBackground(*bg)
        if off_screen:
            renWindow.OffScreenRenderingOn()
        if gradient_bg:
            renderer.GradientBackgroundOn()
            renderer.SetBackground((1.0, 1.0, 1.0))
            renderer.SetBackground2((0.0,0.0,0.0))
        renWindow.SetMultiSamples(0)
        renWindow.SetAlphaBitPlanes(0)
        
        return renderer, renWindow

def get_distance_filter(polyreader1, polyreader2, signedDistance=True):
    distFilter = vtk.vtkDistancePolyDataFilter()
    distFilter.SetInputData(0, polyreader1.GetOutput())
    distFilter.SetInputData(1, polyreader2.GetOutput())
    distFilter.SetSignedDistance(signedDistance)
    distFilter.Update()
    return distFilter

def get_poly_actor(polyreader, edge_visible=False,col=(0.35, 0.3, 0.20), alpha=0.6):
    plyMapper = vtk.vtkPolyDataMapper()
    plyMapper.SetInputConnection(polyreader.GetOutputPort())
    # plyMapper.ScalarVisibilityOff()

    plyActor = vtk.vtkActor()
    plyActor.SetMapper(plyMapper)
    plyActor.GetProperty().SetOpacity(alpha)
    plyActor.GetProperty().SetColor(col)
    # plyActor.GetProperty().LightingOff()
    if edge_visible:
        plyActor.GetProperty().EdgeVisibilityOn()
        plyActor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
    return plyActor

def get_sphere_actor(center, radius=2, col=[1.0, 0.0,0.0],alpha=0.9,
                    ambient=0.3, diffuse=0.6, specular=0.05, specular_power=1):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], 
                     center[1], 
                     center[2])
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(100)
    sphere.SetThetaResolution(100)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*col) # Yellow
    actor.GetProperty().SetOpacity(alpha) 
    actor.GetProperty().SetAmbient(ambient)
    actor.GetProperty().SetDiffuse(diffuse)
    actor.GetProperty().SetSpecular(specular)
    actor.GetProperty().SetSpecularPower(specular_power)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().ShadingOn()
    actor.GetProperty().BackfaceCullingOn()
    mapper.Update()
    # actor.Update()
    return actor

def get_distance_actor(dist_filter, minmax = None):
    plyMapper = vtk.vtkPolyDataMapper()
    plyMapper.SetInputConnection(dist_filter.GetOutputPort())
    print('Port initialized')
    if minmax is None:
        plyMapper.SetScalarRange(dist_filter.GetOutput().GetPointData().GetScalars().GetRange()[0],
                                dist_filter.GetOutput().GetPointData().GetScalars().GetRange()[1])
    else:
        plyMapper.SetScalarRange(minmax[0], minmax[1])        
    print('Range set')
    plyActor = vtk.vtkActor()
    plyActor.SetMapper(plyMapper)
    return plyActor, plyMapper


def get_scalar_bar_actor(mapper, _side):
    legendActor = vtk.vtkScalarBarActor()
    legendActor.SetLookupTable(mapper.GetLookupTable())
    legendActor.SetTitle('Distance [mm]')
    legendActor.UnconstrainedFontSizeOn()
    legendActor.SetNumberOfLabels(7)
    legendActor.SetPosition(0.8, 0.1)
    if _side == 'right':
        legendActor.SetPosition(0.1, 0.2)
    elif _side == 'left':
        legendActor.SetPosition(0.8, 0.2)
    # legendActor.SetPosition(0.8, 0.1 if _side == 'right' else 0.1, 0.2)
    legendActor.SetWidth(0.15)
    legendActor.SetHeight(0.7)
    return legendActor
    
def get_renderer_window(ren, _x=800, _y=900):
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(_x, _y)
    renWin.SetOffScreenRendering(True)
    # renWin.SetMultiSamples(0)
    # renWin.SetAlphaBitPlanes(0)
    # renWin.SetMultiSamples(1)
    # renWin.SetAlphaBitPlanes(False)
    # #renWin.Render()
    # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    # renderWindowInteractor.SetRenderWindow(renWin)
    # # Begin Interaction
    # renderWindowInteractor.Initialize()
    # renWin.Render()
    # renWin.SetWindowName("XYZ Data Viewer")
    # renderWindowInteractor.Start() 
    return renWin

def run_renderer_window(renWin):
    ''' Takes a rendered window object and shows on screen'''
    #renWin.Render()
    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renWin)
    # Begin Interaction
    renWinInteractor.Initialize()
    renWinInteractor.Start() 

def get_reader(img_path):
    _ext = os.path.basename(img_path).split('.')[-1]
    if _ext in ['mha', 'mhd']:
        reader = get_meta_reader(img_path)
        intercept =  None
    elif _ext in ['gz']:
        reader = get_nifty_reader(img_path)
        intercept = reader.GetRescaleIntercept()
    elif _ext in ['nrrd']:
        reader = get_nrrd_reader(img_path)
        intercept =  None
    else:
        raise NotImplementedError   
    return reader, intercept 

def get_bb_actor(bb_path, color = [0, 0, 0]):
    with open(bb_path, 'r') as f:
        all_lines_variable = f.readlines()
        points = []
        for line in all_lines_variable:
            _split_line = line.split('|')
            if _split_line[0]=='point':
                point = [abs(float(coord)) for coord in _split_line[1:4]]
                points.append(point)
            if len(points)==2:
                break
    cubeSource = vtk.vtkCubeSource()
    _xyz = points[1]
    cubeSource.SetBounds(((points[0][0]-_xyz[0],points[0][0]+_xyz[0],
                           points[0][1]-_xyz[1],points[0][1]+_xyz[1],
                           points[0][2]-_xyz[2],points[0][2]+_xyz[2])))

    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputConnection(cubeSource.GetOutputPort())

    cubeActor = vtk.vtkActor()
    # cubeActor.GetProperty().SetDiffuseColor((0,0,1))
    cubeActor.GetProperty().SetRepresentationToWireframe()
    cubeActor.GetProperty().SetColor(*color)
    cubeActor.GetProperty().SetVertexColor(*color)
    cubeActor.GetProperty().SetLineWidth(4)
    cubeActor.GetProperty().RenderLinesAsTubesOn()
    # cubeActor.GetProperty().SetOpacity(0.5)
    # cubeActor.GetProperty().EdgeVisibilityOn()
    cubeActor.SetMapper(cubeMapper)
    return cubeActor

def set_renderer_camera( ren, az=0, el=0, roll=0, 
                    pos=(500, 0, 0), fc=None):
        '''
        Function to set the camera in a renderer object
        Inputs: 
            ren: window renderer object
            az: azimuth angle (degrees)
            el: elevation angle (degrees)
            pos: position of the camera
            fc : focal point (point where the camera looks)
        Outputs:
            ren   : updated renderer object
        '''                          
        # camera =vtk.vtkCamera()
        camera =ren.GetActiveCamera()
        camera.SetClippingRange(100, 4000)
        camera.SetPosition(pos)
        camera.SetViewUp(0,0, 1)
        
        if fc is not None:
            camera.SetFocalPoint(fc)
        camera.Azimuth(az) #Around X
        camera.Elevation(el) #Around Y
        camera.Roll(roll) 
        return ren

def get_image_data_with_intercept(reader, intercept):
    imgMath = vtk.vtkImageMathematics()
    imgMath.SetInputConnection(reader.GetOutputPort())
    imgMath.SetOperationToAddConstant()
    imgMath.SetConstantC(intercept)
    imgMath.Update()
    imageData = imgMath.GetOutput()
    return imageData

def save_snapshot(renWin, fname):
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

def get_text_actor(txt, font_size=12, loc=[20,30], col=[0,0,0]):
    '''
    Function to create VTK text actor (also creates mapper).
    Inputs: 
        txt: text
        loc: location in the window "(0,0) is bottom left"
        col: color in RGB format (0-1, 0-1, 0-1)
    Outputs:
            : VTK text actor
    '''                                 
    txtActor = vtk.vtkTextActor()
    txtActor.SetInput(txt)
    txtprop=txtActor.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(font_size) 
    txtprop.BoldOn()
    txtprop.ShadowOn()
    txtprop.SetFontSize(font_size) 
    txtprop.SetColor(*col)
    txtActor.SetDisplayPosition(*loc)
    return txtActor