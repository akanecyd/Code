import os
import glob
# from typing_extensions import OrderedDict
from vtk.util import numpy_support
import pandas as pd
import numpy as np
import vis_utils
import json
from joblib import Parallel, delayed
import tqdm

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

def dirname(x):
    return os.path.basename(os.path.dirname(x))
def filename(x):
    return os.path.basename(x)

def read_datalist(fpath, field='fileID'):
    datalist = []
    if fpath.endswith('.txt'):
        with open(fpath, 'r') as fr:
            datalist = fr.readlines()
        datalist = [dat.replace('\n', '') for dat in datalist]
    elif fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        print('Dataframe: ', df)
        datalist = df[field].values.tolist()
    return datalist

def create_dir(pth):
    os.makedirs(pth, exist_ok=True)

def save_config_to_json(conf_dict, pth):
    with open(pth, 'w') as fw:
        json.dump(conf_dict,fw,sort_keys=True, indent=4)

def visualize_single_structure_image(image, main_config, struct_config):
    #Create out dir
    _OUT_DIR = main_config['out_dir']
    if os.path.exists(image):
        imagePath = image
        _CASE = os.path.basename(imagePath).split('_')[0]
    else:
        imagePath = os.path.join(struct_config['root'],'%s/%s' % (image,struct_config['ext'])) #image, 
        _CASE = image
    color_dict, opacity_dict, gopacity_dict = vis_utils.get_color_configs(struct_config['tf'])
    #Create transfer functions
    color_trans_func = vis_utils.get_color_trans_func(color_dict)
    opacity_trans_func = vis_utils.get_scalar_funct(opacity_dict.keys(),
                                                    scalar=opacity_dict.values())
    gradient_opacity_func = vis_utils.get_scalar_funct(gopacity_dict.keys(),
                                                    scalar=gopacity_dict.values())

    #Create reader
    reader, intercept = vis_utils.get_reader(imagePath)

    #Get image data
    if intercept:
        if intercept != 0.0:
            imageData = vis_utils.get_image_data_with_intercept(reader, intercept)
    else:
        imageData = reader.GetOutput()

    # Set volume property
    prop = vis_utils.get_volume_property(color_trans_func,
                                        opacity_trans_func,
                                        gradient_opacity_func,
                                        interp=struct_config['interp'],
                                        **struct_config['cam_config'])
    # Render
    # Centering
    if struct_config['center_label'] is not None:
        c = vis_utils.get_label_center(imageData,label=struct_config['center_label'])
    else:
        c = imageData.GetCenter()
    if struct_config['center_y_offset'] is not None:
            c[2] = c[2] + struct_config['center_y_offset']

    # Threshold
    if struct_config['th'] is not None:
        imageData = vis_utils.set_image_threshold(imageData, struct_config['th'])

    # Create actors
    mapper = vis_utils.get_volume_mapper(imageData) 
    actor = vis_utils.get_volume_actor(mapper, prop)

    if struct_config['bb']['status']:
        cube_actors = []
        bb_list = glob.glob(os.path.join(struct_config['bb']['root'], 
                                            '*'+struct_config['bb']['ext']))
        for i,bb in enumerate(bb_list):
            cube_actor = vis_utils.get_bb_actor(bb, struct_config['bb']['colors'][i])
            cube_actors.append(cube_actor)

    for _dir, cam_config in main_config['angles'].items():
        cd = cam_config['cd'] 
        angle = cam_config['angle'] 
        renderer = vis_utils.createDummyRenderer(c, 
                                                 cd, 
                                                 angle, 
                                                 main_config['vu'],
                                                 col=main_config['Background'])
        renderer.AddActor(actor)  
        # Add light
        renderer = vis_utils.add_light_to_label_renderer(renderer,c,angle)

        # Add text
        if main_config['add_text']:
            _txt = '%s %s' % (_CASE,_dir)
            text_actor = vis_utils.get_text_actor(_txt,loc=[20, 
                                                            main_config['y']-20],
                                                        font_size=18) 
            renderer.AddActor(text_actor)

        if struct_config['bb']['status']:
            for cube_actor in cube_actors:
                renderer.AddActor(cube_actor)
        # Show the windows
        renWin= vis_utils.get_renderer_window(renderer,
                                              _x=main_config['x'], 
                                              _y=main_config['y']) 
        w2if = vis_utils.get_window_to_image_filter(renWin)

        #Save window content
        # out_image_path = os.path.join(_OUT_DIR, # _CASE + '_'+\
        #         os.path.basename(imagePath).replace('.mhd', '_%s_%s.png' % 
        #                                             (struct_config['tf'],_dir))) 
        out_image_path = os.path.join(_OUT_DIR, '%s_%s.png' % (_CASE,_dir))
        #         '%s_%s.png' % (image,_dir))         
        vis_utils.write_image(out_image_path, w2if)
        renderer.Clear()

def visualize_multiple_structures_image(image, main_config, structs_config):
    #Create out dir
    _OUT_DIR = main_config['out_dir']
    
    # Creating actors
    actors = []
    for i, struct in enumerate(structs_config.values()):
        if 'skin' in struct['ext']:
            # imagePath = os.path.join(struct['root'], image,'%s%s' % (image,struct['ext']))
            imagePath = os.path.join(struct['root'], image,struct['ext'])#
        else:
            # imagePath = os.path.join(struct['root'],image, '%s%s' % (image,struct['ext']))
            imagePath = os.path.join(struct['root'], image, struct['ext'])  #
        _CASE = image           
        print(imagePath)
        color_dict, opacity_dict, gopacity_dict = vis_utils.get_color_configs(struct['tf'])
        #Create transfer functions
        color_trans_func = vis_utils.get_color_trans_func(color_dict)
        opacity_trans_func = vis_utils.get_scalar_funct(opacity_dict.keys(),
                                                        scalar=opacity_dict.values())
        gradient_opacity_func = vis_utils.get_scalar_funct(gopacity_dict.keys(),
                                                        scalar=gopacity_dict.values())

        #Create reader
        reader, intercept = vis_utils.get_reader(imagePath)


        #Get image data
        if intercept:
            if intercept != 0.0:
                imageData = vis_utils.get_image_data_with_intercept(reader, intercept)
        else:
            imageData = reader.GetOutput()
        # imageData.SetOrigin(0,0,0)

        if struct['th'] is not None:
            imageData = vis_utils.set_image_threshold(imageData, struct['th'])

        # Set volume property
        prop = vis_utils.get_volume_property(color_trans_func,
                                            opacity_trans_func,
                                            gradient_opacity_func,
                                            interp=struct['interp'],
                                            **struct['cam_config'])
        # Render
        # Centering
        if struct['center_label'] is not None:
            c = vis_utils.get_label_center(imageData,label=struct['center_label'])
        else:
            c = imageData.GetCenter()
        if struct['center_y_offset'] is not None:
                c[2] = c[2] + struct['center_y_offset']        
        #Reset origin
        # imageData.SetOrigin((0,0,0))

        #Create mapper
        mapper = vis_utils.get_volume_mapper(imageData)
        # Link mapper to actor
        actor = vis_utils.get_volume_actor(mapper, prop)
        actors.append(actor)

    # Rendering
    for _dir, cam_config in main_config['angles'].items():
        cd = cam_config['cd'] 
        if cd=='adaptive':
            _,_, zdim = imageData.GetDimensions()
            _,_, zsp =  imageData.GetSpacing()
            cd = zsp*zdim*2.3
        angle = cam_config['angle'] 
        #Create renderer
        renderer = vis_utils.createDummyRenderer(c, 
                                                cd, 
                                                angle, 
                                                main_config['vu'],
                                                col=main_config['Background'])
        # Add label actors
        for actor in actors:
            renderer.AddActor(actor)
        
        # Add light
        renderer = vis_utils.add_light_to_label_renderer(renderer,c,angle)
        
        # Add text actor
        if main_config['add_text']:
            # _txt = dirname(imagePath) + filename(imagePath)+ '_' + _dir
            _txt = dirname(imagePath) + '_' + _dir
            text_actor = vis_utils.get_text_actor(_txt,loc=[20, 
                                                            main_config['y']-20]) 
            renderer.AddActor(text_actor)

        # Add bounding box actors
        if struct['bb']['status']:
            bb_list = glob.glob(os.path.join(struct['bb']['root'], 
                                            '*'+struct['bb']['ext']))
            for i,bb in enumerate(bb_list):
                cube_actor = vis_utils.get_bb_actor(bb, struct['bb']['colors'][i])
                renderer.AddActor(cube_actor)
        
        # Show the windows
        renWin= vis_utils.get_renderer_window(renderer,
                                            _x=main_config['x'], 
                                            _y=main_config['y']) 
        w2if = vis_utils.get_window_to_image_filter(renWin)

        #Save window content
        out_image_path = os.path.join(_OUT_DIR, # _CASE + '_' + 
                # _CASE + '_' + os.path.basename(imagePath).replace('.mhd', '_%s_%s.png' % (struct_config['tf'],_dir))) 
                _CASE.replace('.mhd','_') + '_' + _dir + '.png') 
        vis_utils.write_image(out_image_path, w2if)
        renderer.Clear()

def visualize_combined_label_image(image, main_config, structs_config):
    #Create out dir
    _OUT_DIR = main_config['out_dir']
    # if not os.path.exists(image):
    #     _CASE = image
    # else:
    _CASE = os.path.basename(os.path.dirname(image))

    # Creating actors
    actors = []
    for i, struct in enumerate(structs_config.values()):
        if not os.path.exists(image):
            if struct['root_type']=='case+ext':
                imagePath = os.path.join(struct['root'],image, '%s%s' % (image, struct['ext']))
            elif struct['root_type']=='case|ext':
                imagePath = os.path.join(struct['root'],image, '%s/%s' % (image, struct['ext']))
            elif struct['root_type']=='case|case+ext':
                imagePath = os.path.join(struct['root'], image,'%s/%s%s' % (image, image,struct['ext']))
        else:
            imagePath = image
        # print(imagePath)
        color_dict, opacity_dict, gopacity_dict = vis_utils.get_color_configs(struct['tf'])
        #Create transfer functions
        color_trans_func = vis_utils.get_color_trans_func(color_dict)
        opacity_trans_func = vis_utils.get_scalar_funct(opacity_dict.keys(),
                                                        scalar=opacity_dict.values())
        gradient_opacity_func = vis_utils.get_scalar_funct(gopacity_dict.keys(),
                                                        scalar=gopacity_dict.values())

        #Create reader
        reader, intercept = vis_utils.get_reader(imagePath)


        #Get image data
        if intercept:
            if intercept != 0.0:
                imageData = vis_utils.get_image_data_with_intercept(reader, intercept)
        else:
            imageData = reader.GetOutput()
        # imageData.SetOrigin(0,0,0)

        if struct['th'] is not None:
            imageData = vis_utils.set_image_threshold(imageData, struct['th'])

        # Set volume property
        prop = vis_utils.get_volume_property(color_trans_func,
                                            opacity_trans_func,
                                            gradient_opacity_func,
                                            interp=struct['interp'],
                                            **struct['cam_config'])
        # Render
        # Centering
        if struct['center_label'] is not None:
            c = vis_utils.get_label_center(imageData,label=struct['center_label'])
        else:
            c = imageData.GetCenter()
        if struct['center_y_offset'] is not None:
                c[2] = c[2] + struct['center_y_offset']        
        #Reset origin
        

        #Create mapper
        mapper = vis_utils.get_volume_mapper(imageData)
        # Link mapper to actor
        actor = vis_utils.get_volume_actor(mapper, prop)
        actors.append(actor)

    # Rendering
    for _dir, cam_config in main_config['angles'].items():
        cd = cam_config['cd'] 
        angle = cam_config['angle']  
        #Create renderer
        renderer = vis_utils.createDummyRenderer(c, 
                                                cd, 
                                                angle, 
                                                main_config['vu'],
                                                col=main_config['Background'])
        # Add label actors
        for actor in actors:
            renderer.AddActor(actor)
        
        # Add light
        renderer = vis_utils.add_light_to_label_renderer(renderer,c,angle)
        
        # Add text actor
        if main_config['add_text']:
            _txt = '%s %s' % (_CASE,_dir)
            text_actor = vis_utils.get_text_actor(_txt,loc=[20, 
                                                            main_config['y']-20],
                                                        font_size=18) 
            renderer.AddActor(text_actor)

        # Add bounding box actors
        if struct['bb']['status']:
            bb_list = glob.glob(os.path.join(struct['bb']['root'], 
                                            '*'+struct['bb']['ext']))
            for i,bb in enumerate(bb_list):
                cube_actor = vis_utils.get_bb_actor(bb, struct['bb']['colors'][i])
                renderer.AddActor(cube_actor)
        
        # Show the windows
        renWin= vis_utils.get_renderer_window(renderer,
                                            _x=main_config['x'], 
                                            _y=main_config['y']) 
        w2if = vis_utils.get_window_to_image_filter(renWin)

        #Save window content
        out_image_path = os.path.join(_OUT_DIR,'%s_%s_%s.png' % (_CASE,_dir, 'Left' if 'L_' in os.path.basename(imagePath) else 'Right'))
                # os.path.basename(imagePath).replace('.mhd', '_%s.png' % (_dir))) #+'-'+os.path.basename(imagePath) case_id + '_'+
        vis_utils.write_image(out_image_path, w2if)
        renderer.Clear()

def visualize_volumetric_image(image, main_config, struct_config):
    #Create out dir
    _OUT_DIR = main_config['out_dir']
    print(image)
    if os.path.exists(image):
        imagePath = image
        _CASE = os.path.basename(imagePath).split('_')[0]
        # _CASE = os.path.basename(os.path.dirname(imagePath))
    else:
        imagePath = os.path.join(struct_config['root'],image, '%s%s' % (image, struct_config['ext']))
        # imagePath = os.path.join(struct_config['root'], image)
        _CASE = image
    # print(imagePath)
    color_dict, opacity_dict, gopacity_dict = vis_utils.get_color_configs(struct_config['tf'])
    #Create transfer functions
    color_trans_func = vis_utils.get_color_trans_func(color_dict)
    opacity_trans_func = vis_utils.get_scalar_funct(opacity_dict.keys(),
                                                    scalar=opacity_dict.values())
    gradient_opacity_func = vis_utils.get_scalar_funct(gopacity_dict.keys(),
                                                    scalar=gopacity_dict.values())

    #Create reader
    reader, intercept = vis_utils.get_reader(imagePath)

    #Get image data
    if intercept:
        if intercept != 0.0:
            imageData = vis_utils.get_image_data_with_intercept(reader, intercept)
    else:
        imageData = reader.GetOutput()

    if struct_config['skin_mask']:
        skinPath = os.path.join(struct_config['skin_root'], image+ struct_config['skin_ext'])
        skin_reader = vis_utils.get_meta_reader(skinPath)
        skin_reader = vis_utils.cast_reader_tochar(skin_reader)
        skinData = skin_reader.GetOutput()
        imageData = vis_utils.get_masked_data(skinData, imageData)

    # Set volume property
    prop = vis_utils.get_volume_property(color_trans_func,
                                        opacity_trans_func,
                                        gradient_opacity_func,
                                        interp=struct_config['interp'],
                                        **struct_config['cam_config'])
    # Render
    # Centering
    if struct_config['center_label']:
        c = vis_utils.get_label_center(imageData,label=struct_config['center_label'])
    else:
        c = imageData.GetCenter()

    # Renderers
    for _dir, cam_config in main_config['angles'].items():
        cd = cam_config['cd'] 
        if cd=='adaptive':
            _,_, zdim = imageData.GetDimensions()
            _,_, zsp =  imageData.GetSpacing()
            cd = zsp*zdim*2.3
        angle = cam_config['angle'] 
        mapper = vis_utils.get_volume_mapper(imageData) 
        actor = vis_utils.get_volume_actor(mapper, prop)
 # 
        # if struct_config['clipping']:
        #     plane_clip = vis_utils.get_plane_clip(imageData, angle=angle)
        #     mapper.AddClippingPlane(plane_clip)
        renderer = vis_utils.createDummyRenderer(c, 
                                                 cd, 
                                                 angle, 
                                                 main_config['vu'],
                                                 col=main_config['Background'])
        renderer.AddActor(actor) 

        if main_config['add_text']:
            _txt = image + '_' + _dir  #case_id + '_'+  dirname(imagePath) +
            text_actor = vis_utils.get_text_actor(_txt,loc=[20, 
                                                            main_config['y']-20])
            renderer.AddActor(text_actor)

        if struct_config['bb']['status']:
            bb_list = glob.glob(os.path.join(struct_config['bb']['root'], 
                                             '*'+struct_config['bb']['ext']))
            for i,bb in enumerate(bb_list):
                cube_actor = vis_utils.get_bb_actor(bb, struct_config['bb']['colors'][i])
                renderer.AddActor(cube_actor)
        # Show the windows
        renWin= vis_utils.get_renderer_window(renderer,
                                              _x=main_config['x'], 
                                              _y=main_config['y']) 
        w2if = vis_utils.get_window_to_image_filter(renWin)
        import vtk
        # scalarBar = vtk.vtkScalarBarActor()
        #
        # # Set the lookup table to use with the scalar bar
        # scalarBar.SetLookupTable(mapper.GetColorTransferFunction())
        # scalarBar.SetTitle("Scalar Value")
        # scalarBar = vis_utils.get_scalar_bar_actor(mapper=actor)
        # # Add the scalar bar to the renderer
        # renderer.AddActor(scalarBar)
        #
        # import vtk
        # legendActor = vtk.vtkScalarBarActor()
        # legendActor.SetLookupTable(mapper.GetLookupTable())
        # legendActor.SetTitle('Uncertainty')
        # legendActor.UnconstrainedFontSizeOn()
        # legendActor.SetNumberOfLabels(7)
        # renderer.AddActor(legendActor)
        #Save window content
        out_image_path = os.path.join(_OUT_DIR, # _CASE + '_' + 
                # _CASE + '_' + os.path.basename(imagePath).replace('.mhd', '_%s_%s.png' % (struct_config['tf'],_dir))) 
                _CASE.replace('.mhd','_').replace('/','_')+ '_' + _dir + '.png') 
        vis_utils.write_image(out_image_path, w2if)
        renderer.Clear()

if __name__ == '__main__':
    # from vis_configs.vis_configs_combined_muscle_label_intensity import configs # Modify configurations here
    # from vis_configs.vis_configs_muscle_registration import configs # Modify configurations here
    # from vis_configs.vis_configs_head_tsurumi import configs # Modify configurations here
    # from vis_configs.vis_configs_head_kyoto import configs # Modify configurations here
    # from vis_configs.vis_configs_skin_phantom import configs # Modify configurations here
    # from vis_configs.vis_configs_msk_mr import configs # Modify configurations here
    # from vis_configs.vis_configs_distance_maps import configs # Modify configurations here
    # from vis_configs.vis_configs_vessels import configs # Modify configurations here
    # from vis_configs.vis_configs_vessels_osaka import configs # Modify configurations here
    # from vis_configs.vis_configs_foot import configs # Modify configurations here
    from vis_configs_uncertainty import configs # Modify configurations here
    # from vis_configs.vis_configs_muscles_mri import configs # Modify configurations here
    # from vis_configs.vis_configs_CT_with_implants import configs # Modify configurations here
    # from vis_configs.vis_configs_muscles_stryker import configs # Modify configurations here
    # from vis_configs.vis_configs_muscles_hitachi import configs # Modify configurations here
    # from vis_configs.vis_configs_muscles import configs # Modify configurations here
    # from vis_configs.vis_configs_muscles_osaka_lr_skin import configs # Modify configurations here
    # from vis_configs.vis_configs_bones_lr import configs # Modify configurations here
    # from vis_configs.vis_configs_muscle_bone_vessel import configs # Modify configurations here
    # from vis_configs.vis_configs_femur import configs # Modify configurations here
    # from vis_configs.vis_configs_skin import configs # Modify configurations here
    config = configs()
    _OUT_DIR = config.main_config['out_dir']
    create_dir(_OUT_DIR)
    save_config_to_json(config.main_config, os.path.join(_OUT_DIR, 'vis_main_config.json'))
    if hasattr(config, 'labels'):
        save_config_to_json(config.labels, os.path.join(_OUT_DIR, 'vis_label_config.json'))
        structs = list(config.labels.keys())
        if len(structs)==1:
            label_config = config.labels[structs[0]]
            if config.main_config['case_list']:
                img_list = read_datalist(config.main_config['case_list'])
            else:
                img_list = glob.glob(os.path.join(label_config['root'],
                                                  '*'+label_config['ext'] ))
            print(*img_list, sep='\n')
            pbar = tqdm.tqdm(img_list)
            Parallel(n_jobs = config.main_config['n_jobs'])(delayed(visualize_single_structure_image)
                                                           (img, config.main_config,label_config) 
                                                           for img in pbar)
            # for img in pbar:
            #     visualize_single_structure_image(img, config.main_config,label_config)
        else:  
            img_list = read_datalist(config.main_config['case_list'], field='ID')
            # img_list = ['O174']
            print(*img_list, sep='\n')
            pbar = tqdm.tqdm(img_list)            
            # Parallel(n_jobs = config.main_config['n_jobs'])(delayed(visualize_multiple_structures_image)
            #                                                        (img, config.main_config,config.labels) 
            #                                                        for img in pbar)
            for img in pbar:
                # structs_config = config.labels
                visualize_multiple_structures_image(img, config.main_config,config.labels)
    if hasattr(config, 'images'):
        save_config_to_json(config.images, os.path.join(_OUT_DIR, 'vis_image_config.json'))        
        structs = list(config.images.keys())
        for struct in structs:
            print('Visualizing %s' % (struct))
            image_config = config.images[struct]
            print(image_config)
            if config.main_config['case_list']:
                img_list = read_datalist(config.main_config['case_list'])
            else:
                img_list = glob.glob(os.path.join(image_config['root'],'*' + image_config['ext'] )) #'*', 'image',
            # img_list = [x for x in img_list if not os.path.exists(os.path.join(_OUT_DIR, x + '_interpolated_ct_muscle_bone_implant_AP.png'))]
            print(*img_list, sep='\n')
            # img_list = ['O501']
            pbar = tqdm.tqdm(img_list)
            # Parallel(n_jobs = config.main_config['n_jobs'])(delayed(visualize_volumetric_image)
            #                                                        (img,
            #                                                         config.main_config,
            #                                                         image_config) for img in pbar)
            for img in pbar:
                visualize_volumetric_image(img,config.main_config, image_config)
    
    if hasattr(config, 'combined'): 
        save_config_to_json(config.combined, os.path.join(_OUT_DIR, 'vis_label_image_combined_config.json'))        
        image_config = config.combined['muscle']
        if config.main_config['case_list']:
            img_list = read_datalist(config.main_config['case_list'])
        else:
            img_list = glob.glob(os.path.join(image_config['root'],'k*', '*' + image_config['ext'] )) #'*', 'image',
            img_list = [x for x in img_list if 'label' not in x]

        # img_list = read_datalist(config.main_config['case_list'], field='KID')
        print(*img_list, sep='\n')
        pbar = tqdm.tqdm(img_list)
        Parallel(n_jobs = config.main_config['n_jobs'])(delayed(visualize_combined_label_image)
                                                               (img,config.main_config, config.combined) for img in pbar)
        # for img in pbar:
        #     visualize_combined_label_image(img,config.main_config, config.combined)
    
