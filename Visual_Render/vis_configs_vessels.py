from collections import OrderedDict
class configs(object):
    def __init__(self, out_dir ='//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint/Visualizations'):
    # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/Visualizations'):
        self.main_config = {
            'x' : 300,
            'y' : 230,
            'n_jobs' : 4,
            'case_list' : '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt',
                # '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt',
                # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt',
            'vu' : [0,0,1],
            '_ROI_ROOT' : None, #'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/ROIs'
            'angles':{
                      'AP': {'angle': [0,180], 'cd': 500},
                      'PA': {'angle': [180,-180], 'cd': 500},
                      'RL': {'angle': [0,90], 'cd': 500},
                      'LR': {'angle': [0,270], 'cd': 500},
                      },                      
            'clipping' : None,
            'out_dir' : out_dir,
            'Background' : (1.0, 1.0, 1.0),
            'add_text': True
        }

        self.labels = OrderedDict()

        self.labels['skin'] = {
            'root' :' //Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint',
            'ext': 'crop_skin_label.mhd',
                # 'skin_label.mhd',
                # '-skin_label.mhd',
            'tf' : 'label_skin', #,'label' 'label_foot' 'ct_bone', 'ct_bone', 'ct_muscle_bone' 'label_vessel',
            'interp': 'nearest',
            'cam_config': {
                        'ambient': 0.5,
                        'diffuse': 0.4,
                        'specular': 0.0,
                        'spec_power': 10.0
                        },
            'center_label': None, #Label for centering, integer or None
            'center_y_offset': None,
            'flip': False,
            'th': 1,
            'skin_mask': False,
            'bb' : {'status': False,
                    'root': None,
                    'ext' : '.acsv',
                    'colors' : './TFs/Labels_Skin_color.txt'
                        # './TFs/BoundingBox_color.txt'
                    }
        }

        self.labels['vessels'] = {
            'root' : '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/Good alignment/crop_hip_joint',
            'ext': 'crop_vein_artery_bones_label.mhd',
                # 'vessels_pelvis_femur_label.mhd',
                # '-vessels_label.mhd',
            'tf' : 'label_vessel', #,'label' 'label_foot' 'ct_bone', 'ct_bone', 'ct_muscle_bone' 'label_vessel', 
            'interp': 'nearest',
            'cam_config': {
                        'ambient': 0.5,
                        'diffuse': 0.4,
                        'specular': 0.0,
                        'spec_power': 10.0
                        },
            'center_label': None, #Label for centering, integer or None
            'center_y_offset': None,
            'th': None,
            'flip': False,
            'skin_mask': False,
            'bb' : {'status': False,
                    'root': None,
                    'ext' : '.acsv',
                    'colors' : './TFs/Labels_bone_vessel_color_tf.txt'
                    }
        }

        # self.labels['bones'] = {
        #     'root' : '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000',
        #     'ext': '-vessels_label.mhd',
        #     'tf' : 'label_muscle_hip', #,'label' 'label_foot' 'ct_bone', 'ct_bone', 'ct_muscle_bone' 'label_vessel',
        #     'interp': 'nearest',
        #     'cam_config': {
        #                 'ambient': 0.5,
        #                 'diffuse': 0.4,
        #                 'specular': 0.0,
        #                 'spec_power': 10.0
        #                 },
        #     'center_label': None, #Label for centering, integer or None
        #     'center_y_offset': None,
        #     'flip': False,
        #     'skin_mask': False,
        #     'bb' : {'status': False,
        #             'root': None,
        #             'ext' : '.acsv',
        #             'colors' : './TFs/BoundingBox_color.txt'
        #             }
        # }
    