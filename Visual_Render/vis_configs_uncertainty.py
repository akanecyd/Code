from collections import OrderedDict
class configs(object):
    def __init__(self, out_dir = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/Uncert_Visualizations'):
        self.main_config = {
            'x': 300,
            'y': 230,
            'n_jobs': 4,
            'case_list': '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt',
            # '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt',
            # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt',
            'vu': [0, 0, 1],
            '_ROI_ROOT': None,  # 'Z:/mazen/LowerLimbs/HipSegmetnation2/Data/2.0.2/ROIs'
            'angles': {
                'AP': {'angle': [0, 180], 'cd': 500},
                'PA': {'angle': [180, -180], 'cd': 500},
                'RL': {'angle': [0, 90], 'cd': 500},
                'LR': {'angle': [0, 270], 'cd': 500},
            },
            'clipping': None,
            'out_dir': out_dir,
            'Background': (1.0, 1.0, 1.0),
            'add_text': True
        }

        self.images = OrderedDict()
        self.images['uncertainty'] = {
            'root' :'//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000' ,
            'ext': '-vessels_uncert.mhd',
            'tf' : 'uncertainty', #,'label' 'label_foot' 'ct_bone', 'ct_bone', 'ct_muscle_bone' 'label_vessel', 
            'interp': 'linear',
            'cam_config': {
                        'ambient': 0.5,
                        'diffuse': 0.5,
                        'specular': 0.2,
                        'spec_power': 10.0
                        },
            'center_label': None, #Label for centering, integer or None
            'center_y_offset': None,
            'flip': False,
            'th' : 1,
            'skin_mask': False,
            'skin_root':None,
            'skin_ext':None,
            'bb' : {'status': False,
                    'root': None,
                    'ext' : '.acsv',
                    'colors' : './TFs/BoundingBox_color.txt'
                    }
        }
