import os
from pickle import TRUE
import vtk
import numpy as np
from utils import vis_utils
from utils.VTK import VTK_Helper
import logging
import json
from joblib import delayed,Parallel
import pandas as pd
import tqdm
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


# _SRC = './samples/surface_distance_samples/sacrum.vtk'
# _TGT = './samples/surface_distance_samples/pelvis.vtk'

_CASE_LIST = '//Salmon/User/Chen/Vessel_data/revised_nerve/caseid_list_nerves.txt'
_GT_POLY_ROOT = '//Salmon/User/Chen/Vessel_data/revised_nerve/crop_hipjiont_75mm_ver3/Polygons'
_GT_Pelvis_ROOT = '//Salmon/User/mazen/Segmentation/Data/HipMusclesDataset/Polygons'
# _SEG_POLY_ROOT = 'X:/mazen/Segmentation/Data/HipMusclesDataset/Polygons_SingleSides'
_SEG_POLY_ROOT = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/vessels_20_5fold_wo_muscles/Polygons'
# _FIG_TGT = 'X:/mazen/Segmentation/Data/HipMusclesDataset/Polygons_SingleSides/DistanceFigures'
_FIG_TGT = '//Salmon/User/Chen/Result/ploygon_20_rim_to_vessels\GT_pelvis'
# _FIG_TGT = 'X:/mazen/Segmentation/Data/HipMusclesDataset/Polygons_SingleSides/Test'
os.makedirs(_FIG_TGT, exist_ok=True)

_DIST_MINMAX = (0, 20)
_SUB_ALPHAS  =[0.3,
                0.5,
                0.5]

def read_datalist(fpath, root = None, ext=None,field='fileID'):
    datalist = []
    if fpath.endswith('.txt'):
        datalist = np.genfromtxt(fpath, dtype=str)
    elif fpath.endswith('.csv'):
        df = pd.read_csv(fpath,header=0)
        # print('Dataframe: ', df)
        datalist = df[field].values.tolist()
    if root:
        datalist = ['%s/%s' % (root,item) for item in datalist]
    if ext:
        datalist = ['%s/%s' % (item,ext) for item in datalist]        
    return datalist

def VisualizeSurfaceDistance(
        src_poly_path,
        tgt_poly_path, 
        sub_tgt_poly_path=[],
        sub_tgt_poly_colors=[],
        sub_tgt_poly_alphas=[],
        tag = ''):
    '''
    Function to generate polygon data from label file.
    Inputs: 
        in_label_path: location of the input label file
        out_file_path: location of output polygon file
        tgt_num_cells: number of cells to be generated
        label_val    : target label number (integer between 1 and max(labels))
        out_ext      : extension of the polygon data file
        vis          : flag indicating whether to visualize the generated polygon
        tag          : tag to add to the snapshot file name
    Outputs:
        None
    ''' 
    try:
        #Read label image
        logging.info('Generation settings \n %s\n' % (locals()))
        reader_src = vis_utils.get_poly_reader(_SRC)
        reader_tgt = vis_utils.get_poly_reader(_TGT)
        


        logging.info('Surface (polygon) generated')

        # Surface reaser
        num_cells_src = reader_src.GetOutput().GetNumberOfCells()
        num_cells_tgt = reader_tgt.GetOutput().GetNumberOfCells()
        logging.info('Current number of cells inm source: %d' % (num_cells_src))
        logging.info('Current number of cells inm source: %d' % (num_cells_tgt))

        src_actor = vis_utils.get_poly_actor(
                    reader_src, 
                    edge_visible=True, 
                    col=(1.0, 0.0, 0.0), 
                    alpha=0.6
                    )
        tgt_actor = vis_utils.get_poly_actor(
                    reader_tgt, 
                    edge_visible=True, 
                    col=(0.9, 0.8, 0.55), 
                    alpha=0.9
                    )
        _sub_actors = []
        if _SUB_ACTORS is not None:
            for _sub, _col, _alpha in zip(sub_tgt_poly_path, 
                                          sub_tgt_poly_colors, 
                                          sub_tgt_poly_alphas):
                _tmp_reader = vis_utils.get_poly_reader(_sub)
                _sub_actors.append(vis_utils.get_poly_actor(
                    _tmp_reader, 
                    edge_visible=False, 
                    col=_col, 
                    alpha=_alpha
                    ))
        logging.info('Actors loaded')
        # renderer.AddActor(src_actor)
        # renderer.AddActor(tgt_actor)
        # c = src_actor.GetOutput().GetCenter()
        pos = [600, 0, 0]
        logging.info('Camera position set')
        
        logging.info('Renderer set')
        # Show
        renderer, renWindow = vis_utils.get_poly_renderer(bg = (1.0, 1.0, 1.0),
                                                          off_screen=True,
                                                          gradient_bg = TRUE)
        # logging.info('Renderer and window loaded')
        c = _tmp_reader.GetOutput().GetCenter()
        pos = [c[0],
               c[1]+650,
               c[2]]
        renderer =vis_utils.set_renderer_camera(renderer, 
                                                pos=pos,
                                                fc=c, 
                                                el=0,
                                                # az=-120,
                                                az=135 if _side=='right' else -135,
                                                roll=0)
        renWindow.Render()
        logging.info('Started rendering')
        # distance filter both from rim and vessel sides
        dfilter_vessel= vis_utils.get_distance_filter(reader_src, 
                                                      reader_tgt, 
                                                      signedDistance=False)
        dfilter_rim= vis_utils.get_distance_filter(reader_tgt, 
                                                   reader_src, 
                                                   signedDistance=False)
        # get distance map from filter
        distanceVesselMap = dfilter_vessel.GetOutput()
        distanceRimMap = dfilter_rim.GetOutput()
        # get min distance and min distance point coordinate
        distanceVesselArray = distanceVesselMap.GetPointData().GetArray('Distance')
        distance_vessel_array_npy = VTK_Helper.vtk_to_numpy(distanceVesselArray)
        points_vessel_array_npy = VTK_Helper.vtk_to_numpy(distanceVesselMap.GetPoints().GetData())
        logging.info('Distance vessel array: %s', distance_vessel_array_npy)
        min_dist_vessel = np.min(distance_vessel_array_npy)
        min_dist_vessel_idx = np.argmin(distance_vessel_array_npy)
        logging.info('Min vessel point: %s', min_dist_vessel_idx)
        logging.info('Min vessel point coords: %s', points_vessel_array_npy[min_dist_vessel_idx])
        logging.info('Min Vessel Distance: %0.3f mm' % (min_dist_vessel))

        distanceRimArray = distanceRimMap.GetPointData().GetArray('Distance')
        distance_rim_array_npy = VTK_Helper.vtk_to_numpy(distanceRimArray)
        points_rim_array_npy = VTK_Helper.vtk_to_numpy(distanceRimMap.GetPoints().GetData())
        logging.info('Distance rim array: %s', distance_rim_array_npy)
        min_dist_rim = np.min(distance_rim_array_npy)
        min_dist_rim_idx = np.argmin(distance_rim_array_npy)
        logging.info('Min rim point: %s', min_dist_rim_idx)
        logging.info('Min rim point coords: %s', points_rim_array_npy[min_dist_rim_idx])
        logging.info('Min Rim Distance: %0.3f mm' % (min_dist_rim))
        # add min distance text
        fig_txt = 'Min distance: %0.3f mm' % (min_dist_vessel)
        txt_actor = vis_utils.get_text_actor(fig_txt, font_size=24)
        # add color bar
        dist_actor, dist_mapper = vis_utils.get_distance_actor(dfilter_vessel,
                                                               minmax = _DIST_MINMAX)#
        rim_dist_actor, _ = vis_utils.get_distance_actor(dfilter_rim,
                                                               minmax = _DIST_MINMAX)
        logging.info('Distance actor initialized')
        # scale bar
        bar_actor = vis_utils.get_scalar_bar_actor(mapper=dist_mapper,_side=_side)
        logging.info('Bar actor initialized')
        # add min distance point
        point_vessel_actor = vis_utils.get_sphere_actor(points_vessel_array_npy[min_dist_vessel_idx])
        point_rim_actor = vis_utils.get_sphere_actor(points_rim_array_npy[min_dist_rim_idx], col=[1.0,1.0,0.0])
        logging.info('Point actor initialized')
        # Assign actor to the renderer.
        # renderer.AddActor(src_actor)

        # renderer.AddActor(tgt_actor)
        renderer.AddActor(dist_actor)
        renderer.AddActor(rim_dist_actor)
        renderer.AddActor(point_vessel_actor)
        renderer.AddActor(point_rim_actor)
        renderer.AddActor(bar_actor)
        renderer.AddActor(txt_actor)

        
        if _SUB_ACTORS is not None:
            for _rd in _sub_actors:
                renderer.AddActor(_rd)
        
        # Create a renderwindowinteractor.
        # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        # renderWindowInteractor.SetRenderWindow(renWindow)
        # renderWindowInteractor.Initialize()
        # renWindow.Render()
        # renderWindowInteractor.Start()
        fname = os.path.join(_FIG_TGT, os.path.basename(_SRC).replace('.vtk', '_' + tag +'_pelvis.png'))
        vis_utils.save_snapshot(renWindow, fname)
        del renWindow
        return min_dist_vessel
       

    except Exception as exc:
        logging.info('%s' % exc)
        logging.info('Files \n%s \n%s\nnot generated...' % (src_poly_path, tgt_poly_path))


if __name__ == '__main__':
    _cases = read_datalist(_CASE_LIST)
    # _cases = ['k1873']
    for _organ1, _organ2, _organ3 in zip(('artery', 'vein'), ('vein', 'artery'), ('pelvis')):
        _sides = ['left', 'right']
        for i, _side in enumerate(['right', 'left']):
            # _TGT_STRUCT = 'rim_%s' % _side
            _TGT_STRUCT = '%s_pelvis' % _side
            _COMP_STRUCT = '%s_%s' % (_organ1, _side)
            _SRC_STRUCT = '%s_%s' % (_organ2, _side)
            _opo_pelvis = '%s_pelvis' % _sides[i]
            # tag = '-vessels_label_lr'
            tag = ''

            if 'vein' in _COMP_STRUCT:
                _SUB_COLORS = [
                    (0.0, 0.6, 0.8),  # vein
                    # (0.8, 0.45, 0.25), # artery
                    (0.9, 0.9, 0.9),  # _opo_pelvis
                    (0.9, 0.9, 0.9)
                ]
            elif 'artery' in _COMP_STRUCT:
                _SUB_COLORS = [
                    # (0.0, 0.6, 0.8), # vein
                    (0.8, 0.45, 0.25),  # artery
                    (0.9, 0.9, 0.9),
                    (0.9, 0.9, 0.9)
                ]

            print(_cases)
            dists = dict()
            for _case in tqdm.tqdm(_cases):
                # _SRC = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _SRC_STRUCT))
                # _TGT = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))

                # _SRC = os.path.join(_SEG_POLY_ROOT, '%s%s_%s_.vtk' % (_case,tag, _SRC_STRUCT))
                # vessel(vein/artery)_side
                _SRC = os.path.join(_GT_POLY_ROOT, '%s%s_%s.vtk' % (_case, tag, _SRC_STRUCT))
                # rim_side
                _TGT = os.path.join(_GT_Pelvis_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))

                # 'X:/mazen/Segmentation/Data/HipMusclesDataset/Polygons/k1565_femur.vtk',
                _SUB_ACTORS = [os.path.join(_GT_POLY_ROOT, '%s_%s.vtk' % (_case, _COMP_STRUCT)),
                               os.path.join(_GT_Pelvis_ROOT, '%s_%s.vtk' % (_case, _opo_pelvis))]
                # os.path.join(_GT_POLY_ROOT, '%s_acetabulum.vtk' % (_case)),
                # os.path.join(_GT_Pelvis_ROOT, '%s_pelvis.vtk' % (_case))]

                min_dist = VisualizeSurfaceDistance(_SRC, _TGT,
                                                    sub_tgt_poly_path=_SUB_ACTORS,
                                                    sub_tgt_poly_colors=_SUB_COLORS,
                                                    sub_tgt_poly_alphas=_SUB_ALPHAS,
                                                    tag=_side)
                dists[_case] = min_dist

            print(dists)
            df = pd.DataFrame(list(dists.items()), columns=['ID', _SRC_STRUCT]).set_index('ID')
            print(df)
            df.to_csv(os.path.join(_FIG_TGT, '%s.csv' % _SRC_STRUCT))

    # _cases = read_datalist(_CASE_LIST)
    # # _cases = ['k1873']
    # for _organ1, _organ2 ,_organ3 in zip(('artery', 'vein'), ('vein', 'artery'), ('pelvis')):
    #     _sides = ['left','right']
    #     for i,_side in enumerate(['right', 'left']):
    #         #_TGT_STRUCT = 'rim_%s' % _side
    #         _TGT_STRUCT = '%s_pelvis' % _side
    #         _COMP_STRUCT = '%s_%s' % (_organ1, _side)
    #         _SRC_STRUCT = '%s_%s' % (_organ2, _side)
    #         _opo_pelvis = '%s_pelvis' % _sides[i]
    #         # tag = '-vessels_label_lr'
    #         tag = ''
    #
    #         if 'vein' in _COMP_STRUCT:
    #             _SUB_COLORS = [
    #                             (0.0, 0.6, 0.8), # vein
    #                         # (0.8, 0.45, 0.25), # artery
    #                         (0.9, 0.9, 0.9) ,#_opo_pelvis
    #                          (0.9, 0.9, 0.9)
    #                         ]
    #         elif 'artery' in _COMP_STRUCT:
    #             _SUB_COLORS = [
    #                             # (0.0, 0.6, 0.8), # vein
    #                         (0.8, 0.45, 0.25), # artery
    #                         (0.9, 0.9, 0.9) ,
    #                          (0.9, 0.9, 0.9)
    #                         ]
    #
    #
    #         print(_cases)
    #         dists = dict()
    #         for _case in tqdm.tqdm(_cases):
    #             # _SRC = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _SRC_STRUCT))
    #             # _TGT = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))
    #
    #             # _SRC = os.path.join(_SEG_POLY_ROOT, '%s%s_%s_.vtk' % (_case,tag, _SRC_STRUCT))
    #             # vessel(vein/artery)_side
    #             _SRC = os.path.join(_GT_POLY_ROOT, '%s%s_%s.vtk' % (_case,tag, _SRC_STRUCT))
    #             # rim_side
    #             _TGT = os.path.join(_GT_Pelvis_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))
    #
    #             # 'X:/mazen/Segmentation/Data/HipMusclesDataset/Polygons/k1565_femur.vtk',
    #             _SUB_ACTORS = [os.path.join(_GT_POLY_ROOT, '%s_%s.vtk' % (_case, _COMP_STRUCT)),
    #                            os.path.join(_GT_Pelvis_ROOT, '%s_%s.vtk' % (_case, _opo_pelvis))]
    #             # os.path.join(_GT_POLY_ROOT, '%s_acetabulum.vtk' % (_case)),
    #             # os.path.join(_GT_Pelvis_ROOT, '%s_pelvis.vtk' % (_case))]
    #
    #             min_dist = VisualizeSurfaceDistance(_SRC, _TGT,
    #                                             sub_tgt_poly_path=_SUB_ACTORS,
    #                                             sub_tgt_poly_colors=_SUB_COLORS,
    #                                             sub_tgt_poly_alphas=_SUB_ALPHAS,
    #                                             tag = _side)
    #             dists[_case] = min_dist
    #
    #         print(dists)
    #         df = pd.DataFrame(list(dists.items()), columns=['ID', _SRC_STRUCT]).set_index('ID')
    #         print(df)
    #         df.to_csv(os.path.join(_FIG_TGT, '%s.csv' % _SRC_STRUCT))

                