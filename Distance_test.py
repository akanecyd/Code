import os
from pickle import TRUE
import vtk
import numpy as np
from utils import vis_utils
from utils.VTK import VTK_Helper
import logging
import json
import GeneratePolygons
from joblib import delayed, Parallel
import pandas as pd
import tqdm
from utils import mhd
fix_label = np.zeros([100, 100, 100], dtype=np.uint8)
fix_label[10:15, 20:25, 20:80] = 1
fix_label[17:20, 20:25, 20:80] = 2
test_file = 'D:/temp/distance_test/fix_distance_label.mhd'
mhd.write(test_file, image=fix_label, header={'CompressedData':False,
                                              'ElementSpacing': [1, 1, 1],
                                              'Offset': [0, 0, 0]})
_ALPHA = 1
_EL = 0
_AZ = 180
_COL = [0.8, 0.8, 0.8]  # Polygon color in visualizations
_WIN_SIZE = [500, 500]
_LABELS_F = 'D:/temp/distance_test/fix_distance.json'
with open(_LABELS_F, 'r') as f:
    _LABELS = json.load(f)
_TGT = 'D:/temp/distance_test'
os.makedirs(_TGT, exist_ok=True)
pg = GeneratePolygons.PolygonGenerator(out_dir=_TGT)
tgts = list(_LABELS.keys())[0:2]
img = test_file
for tgt in tgts:
    GeneratePolygons.process_image(img, pg, tgt)
        # Parallel(n_jobs=_N_JOBS)(delayed(process_image)(img, pg, tgt))
logging.info('MHD to Polygon done')
# _LABELS_F = './structures_json/osaka_vessels_acetabulum.json'
# _LABELS_F = './structures_json/osaka_vessels_acetabulum_lr.json'
# _LABELS_F = './structures_json/osaka_vessels_lr.json'
_OUT_EXT = '.vtk'
_IN_EXT = 'fix_distance_label.mhd'
_ROOT = 'D:/temp/distance_test'

_POLY_ROOT = _TGT + '/Polygons'
os.makedirs(_TGT, exist_ok=True)
_N_JOBS = 5
_TGT_NUM_CELLS = 10000
_FIG_TGT = _POLY_ROOT + '/PolyFigures'
os.makedirs(_FIG_TGT, exist_ok=True)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')

# _SRC = './samples/surface_distance_samples/sacrum.vtk'
# _TGT = './samples/surface_distance_samples/pelvis.vtk'

_SRC = os.path.join(_POLY_ROOT, 'distance_test_fix.vtk' )

_TGT = os.path.join(_POLY_ROOT, 'distance_test_moving.vtk')

_DIST_MINMAX = (0, 5)
_SUB_ALPHAS = [0.3,
               0.5,
               0.5]


def read_datalist(fpath, root=None, ext=None, field='fileID'):
    datalist = []
    if fpath.endswith('.txt'):
        datalist = np.genfromtxt(fpath, dtype=str)
    elif fpath.endswith('.csv'):
        df = pd.read_csv(fpath, header=0)
        # print('Dataframe: ', df)
        datalist = df[field].values.tolist()
    if root:
        datalist = ['%s/%s' % (root, item) for item in datalist]
    if ext:
        datalist = ['%s/%s' % (item, ext) for item in datalist]
    return datalist


def VisualizeSurfaceDistance(
        src_poly_path,
        tgt_poly_path):
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
        # Read label image
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
        logging.info('Actors loaded')
        # renderer.AddActor(src_actor)
        # renderer.AddActor(tgt_actor)
        # c = src_actor.GetOutput().GetCenter()
        #pos = [100, 0, 0]
        logging.info('Camera position set')

        logging.info('Renderer set')
        # Show
        renderer, renWindow = vis_utils.get_poly_renderer(bg=(1.0, 1.0, 1.0),
                                                          off_screen=True,
                                                          gradient_bg=TRUE)
        # logging.info('Renderer and window loaded')
        _tmp_reader = vis_utils.get_poly_reader(_TGT)
        c = _tmp_reader.GetOutput().GetCenter()
        pos = [c[0],
               c[1] + 100,
               c[2]]
        renderer = vis_utils.set_renderer_camera(renderer,
                                                 pos=pos,
                                                 fc=c,
                                                 el=0,
                                                 # az=-120,
                                                 az=50 ,
                                                 roll=0)
        renWindow.Render()
        logging.info('Started rendering')
        # distance filter both from rim and vessel sides
        dfilter_fix = vis_utils.get_distance_filter(reader_src,
                                                       reader_tgt,
                                                       signedDistance=False)
        dfilter_moving = vis_utils.get_distance_filter(reader_tgt,
                                                    reader_src,
                                                    signedDistance=False)
        # get distance map from filter
        distance_map_fix = dfilter_fix.GetOutput()
        distance_map_moving = dfilter_moving.GetOutput()
        # get min distance and min distance point coordinate
        distance_fix_Array = distance_map_fix.GetPointData().GetArray('Distance')
        distance_fix_array_npy = VTK_Helper.vtk_to_numpy(distance_fix_Array)
        points_fix_array_npy = VTK_Helper.vtk_to_numpy(distance_map_fix.GetPoints().GetData())
        logging.info('Distance fixed array: %s', distance_fix_array_npy)
        min_dist_fix = np.min(distance_fix_array_npy)
        min_dist_fix_idx = np.argmin(distance_fix_array_npy)
        logging.info('Min fixed point: %s', min_dist_fix_idx)
        logging.info('Min fixed point coords: %s', points_fix_array_npy[min_dist_fix_idx])
        logging.info('Min fixed Distance: %0.3f mm' % (min_dist_fix))

        # distanceRimArray = distanceRimMap.GetPointData().GetArray('Distance')
        # distance_rim_array_npy = VTK_Helper.vtk_to_numpy(distanceRimArray)
        # points_rim_array_npy = VTK_Helper.vtk_to_numpy(distanceRimMap.GetPoints().GetData())
        # logging.info('Distance rim array: %s', distance_rim_array_npy)
        # min_dist_rim = np.min(distance_rim_array_npy)
        # min_dist_rim_idx = np.argmin(distance_rim_array_npy)
        # logging.info('Min rim point: %s', min_dist_rim_idx)
        # logging.info('Min rim point coords: %s', points_rim_array_npy[min_dist_rim_idx])
        # logging.info('Min Rim Distance: %0.3f mm' % (min_dist_rim))
        # add min distance text
        fig_txt = 'Min distance: %0.5f mm' % (min_dist_fix)
        txt_actor = vis_utils.get_text_actor(fig_txt, font_size=24)
        # add color bar
        dist_actor, dist_mapper = vis_utils.get_distance_actor(dfilter_fix,
                                                               minmax=_DIST_MINMAX)  #
        moving_dist_actor, _ = vis_utils.get_distance_actor(dfilter_moving,
                                                         minmax=_DIST_MINMAX)
        logging.info('Distance actor initialized')
        # scale bar
        bar_actor = vis_utils.get_scalar_bar_actor(mapper=dist_mapper, _side=None)
        logging.info('Bar actor initialized')
        # add min distance point
        point_fix_actor = vis_utils.get_sphere_actor(points_fix_array_npy[min_dist_fix_idx], radius=0.5)
        # point_rim_actor = vis_utils.get_sphere_actor(points_rim_array_npy[min_dist_rim_idx], col=[1.0, 1.0, 0.0])
        logging.info('Point actor initialized')
        # Assign actor to the renderer.
        # renderer.AddActor(src_actor)

        renderer.AddActor(tgt_actor)
        renderer.AddActor(dist_actor)
        # renderer.AddActor(rim_dist_actor)
        renderer.AddActor(point_fix_actor)
        # renderer.AddActor(point_rim_actor)
        renderer.AddActor(bar_actor)
        renderer.AddActor(txt_actor)



        # Create a renderwindowinteractor.
        # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        # renderWindowInteractor.SetRenderWindow(renWindow)
        # renderWindowInteractor.Initialize()
        # renWindow.Render()
        # renderWindowInteractor.Start()
        fname = os.path.join(_FIG_TGT, os.path.basename(_SRC).replace('.vtk', 'fixed_distance.png'))
        vis_utils.save_snapshot(renWindow, fname)
        del renWindow
        return min_dist_fix


    except Exception as exc:
        logging.info('%s' % exc)
        logging.info('Files \n%s \n%s\nnot generated...' % (src_poly_path, tgt_poly_path))


if __name__ == '__main__':
    # _SRC = os.path.join(_TGT, 'distance_test_fix.vtk' )
    # # rim_side
    # _TGT = os.path.join(_TGT, 'distance_test_moving.vtk')
    min_dist = VisualizeSurfaceDistance(_SRC, _TGT)

    # _cases = ['k1873']
    # for _organ1, _organ2, _organ3 in zip(('artery', 'vein'), ('vein', 'artery'), ('pelvis')):
    #     _sides = ['left', 'right']
    #     for i, _side in enumerate(['right', 'left']):
    #         # _TGT_STRUCT = 'rim_%s' % _side
    #         _TGT_STRUCT = '%s_pelvis' % _side
    #         _COMP_STRUCT = '%s_%s' % (_organ1, _side)
    #         _SRC_STRUCT = '%s_%s' % (_organ2, _side)
    #         _opo_pelvis = '%s_pelvis' % _sides[i]
    #         # tag = '-vessels_label_lr'
    #         tag = ''
    #
    #         if 'vein' in _COMP_STRUCT:
    #             _SUB_COLORS = [
    #                 (0.0, 0.6, 0.8),  # vein
    #                 # (0.8, 0.45, 0.25), # artery
    #                 (0.9, 0.9, 0.9),  # _opo_pelvis
    #                 (0.9, 0.9, 0.9)
    #             ]
    #         elif 'artery' in _COMP_STRUCT:
    #             _SUB_COLORS = [
    #                 # (0.0, 0.6, 0.8), # vein
    #                 (0.8, 0.45, 0.25),  # artery
    #                 (0.9, 0.9, 0.9),
    #                 (0.9, 0.9, 0.9)
    #             ]
    #
    #         print(_cases)
    #         dists = dict()
    #         for _case in tqdm.tqdm(_cases):
    #             # _SRC = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _SRC_STRUCT))
    #             # _TGT = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))
    #
    #             # _SRC = os.path.join(_SEG_POLY_ROOT, '%s%s_%s_.vtk' % (_case,tag, _SRC_STRUCT))
    #             # vessel(vein/artery)_side
    #             _SRC = os.path.join(_GT_POLY_ROOT, '%s%s_%s.vtk' % (_case, tag, _SRC_STRUCT))
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
    #                                                 sub_tgt_poly_path=_SUB_ACTORS,
    #                                                 sub_tgt_poly_colors=_SUB_COLORS,
    #                                                 sub_tgt_poly_alphas=_SUB_ALPHAS,
    #                                                 tag=_side)
    #             dists[_case] = min_dist
    #
    #         print(dists)
    #         df = pd.DataFrame(list(dists.items()), columns=['ID', _SRC_STRUCT]).set_index('ID')
    #         print(df)
    #         df.to_csv(os.path.join(_FIG_TGT, '%s.csv' % _SRC_STRUCT))

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

