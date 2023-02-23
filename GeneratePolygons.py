import os
import vtk
import numpy as np
from utils.VTK import VTK_Helper
import logging
import json
from joblib import delayed, Parallel
import pandas as pd
import tqdm
from utils import mhd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')
# fix_label = np.zeros([100, 100, 100], dtype=np.uint8)
# fix_label[10:15, 20:25, 20:80] = 1
# fix_label[17:19, 20:25, 20:80] = 2
# mhd.write('D:/temp/distance_test/fix_distance_label.mhd', image=fix_label, header={'CompressedData': True,
#                                                                          'ElementSpacing': [1, 1, 1],
#                                                                          'Offset': [0, 0, 0]})
# _ALPHA = 1
_EL = 0
_AZ = 180
_COL = [0.8, 0.8, 0.8]  # Polygon color in visualizations
_WIN_SIZE = [500, 500]
# _LABELS_F = 'D:/temp/distance_test/fix_distance.json'
_LABELS_F = './structures_json/osaka_vessels_acetabulum.json'
# _LABELS_F = './structures_json/osaka_vessels_acetabulum_lr.json'
# _LABELS_F = './structures_json/osaka_vessels_lr.json'
with open(_LABELS_F, 'r') as f:
    _LABELS = json.load(f)
_OUT_EXT = '.vtk'
_IN_EXT = 'fix_distance_label.mhd'
_ROOT = 'D:/temp/distance_test'
_TGT = 'D:/temp/distance_test/Polygons'
os.makedirs(_TGT, exist_ok=True)
_N_JOBS = 5
_TGT_NUM_CELLS = 10000
_TGT_FIG = _TGT + '/PolyFigures'
os.makedirs(_TGT_FIG, exist_ok=True)
# _FILE_LIST = 'E:/Projects/Lower Limbs/NII/Pelvis/PPTs/20210927_Filtered/20211004_FilteredIDs_withImplants.csv'
#_FILE_LIST = 'X:/mazen/Segmentation/Codes/datalists/osaka/vessels/caseid_list_vessels_20.txt'


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


class PolygonGenerator(object):
    '''
    PolygonGenerator class for generating polygon models from a given
    a label image.
    '''

    # Constructor
    def __init__(self, out_dir='./Results/PolygonModels'):
        # Output directory
        self.out_dir = out_dir
        if (not os.path.isdir(self.out_dir)):
            os.makedirs(self.out_dir, exist_ok=True)
        logging.info('Polygon Generator initialized.')
        logging.info('Output directory set to: {0}'.format(self.out_dir))

    @staticmethod
    def PolyWriter(polyreader, fpath):
        '''
        Function to save polygon data.
        Inputs: 
            polyreader: vtkPolyDataReader object
            fpath     : output file path
        Outputs:
            None
        '''
        print(fpath)
        ext = fpath.split(".")[-1]
        print(ext)
        if ext == "vtk":
            writer = vtk.vtkPolyDataWriter()
        elif ext == "vtp":
            writer = vtk.vtkPolyDataWriter()
        elif ext == "fib":
            writer = vtk.vtkPolyDataWriter()
        elif ext == "ply":
            writer = vtk.vtkPLYWriter()
        elif ext == "stl":
            writer = vtk.vtkSTLWriter()
        elif ext == "xml":
            writer = vtk.vtkXMLPolyDataWriter()
        elif ext == "obj":
             raise("mni obj or Wavefront obj ?")

        writer.SetFileName(fpath)
        writer.SetFileTypeToBinary()
        writer.SetInputData(polyreader.GetOutput())
        writer.Update()
        writer.Write()

    @staticmethod
    def append_encoded(inlist, new_el, encoder='ascii'):
        if isinstance(new_el, str):
            inlist.append(new_el.encode(encoder))  # .encode(encoder)
        elif isinstance(new_el, np.ndarray):
            line = ' '.join([str(el) for el in new_el]) + ' '
            inlist.append(line.encode(encoder))  # .encode(encoder)

    @classmethod
    def LegacyPolyWriter(self, vertices, faces, fpath):
        '''
        Function to save polygon data in legacy VTK format.
        Inputs: 
            vertices: numpy array including vertices (x,y,z)
            faces   : numpy array inluding faces (3, p1_index, p2_index, p3_index)
            fpath     : output file path
        Outputs:
            None
        '''
        ext = fpath.split(".")[-1]
        if ext == "vtk":
            vtk_file = []
            self.append_encoded(vtk_file, "# vtk DataFile Version 3.0")
            self.append_encoded(vtk_file, "vtk output by custom writer")
            self.append_encoded(vtk_file, "ASCII")
            self.append_encoded(vtk_file, "DATASET POLYDATA")
            num_vertices = len(vertices)
            num_faces = int(len(faces) / 4)
            self.append_encoded(vtk_file, "POINTS %s float" % (num_vertices))
            for i in range(0, num_vertices, 3):
                self.append_encoded(vtk_file, np.concatenate(vertices[i:i + 3]))
            self.append_encoded(vtk_file, "\r\nPOLYGONS %s %s" %
                                (num_faces, len(faces)))
            for i in range(0, len(faces), 4):
                self.append_encoded(vtk_file, faces[i:i + 4])
            self.append_encoded(vtk_file, "\r\n".encode('ascii'))
            self.append_encoded(vtk_file, "\r\nCELL_DATA %s" % (num_faces))
            self.append_encoded(vtk_file, "POINT_DATA %s" % (num_vertices))
            with open(fpath, mode='wb') as f:
                f.writelines(["%s\r\n".encode('ascii') % line for line in vtk_file])
        else:
            raise "extesnion %s not supported. Use PolyWriter" % (ext)

    def _PolyDataDecimator(self, polyreader, ratio=0.5, largest=False):
        '''
        Function to decimate (undersample) polygon data.
        Inputs: 
            polyreader: vtkPolyDataReader object
            ratio     : ratio of faces to remove
            largest   : flag to keep only largest polygon
        Outputs:
            redpoly   : reduced polygon data object
        '''
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

    def _PolySmootherSinc(self, polyreader, iter=15, feature_angle=120, pass_band=0.001):
        '''
        Function to decimate (undersample) polygon data.
        Inputs: 
            polyreader: vtkPolyDataReader object
            iter      : number of iterations windowed of applying the sinc 
                        function interpolation kernel to all vertices
            feature_angle: angle to determine the sharp edges
            pass band: pass band settings of the interpolation filter
        Outputs:
            smoother   : filtered polygon data object
        '''
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(polyreader.GetOutput())
        smoother.SetNumberOfIterations(iter)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetPassBand(pass_band)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        return smoother

    @staticmethod
    def _MetaImageReader(img_path):
        '''
        Function to read the label image (in mhd or mha formats).
        Inputs: 
            img_path   : path indicating the meta image location
        Outputs:
            reader    : meta image reader object
        '''
        ext = os.path.splitext(img_path)[-1]
        assert ext in ['.mha', '.mhd'], 'Input is not a meta image!'
        reader = vtk.vtkMetaImageReader()
        reader.SetFileName(img_path)
        reader.Update()
        return reader

    def _GeneratePolyDatafromScalar(self, imgreader, label=1):
        '''
        Function to generate polygon data corresponding with 
        isovlaue in vtk image reader object.
        Inputs: 
            imreader : VTK meta image reader
            label    : label (isosurface) value
        Outputs:
            Extractor   : VTK polydata object
        '''
        # Extractor = vtk.vtkDiscreteMarchingCubes()
        Extractor = vtk.vtkDiscreteFlyingEdges3D()
        Extractor.SetInputData(imgreader)
        Extractor.SetValue(0, label)
        Extractor.Update()
        return Extractor

    def _PolyRenderer(self, bg=None, wsize=[500, 500]):
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
        renWindow.OffScreenRenderingOn()
        renWindow.SetMultiSamples(0)
        renWindow.SetAlphaBitPlanes(0)
        return renderer, renWindow

    def _SaveSnapshot(self, renWin, fname):
        '''
        Function to save a snapshot for the renderer window.
        Inputs:
            renWin: rendering window
            fname:  filename
        Outputs:
            None
        '''
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renWin)
        w2if.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(fname)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

    def _TextActor(self, txt, font_size=12, loc=[20, 30], col=[0, 0, 0]):
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
        txtprop = txtActor.GetTextProperty()
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(font_size)
        txtprop.SetColor(*col)
        txtActor.SetDisplayPosition(*loc)
        return txtActor

    def _PolyActor(self, polyreader, edge_visible=True,
                   col=(1.0, 0, 0), alpha=1):
        '''
        Function to create VTK actor (also creates mapper).
        Inputs: 
            polyreader: vtkPolyDataReader object
            edge_visible: flag for wireframe visualization
            col         : color in RGB format (0-1,0-1,0-1)
            alpha       : opacity ratio (0-1)      
        Outputs:
            plyActor   : VTK actor object
        '''
        plyMapper = vtk.vtkPolyDataMapper()
        plyMapper.SetInputConnection(polyreader.GetOutputPort())
        plyActor = vtk.vtkActor()
        plyActor.SetMapper(plyMapper)
        plyActor.GetProperty().SetOpacity(alpha)
        plyActor.GetProperty().SetColor(col)
        if edge_visible:
            plyActor.GetProperty().EdgeVisibilityOn()
        return plyActor

    def _SetCamera(self, ren, az=0, el=0,
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
        camera = ren.GetActiveCamera()
        camera.SetClippingRange(100, 4000)
        camera.SetPosition(pos)
        camera.SetViewUp(0, 0, 1)

        if fc is not None:
            camera.SetFocalPoint(fc)
        camera.Azimuth(az)  # Around X
        camera.Elevation(el)  # Around Y
        camera.Roll(0)
        return ren

    def _VisualizePolygon(self, polyreader,
                          txt, d=400):
        '''
        Function to visualize the generated polygon model.
        Inputs: 
            polyreader: vtkPolyDataReader object
            txt       : text to add to the figure
            d         : distance between camera and object 
                        (in y direction)
        Outputs:
            None
        '''
        ren, win = self._PolyRenderer(bg=(1, 1, 1), wsize=_WIN_SIZE)
        poly_actor = self._PolyActor(polyreader,
                                     edge_visible=True,
                                     col=_COL, alpha=_ALPHA)
        text_actor = self._TextActor(txt)
        ren.AddActor(poly_actor)
        ren.AddActor(text_actor)
        win.Render()
        c = polyreader.GetOutput().GetCenter()
        pos = [c[0], c[1] + d, c[2]]
        ren = self._SetCamera(ren, pos=pos, az=_AZ, el=_EL)
        snap_fname = os.path.join(_TGT_FIG, txt + '_surface.png')
        self._SaveSnapshot(win, snap_fname)
        del win

    @classmethod
    def _DecomposePolyData(self, surface):
        '''
        Function to decompose vtkPolyData object into vertices and faces
        Inputs:
            surface: vtkPolyData object
        Outputs:
            vs: vertices as numpy array
            fc: faces as numpy array    
        '''
        vertices = VTK_Helper.vtk_to_numpy(surface.GetOutput().GetPoints().GetData())
        faces = VTK_Helper.vtk_to_numpy(surface.GetOutput().GetPolys().GetData())
        return vertices, faces

    def GeneratePolygonFile(self,
                            in_label_path,
                            out_file_path,
                            tgt_num_cells=10000,
                            label_val=1,
                            out_ext='.vtk',
                            vis=True,
                            tag='',
                            legacy=True):
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
            reader = self._MetaImageReader(in_label_path)
            logging.info('Reader created')

            # Generate polydata
            surface = self._GeneratePolyDatafromScalar(reader.GetOutput(), label=label_val)
            logging.info('Surface (polygon) generated')
            num_cells = surface.GetOutput().GetNumberOfCells()
            _ratio = 1 - tgt_num_cells / num_cells
            logging.info('Current number of cells: %d' % (num_cells))
            logging.info('Reduction ratio: %0.4f' % (_ratio))

            # Smooth polydata
            surface = self._PolySmootherSinc(surface)
            logging.info('Surface (polygon) smoothed')

            # Decimate
            surface = self._PolyDataDecimator(surface, ratio=_ratio)
            logging.info('Surface (polygon) decimated')

            # Write
            # out_file_path = os.path.join(self.out_dir, 'test%s' % (out_ext))

            if legacy:
                vs, fs = self._DecomposePolyData(surface)
                self.LegacyPolyWriter(vs, fs, out_file_path)
            else:
                self.PolyWriter(surface, out_file_path)
            logging.info('Polygon written')

            # Visualize
            if vis:
                img_name = os.path.splitext(os.path.basename(in_label_path))[0]
                txt = '%s%s' % (img_name, tag)
                self._VisualizePolygon(surface, txt)
                logging.info('Polygon snapshot saved to %s' % (txt))

            logging.info('File %s done...' % (in_label_path))
        except:
            logging.info('File %s not generated...' % (in_label_path))


def process_image(in_img_path, pg, tgt='Pelvis'):
    _case = os.path.basename(os.path.dirname(in_img_path))
    # out_file_path = os.path.join(_TGT, '%s' % 
    #             _case + '_' + (os.path.basename(in_img_path).replace('.mhd','_%s%s' % (tgt,_OUT_EXT)))).replace(os.sep, '/')
    out_file_path = os.path.join(_TGT, '%s_%s%s' %
                                 (_case, tgt, _OUT_EXT))
    # out_file_path = os.path.join(_TGT, '%s_%s%s' % 
    #             (os.path.basename(os.path.dirname(in_img_path)), tgt,_OUT_EXT)).replace(os.sep, '/')
    if ~os.path.isfile(out_file_path):
        logging.info('Started processing %s' % (in_img_path))
        label_val = _LABELS[tgt]  # Pelvis
        logging.info('Processing %s (Label value: %s)' % (tgt, _LABELS[tgt]))
        pg.GeneratePolygonFile(in_img_path,
                               out_file_path,
                               tgt_num_cells=_TGT_NUM_CELLS,
                               label_val=label_val,
                               out_ext=_OUT_EXT)


if __name__ == "__main__":
    ####### TEST CODE #####
    # Build class
    # pg = PolygonGenerator(out_dir= 'tests_polygon_generator')
    test_file = 'D:/temp/distance_test/fix_distance_label.mhd'
    pg = PolygonGenerator(out_dir=_TGT)
    # label_files = read_datalist(_FILE_LIST, root=_ROOT, ext=_IN_EXT)
    # idxs = []
    # for idx, lbl_file in enumerate(tqdm.tqdm(label_files)):
    #     out_file_path = '%s/%s' % (_TGT, os.path.basename(lbl_file).replace('.mhd', _OUT_EXT))
    #     if not os.path.isfile(os.path.normpath(out_file_path)):
    #         # logging.info('File not found: %s' % (out_file_path))
    #         # logging.info('File not found: %s' % (os.path.normpath(out_file_path)))
    #         idxs.append(idx)
    # logging.info('%s indices found!' % (len(idxs)))
    # selected_label_files = [label_files[idx] for idx in idxs]
    # logging.info('%s files selected!' % (len(selected_label_files)))
    tgts = list(_LABELS.keys())[0:2]
    img = test_file
    for tgt in tgts:
        process_image(img, pg, tgt)
        # Parallel(n_jobs=_N_JOBS)(delayed(process_image)(img, pg, tgt))
    logging.info('All cases done.')
