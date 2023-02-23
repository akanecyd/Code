import os.path

import vtk
import json
from jsonpath import jsonpath
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


# Read the points from the .json file
# reader = vtk.vtkJSONDataArrayReader()

points_path ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/acetablum_points'\
    # "//Salmon/User/mazen/Segmentation/Codes/results/Nara/hip_vessels/18_5fold/seperation_left_right/Polygons/rim_points"
    # '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/acetablum_points'
    # "//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/acetablum_points"
_CASE_LIST ='//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
    # '//Salmon/User/Chen/Vessel_data/Nara_enhance_plain/crop_cases_list.txt'
    # '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
with open(_CASE_LIST) as f:
    case_IDs = f.read().splitlines()

def remove_duplicates(x_coords, y_coords, z_coords):
    new_x = []
    new_y = []
    new_z = []
    for i in range(len(x_coords)):
        if (x_coords[i], y_coords[i], z_coords[i]) not in zip(new_x, new_y, new_z):
            new_x.append(x_coords[i])
            new_y.append(y_coords[i])
            new_z.append(z_coords[i])
    return new_x, new_y, new_z

def readPointsfromJson(points_path,vtp_filename,num_true_pts):
    with open(points_path, 'r') as f:
        points = json.load(f)
    control_points = jsonpath(points, "$..controlPoints")[0]
    X_points = []
    Y_points = []
    Z_points = []
    for i, point in enumerate(control_points):
        coord = point["position"]
        X_points.append(coord[0])
        Y_points.append(coord[1])
        Z_points.append(coord[2])
    # points = np.unique(list(zip( X_points, Y_points, Z_points)), axis=0)
    X_clean, Y_clean, Z_clean = remove_duplicates(X_points, Y_points, Z_points)
    tck, u = interpolate.splprep([np.array(X_clean), np.array(Y_clean), np.array(Z_clean)], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.plot(X_points, Y_points, Z_points, 'r*')
    # ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(x_fine, y_fine, z_fine, 'g')
    plt.title(os.path.basename(points_path))
    fig2.show()
    plt.show()
    Vertices = vtk.vtkCellArray()
    interpolatedPoints = vtk.vtkPoints()
    for i in range(0, num_true_pts):
        id = interpolatedPoints.InsertNextPoint(x_fine[i], y_fine[i], z_fine[i])
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(id)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(interpolatedPoints)
    polydata.SetVerts(Vertices)
    polydata.Modified()
    # if vtk.VTK_MAJOR_VERSION <= 5:
    #     polydata.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName( vtp_filename)
    writer.Write()


if __name__ == '__main__':
    num_true_pts = 200
    with open(_CASE_LIST) as f:
        case_IDs = f.read().splitlines()
    print ( case_IDs)

    # case_IDs=['N0024']
    for case_ID in case_IDs:
        print (case_ID)
        rim_right_path = os.path.join(points_path, case_ID+'_rim_right.mrk.json' )
        rim_right_vtp = os.path.join(points_path, case_ID + '_rim_right.vtp')
        readPointsfromJson(points_path=rim_right_path, vtp_filename=rim_right_vtp, num_true_pts=num_true_pts)
        rim_left_path = os.path.join(points_path, case_ID + '_rim_left.mrk.json')
        rim_left_vtp = os.path.join(points_path, case_ID + '_rim_left.vtp')
        readPointsfromJson(points_path=rim_left_path, vtp_filename=rim_left_vtp, num_true_pts=num_true_pts)


#     # points.InsertNextPoint(1, 2, 0)
#     # points.InsertNextPoint(2, 3, 0)
#     # points.InsertNextPoint(3, 2, 0)




# Interpolate the points using a cardinal spline
# spline = vtk.vtkCardinalSpline()
# for i,point in enumerate(control_points):
#     coord = point["position"]
#     spline.AddPoint(coord[0], coord[1], coord[2])
# Use the spline to interpolate a curve

# n_points = spline1.GetN    umberOfPoints()


# Populate the vtkPoints object with the interpolated points
# for i in range(0,num_true_pts):
#     interpolatedPoints.InsertNextPoint(x_fine[i], y_fine[i], z_fine[i])
# print(max(x_fine),min(x_fine))
# print(max(y_fine),min(y_fine))
# print(max(z_fine),min(z_fine))
#
# # Create a vtkPolyData object to hold the interpolated points
# polyData = vtk.vtkPolyData()
# polyData.SetPoints(interpolatedPoints)
#
# # Create a VTK polydata writer
# polyDataWriter = vtk.vtkPolyDataWriter()
# polyDataWriter.SetInputData(polyData)
# # Set the input data to be written
#
#
# # Set the file name
#
# polyDataWriter.SetFileName("//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/seperation_left_right/Polygons/acetablum_points/interpolated_points.vtk")
#
# polyDataWriter.Write()
# Write the data to file
# polyDataWriter.Write()
# # curve = spline.GetParametricCurve(points[0], points[-1], 100)
# # # Print the points in the interpolated curve
# # for i in range(curve.GetNumberOfPoints()):
# #     print(curve.GetPoint(i))
# #
# # # Get the curve as a polydata
# # curve = spline.GetOutput()
#
# Write the curve to a file