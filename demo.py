# This code reads a .bin file, stores the data in a .csv file, and 
# processes (filtering, segmenting, & clustering) LiDAR point
# cloud data using "open3d" library in Python.
 



# import standard libraries
import pandas as pd

import math
import numpy as np
import time
import pymap3d
import glob
import pyransac3d
import open3d as o3d
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib import figure
import csv

def main():

    lidar_path = 'kitti_data/*.bin'

    file = glob.glob(lidar_path)
    sorted(file, key = lambda x: os.path.dirname(x))
    file = sorted(file, key = lambda x: float(x[17:-4]))
    print(file)
    cont = 0
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name = 'Lidar')

    while (len(file)-1 > cont):

        #Read point cloud
        #print("file")
        #print(file)
        # 
        num = np.fromfile(file[cont], dtype='float32', count=-1, sep='', offset=0)

        new = np.asarray(num).reshape(-1, 4)

        # Assigning data to different variables
        X = num[0::4]
        Y = num[1::4]
        Z = num[2::4]

        # Creating point cloud
        xyz = np.zeros((np.size(X), 3))
        xyz[:, 0] = X
        xyz[:, 1] = Y
        xyz[:, 2] = Z
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        #.voxel_down_sample(voxel_size=0.05)

        pcd_nplane = inrange_segment_plane(pcd = pcd)

        vis.add_geometry(pcd_nplane)
        ctr = vis.get_view_control()
        ctr.set_front([0,0,1])
        ctr.set_zoom(0.4)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.clear_geometries()
        cont += 1


    return None

def inrange_segment_plane (pcd):

    # Remove points out of range of 80 meters
    arr = np.asarray(pcd.points)
    r = np.sqrt(np.square(arr[:,0])+np.square(arr[:,1]))
    inrange = np.where(r < 80)
    pcd_inrange = pcd.select_by_index(inrange[0], invert=False)
    arr_inrange = np.asarray(pcd_inrange.points)

    # Segment plane with RANSAC algorithm
    #segmplane = dfpc0.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=120)
    plane1 = pyransac3d.Plane()
    best_eq, best_inliers = plane1.fit(arr_inrange, thresh=0.25, minPoints=1, maxIteration=100)

    #inliers_result = pd.DataFrame(segmplane[1])

    # Convert list to numpy array
    #inliers_result_arr = np.asarray(inliers_result)
    #inliers_result_arr.astype(np.uint32)
    
    # Remove plane
    pcd_out = pcd_inrange.select_by_index(best_inliers, invert=True)

    #pcd_plane = dfpc0.select_by_index(inliers_result_arr, invert=False)
    
    # Visualize planes
    #o3d.visualization.draw_geometries([pcd_out])
    #o3d.visualization.draw_geometries([pcd_plane])

    # Clustering
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd_out.cluster_dbscan(eps=0.6, min_points=25, print_progress=False))
        #eps: distance to neighbor in cluster
        #min_points: min number of points required to form a cluster
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_out.colors = o3d.utility.Vector3dVector(colors[:, :3])

    #visualization
    # o3d.visualization.draw_geometries([pcd_out])
    print("pcd_out")
    print(pcd_out)

    labelst = labels.transpose()
    print("labelst")
    print(labelst)

    return pcd_out

main()