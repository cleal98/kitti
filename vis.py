# This code reads a .bin file, stores the data in a .csv file, and 
# processes (filtering, segmenting, & clustering) LiDAR point
# cloud data using "open3d" library in Python.
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
import open3d as o3d
import csv

# Reading data
inputPath = 'kitti_data/kitti-3.bin'


num = np.fromfile(inputPath, dtype='float32', count=-1, sep='', offset=0)

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
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(xyz)

o3d.visualization.draw_geometries([pc])

# # Downsampling
# downpcd = pc.voxel_down_sample(voxel_size=0.1)

# # Segmentation
# plane_model, inliers = downpcd.segment_plane(distance_threshold=1, ransac_n=3, num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# inlier_cloud = downpcd.select_by_index(inliers)
# outlier_cloud = downpcd.select_by_index(inliers, invert=True)

# # Clustering using DBSCAN
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(inlier_cloud.cluster_dbscan(eps=2, min_points=10, print_progress=True))
#     # esp: Distance to neighbours in a cluster
#     # min_points: Minimun number of points required to form a cluster
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} cluster`s")
# colors = plt.get_cmap("tab20")(labels/(max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# inlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

# # Visualization
# o3d.visualization.draw_geometries([inlier_cloud])