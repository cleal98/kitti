import open3d
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import sys
import sklearn.cluster
import hdbscan


def read_velodyne_bin(path):
    '''
    :param path:
    :retorno: matriz de homografía de la nube de puntos, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

path='kitti_data/kitti-5.bin'

# Se coloca la ruta de los datos
origindata=read_velodyne_bin(path)

pcd = open3d.geometry.PointCloud()

pcd.points = open3d.utility.Vector3dVector(origindata)
open3d.visualization.draw_geometries([pcd])

def PlaneLeastSquare(X:np.ndarray):
    #z=ax+by+c,return a,b,c
    A=X.copy()
    b=np.expand_dims(X[:,2],axis=1)
    A[:,2]=1
   
    #Se resuleve directamente por X = (AT * A) -1 * AT * b

    A_T = A.T
    A1 = np.dot(A_T,A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2,A_T)
    x= np.dot(A3, b)
    return x

def PlaneRANSAC(X:np.ndarray,tao:float,e=0.4,N_regular=100):
    #return plane ids
    s=X.shape[0]
   
    count=0
    p=0.99
    dic={}
   
    #Se determina el número de iteraciones
    if math.log(1-(1-e)**s)<sys.float_info.min:
        N=N_regular
    else:
        N=math.log(1-p)/math.log(1-(1-e)**s)
       
    #Se inician las iteraciones
    while count < N:
       
       
        ids=random.sample(range(0,s),3)
        Points=X[ids]
        p1,p2,p3=X[ids]
        #Se determina si es colineal
        L=p1-p2
        R=p2-p3
        if 0 in L or 0 in R:
            continue
        else:
            if L[0]/R[0]==L[1]/R[1]==L[2]/R[2]:
                continue
               
        #Se calculan los parámetros del plano
        a = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);    
        b = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);    
        c = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);    
        d = 0 - (a * p1[0] + b*p1[1] + c*p1[2]);
       
        dis=abs(a*X[:,0]+b*X[:,1]+c*X[:,2]+d)/(a**2+b**2+c**2)**0.5
       
        idset=[]
        for i ,d in enumerate(dis):
            if d <tao:
                idset.append(i)
       
       
        #Se utiliza la función definida anteriormente PLANELEASTSQUARE
        p=PlaneLeastSquare(X[idset])
        a,b,c,d=p[0],p[1],-1,p[2]
       
       
        dic[len(idset)]=[a,b,c,d]
       
        if len(idset)>s*(1-e):
            break
       
        count+=1
   
    parm=dic[max(dic.keys())]
    a,b,c,d=parm
    dis=abs(a*X[:,0]+b*X[:,1]+c*X[:,2]+d)/(a**2+b**2+c**2)**0.5
       
    idset=[]
    for i ,d in enumerate(dis):
        if d <tao:
            idset.append(i)
    return np.array(idset)

planeids=PlaneRANSAC(origindata,0.4)
planedata=origindata[planeids]
planepcd = open3d.geometry.PointCloud()
planepcd.points = open3d.utility.Vector3dVector(planedata)


c=[0,0,255]
cs=np.tile(c,(planedata.shape[0],1))
planepcd.colors = open3d.utility.Vector3dVector(cs)

othersids=[]
for i in range(origindata.shape[0]):
    if i not in planeids:
        othersids.append(i)
otherdata=origindata[othersids]
otherpcd = open3d.geometry.PointCloud()
otherpcd.points = open3d.utility.Vector3dVector(otherdata)
c=[255,0,0]
cs=np.tile(c,(otherdata.shape[0],1))
otherpcd.colors=open3d.utility.Vector3dVector(cs)


#VISUALIZAR EL SUELO Y ENTORNO
open3d.visualization.draw_geometries([planepcd,otherpcd])
#VISUALIZAR SIN SUELO
open3d.visualization.draw_geometries([otherpcd])


# k mEANS
"""
Css=sklearn.cluster.KMeans(n_clusters=40).fit(otherdata)
ypred=np.array(Css.labels_)
ddraw=[]

colorset=[[222,0,0],[0,224,0],[0,255,255],[222,244,0],[255,0,255],[128,0,0]]

for cluuus in set(ypred):
   
    kaka=np.where(ypred==cluuus)
    ppk=open3d.geometry.PointCloud()
    ppk.points = open3d.utility.Vector3dVector(otherdata[kaka])

    c=colorset[cluuus%6]
    
    cs=np.tile(c,(otherdata[kaka].shape[0],1))
    ppk.colors=open3d.utility.Vector3dVector(cs)
    ddraw.append(ppk)

open3d.visualization.draw_geometries(ddraw)
ddraw.append(planepcd)
open3d.visualization.draw_geometries(ddraw)
"""


#DBSCAN

#Css=sklearn.cluster.DBSCAN(eps=0.50, min_samples=4).fit(otherdata)
Css=sklearn.cluster.DBSCAN(eps=0.45, min_samples=4).fit(otherdata)
#Css=sklearn.cluster.DBSCAN(eps=0.45, min_samples=22).fit(otherdata)

ypred=np.array(Css.labels_)
ddraw=[]

colorset=[[255,0,0],[0,224,0],[0,255,255],[222,244,0],[255,0,255],[128,0,0]]
for cluuus in set(ypred):
   
    kaka=np.where(ypred==cluuus)
    ppk=open3d.geometry.PointCloud()
    ppk.points = open3d.utility.Vector3dVector(otherdata[kaka])

    c=colorset[cluuus%6]
    if cluuus==-1:
        c=[0,0,0]
        #c=[255,255,255]

    cs=np.tile(c,(otherdata[kaka].shape[0],1))
    ppk.colors=open3d.utility.Vector3dVector(cs)
    ddraw.append(ppk)

open3d.visualization.draw_geometries(ddraw)

ddraw.append(planepcd)
open3d.visualization.draw_geometries(ddraw)


"""HDBSCAN
Css=hdbscan.HDBSCAN(min_cluster_size=15).fit(otherdata)
#Css=hdbscan.HDBSCAN(cluster_selection_epsilon=0.45, min_samples=22).fit(otherdata)
#Css=hdbscan.HDBSCAN(min_cluster_size=30,cluster_selection_epsilon=0.5).fit(otherdata)

ypred=np.array(Css.labels_)
ddraw=[]

colorset=[[222,0,0],[0,224,0],[0,255,255],[222,244,0],[255,0,255],[128,0,0]]
for cluuus in set(ypred):
   
    kaka=np.where(ypred==cluuus)
    ppk=open3d.geometry.PointCloud()
    ppk.points = open3d.utility.Vector3dVector(otherdata[kaka])

    c=colorset[cluuus%6]
    if cluuus==-1:
        c=[0,0,0]
        #c=[255,255,255]

    cs=np.tile(c,(otherdata[kaka].shape[0],1))
    ppk.colors=open3d.utility.Vector3dVector(cs)
    ddraw.append(ppk)

open3d.visualization.draw_geometries(ddraw)
ddraw.append(planepcd)
open3d.visualization.draw_geometries(ddraw)
"""



