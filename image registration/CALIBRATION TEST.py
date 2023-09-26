import numpy as np
import sys
import nest_asyncio
import math
from scipy.spatial.transform import Rotation
from geopy import distance
from math import cos
from math import sin
from math import asin
from math import atan2
from haversine import Unit
from pyproj import Proj
import pyproj
from math import cos, radians

# import rasterio
import random
from natsort import natsorted, realsorted, ns
from tqdm.notebook import trange, tqdm
from time import sleep
import osmnx as ox
import haversine as hs
import shutil
from numpy import asarray
from IPython import display
import os
from os import listdir
import utm
import geopandas
from pathlib import Path
import glob
import concurrent.futures
import geotiler
import cv2
import matplotlib.pyplot as plt
import h5py
import imutils
import mplleaflet
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use('TkAgg')

Logname = 'AtCityBMW_Applanix-20220601T115459Z469'
path = 'C:/Users/bjqb7h/Downloads/Thesis2022/GPS DATA/'
path_position = 'sensors/ApplanixDGPS'
path_orientation = 'sensors/ApplanixDGPS/orientation'
path_timestamps = 'sensors/ApplanixDGPS/timestamps'
path_meta = 'C:/Users/bjqb7h/Downloads/Thesis2022/Meta'

hDF5_PATH = 'C:/Users/bjqb7h/Downloads/Thesis2022/NUMPY/New'
dir0 = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS'
dir1 = 'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS'
dir2 = 'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'

# Read Hdf5 filel for the Radar data and GPS data.
hf1 = h5py.File(path + Logname + '.h5', 'r')
dset = hf1[path_position]['position']
dset1 = hf1[path_orientation]
dsetA = hf1.get(path_timestamps)
GPSTimestamp = np.array(dsetA)

# Extract all the Latitudes,Longitude and Orientation(Quaternions)
Long = dset[:, 0]  # Longitude
Lat = dset[:, 1]  # Latitude
Alt = dset[:, 2]  # altitude
q0 = dset1[:, 0]  # Quaternions 1
q1 = dset1[:, 1]  # Quaternions 2
q2 = dset1[:, 2]  ##Quaternions 3
q3 = dset1[:, 3]  ##Quaternions 4

size = len(Long)  # Size of the dataset ususally len(long)

# nest_asyncio.apply()



GNSS_lever_arm = np.array([-0.335, -0.424, -1.082])
IMU_lever_arm = np.array([-0.181, -0.001, -0.206])
IMU_angles = np.array([-0.201, 0.100, -0.229])
ref_to_veh_angles = np.array([0, 0, 0])
x_transformed = np.zeros((len(Long), 1), dtype=np.float64)
y_transformed = np.zeros((len(Long), 1), dtype=np.float64)





def quat_to_rotmat(q0, q1, q2, q3):
    # Converts quaternions q0, q1, q2, and q3 to a rotation matrix.

    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * q2 ** 2 - 2 * q3 ** 2
    R[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    R[0, 2] = 2 * q0 * q2 + 2 * q1 * q3
    R[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    R[1, 1] = 1 - 2 * q1 ** 2 - 2 * q3 ** 2
    R[1, 2] = 2 * q2 * q3 - 2 * q0 * q1
    R[2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    R[2, 1] = 2 * q0 * q1 + 2 * q2 * q3
    R[2, 2] = 1 - 2 * q1 ** 2 - 2 * q2 ** 2
    return R



RotationM = np.zeros((len(Long), 3, 3), dtype=np.float64)
for i in range(0, len(q0)):
    RotationM[i, :, :] = quat_to_rotmat(q0[i], q1[i], q2[i], q3[i])

# print(RotationM[2,:,:])

# Z=Quaternion(q0[1],q1[1],q2[1],q3[1]).rotation_matrix


H = np.zeros((len(Long), 4, 4), dtype=np.float64)

utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
x, y = utm_proj(Long, Lat)



for i in range(len(Long)):
    H[i, :, :] = np.eye(4)
    H[i, :3, :3] = RotationM[i, :, :]
    t = np.array([x[i], y[i], 0])  # GPS Longitude . latitude in x ,y,z
    H[i, :3, 3] = t.flatten()

print(H[0,:3,:3])

# print(H[2,:,:])


F1 = [0, 0, 0, 1]
F2 = [1, 0, 0, 1]
F3=  [0, 1, 0, 1]



T = np.eye(4)  #
R1=Rotation.from_euler("zx",[203,180],degrees=True).as_matrix()
print(R1)
# T[:3,:3]=R1
T[:3,:3] = R1
T1=T.copy()
T[:3,3] = -np.matmul(R1,np.array([-0.335,-0.424, -1.082]))
print(T)
#inverse of T and apply to the dgps maps : it should give map in vcs{Resampling}
#Map centre transform wit
#plot this position and line into gps map

for i in range(0,len(Long),5):
    NewH = np.matmul(H[i, :, :], T)
    NewH1 = np.matmul(H[i, :, :], T1)
    NewF1 = np.matmul(NewH, F1)
    NewF2 = np.matmul(NewH, F2)
    NewF3 = np.matmul(NewH, F3)
    NewF4= np.matmul(NewH1, F1)


    plt.plot([NewF1[0], NewF2[0]], [NewF1[1], NewF2[1]], 'b-',label='Reference for x axis')
    plt.plot([NewF1[0], NewF3[0]], [NewF1[1], NewF3[1]], 'r-',label='Reference for y axis ')
    plt.plot(NewF4[0], NewF4[1], 'g.',label='Uncalibrated data')


plt.axis('equal')
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()


