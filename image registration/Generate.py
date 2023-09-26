import numpy as np
import nest_asyncio
import math
from scipy.spatial.transform import Rotation
from geopy import distance
from pyproj import Proj
import matplotlib
matplotlib.use('Agg')
from natsort import natsorted,realsorted, ns
from tqdm.notebook import trange, tqdm
from time import sleep
import osmnx as ox
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

path_log = r'C:/Users/bjqb7h/Downloads/Thesis2022/GPS DATA/'
metapath = r'C:/Users/bjqb7h/Downloads/Thesis2022/Meta/'
Crop_save_path =r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
log='AtCityBMW_Applanix-20220601T115459Z469'
hDF5_PATH = 'C:/Users/bjqb7h/Downloads/Thesis2022/NUMPY/New'

hf1 = h5py.File(path_log + log + '.h5', 'r')
dset = hf1['sensors/ApplanixDGPS']['position']
dset1 = hf1['sensors/ApplanixDGPS/orientation']
dsetA = hf1.get('sensors/ApplanixDGPS/timestamps')
GPSTimestamp = np.array(dsetA)
hf2 = h5py.File(metapath+ log+ '.h5', 'r')
dsetB = hf2.get('radar_timestamps')
dsetB = np.array(dsetB)
Radar1Timestamp = dsetB[:, 0]

# Extract all the Latitudes,Longitude and Orientation(Quaternions)
Long = dset[:, 0]  # Longitude
Lat =dset[:, 1]  # Latitude
q0 = dset1[:, 0]  # Quaternions 1
q1 = dset1[:, 1]  # Quaternions 2
q2 = dset1[:, 2]  ##Quaternions 3
q3 = dset1[:, 3]  ##Quaternions 4
size = len(Long)  # Size of the dataset ususally len(long)
Radar1Timestamp = np.round(Radar1Timestamp, 2)  # RadarTime stamps
GPSTimestamp = np.round(GPSTimestamp,2)  # GPS TIMESTAMPS Converting the last two decimal places to the nearest one.
nest_asyncio.apply()


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


def GPS_data_Transformation( Long, Lat, q0, q1, q2, q3):
    RotationM = np.zeros((len(Long), 3, 3), dtype=np.float64)

    H = np.zeros((len(Long), 4, 4), dtype=np.float64)

    for i in range(0, len(q0)):
        RotationM[i, :, :] = quat_to_rotmat(q0[i], q1[i], q2[i], q3[i])
    # Generating the transformation matrix H which contain parameters
    # to transform the GPS coordinates from the antenna coordinate system to the vehicle coordinate system.

    utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
    x, y = utm_proj(Long, Lat)  # Converting the local coordinate system to world coordinate system

    for i in range(len(Long)):
        H[i, :, :] = np.eye(4)
        H[i, :3, :3] = RotationM[i, :, :]
        t = np.array([x[i], y[i], 0])  # GPS Longitude . latitude in x ,y,z
        H[i, :3, 3] = t.flatten()

    return H


# def GenerateCalibrated_data( Long, Lat, q0, q1, q2, q3):
    F1 = [0, 0, 0, 1]
    F2 = [1, 0, 0, 1]
    F3 = [0, 1, 0, 1]
    F4 = [1, 1, 1, 1]
    NewH1 = np.zeros((len(Long), 4, 4), dtype=np.float64)
    NewF5 = np.zeros((len(Long), 4), dtype=np.float64)

    T = np.eye(4)  # # Creating a 4*4 matrix

    R1 = Rotation.from_euler("zx", [201, 180], degrees=True).as_matrix()  #
    # Rotation and translation parameters of the antenna coordinate system relative to the vehicle coordinate system

    T[:3, :3] = R1  # moving the rotation parameters R1 to test matrix T
    T1 = T.copy()  # Creating a copy to show the transformation without translations added
    T[:3, 3] = -np.matmul(R1, np.array([-0.335, -0.424, -1.082]))  # Adding translations to the test matrix T

    H1 = GPS_data_Transformation(Long, Lat, q0, q1, q2, q3)

    for i in range(0, len(Long)):
        NewH1[i, :, :] = np.matmul(H1[i, :, :], T)
        NewH2 = np.matmul(H1[i, :, :], T1)
        NewF1 = np.matmul(NewH1, F1)
        NewF2 = np.matmul(NewH1, F2)
        NewF3 = np.matmul(NewH1, F3)
        NewF4 = np.matmul(NewH2, F1)
        # with translations
        NewF5[i, :] = np.matmul(NewH1[i, :, :], F4)

        plt.plot([NewF1[0], NewF2[0]], [NewF1[1], NewF2[1]], 'b-', label='Reference for x axis')
        plt.plot([NewF1[0], NewF3[0]], [NewF1[1], NewF3[1]], 'r-', label='Reference for y axis ')
        plt.plot(NewF4[0], NewF4[1], 'g.', label='Uncalibrated data')

    plt.axis('equal')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.show()
        # utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
        # longitude, latitude = utm_proj(NewF5[:, 0], NewF5[:, 1], inverse=True)

    # return longitude, latitude