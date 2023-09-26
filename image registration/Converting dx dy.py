import numpy as np
import sys
from math import cos, radians
import matplotlib.transforms as mtransforms
from shapely.geometry import Polygon
import nest_asyncio
import math
from geopy import distance
from math import cos
from haversine import haversine, Unit
from pyproj import Proj
import pyproj
#import rasterio
import random
from natsort import natsorted,realsorted, ns
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
from scipy.spatial.transform import Rotation
import geotiler
import cv2
import matplotlib.pyplot as plt
import h5py
import imutils
import mplleaflet
from PIL import Image, ImageDraw


Logname = 'AtCityBMW_Applanix-20220601T115459Z469'
path = 'C:/Users/bjqb7h/Downloads/Thesis2022/GPS DATA/'
path_position = 'sensors/ApplanixDGPS'
path_orientation = 'sensors/ApplanixDGPS/orientation'
path_timestamps = 'sensors/ApplanixDGPS/timestamps'
path_meta = 'C:/Users/bjqb7h/Downloads/Thesis2022/Meta/'
path_save ='C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset'
Rdirectory =r'C:\Users\bjqb7h\Downloads\Thesis2022\HD Maps_Rotate'
path_rotate = 'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/'
path_crop ='C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate'
path_cropped ='C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped'
Sep_path_save ='C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS/'
Sep_Rdirectory = r'C:\Users\bjqb7h\Downloads\Thesis2022\HD Maps_Rotate\GPSRADAR\RotatedMAPS'
Sep_path_Rotate = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS'
Crop_directory= r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS'
Crop_save_path =r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
hDF5_PATH = 'C:/Users/bjqb7h/Downloads/Thesis2022/NUMPY/New'
dir0 = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS'
dir1 = 'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS'
dir2 = 'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'


#Read Hdf5 filel for the Radar data and GPS data.
hf1 = h5py.File(path+Logname+'.h5','r')
dset = hf1[path_position]['position']
dset1= hf1[path_orientation]
dsetA = hf1.get(path_timestamps)
GPSTimestamp=np.array(dsetA)



hf2 = h5py.File(path_meta+Logname+'.h5','r')
dsetB= hf2.get('radar_timestamps')
dsetB=np.array(dsetB)
Radar1Timestamp = dsetB[:,0]
Radar1Timestamp = np.round(Radar1Timestamp,2)#RadarTime stamps
GPSTimestamp= np.round(GPSTimestamp,2)#GPS


#Extract all the Latitudes,Longitude and Orientation(Quaternions)
Long=dset[:,0] #Longitude
Lat=dset[:,1]  #Latitude
q0=dset1[:,0]  #Quaternions 1
q1=dset1[:,1]  #Quaternions 2
q2=dset1[:,2]  ##Quaternions 3
q3=dset1[:,3] ##Quaternions 4

size= len(Long) #Size of the dataset ususally len(long)

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


def GPS_data_Transformation(Long, Lat, q0, q1, q2, q3):
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


def GenerateCalibrated_data(Long, Lat, q0, q1, q2, q3):
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
        NewH1[i, :, :] = np.matmul(H1[i, :, :], T)  # with translations
        NewF5[i, :] = np.matmul(NewH1[i, :, :], F4)

    utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
    longitude, latitude = utm_proj(NewF5[:, 0], NewF5[:, 1], inverse=True)

    return longitude, latitude


def add_gaussian_noise(x,y,N):
    noise = np.random.multivariate_normal(mean=[0, 0], cov=[[N, 0], [0, N]], size=x.shape)
    x_noisy = x + noise[:, 0]
    y_noisy = y + noise[:, 1]
    return x_noisy, y_noisy


def calculate_mse(original_latitude, original_longitude, new_latitude, new_longitude):
    mse = 0
    for x1, y1, x2, y2 in zip(original_latitude, original_longitude, new_latitude, new_longitude):
        mse += (x1 - x2) ** 2 + (y1 - y2) ** 2
    mse = mse / len(original_latitude)
    return math.sqrt(mse)

def calculate_mse_meters(original_latitude, original_longitude, new_latitude, new_longitude):
    mse = 0
    for x1, y1, x2, y2 in zip(original_latitude, original_longitude, new_latitude, new_longitude):
        lat1, lon1, lat2, lon2 = map(math.radians, [x1, y1, x2, y2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        m = 6371000 * c
        mse += m ** 2
    mse = mse / len(original_latitude)
    return math.sqrt(mse)

def calculate_sse_meters(original_latitude, original_longitude, new_latitude, new_longitude):
    sse = 0
    for x1, y1, x2, y2 in zip(original_latitude, original_longitude, new_latitude, new_longitude):
        lat1, lon1, lat2, lon2 = map(math.radians, [x1, y1, x2, y2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        m = 6371000 * c
        sse += m ** 2
    return sse

def calculate_sse(original_latitude, original_longitude, new_latitude, new_longitude):
    return np.sum((original_latitude - new_latitude)**2 + (original_longitude - new_longitude)**2)

#####################################################


l1,l2 = GenerateCalibrated_data( Long, Lat, q0, q1, q2, q3)
utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
x, y = utm_proj(l1, l2)
NLong,NLat = add_gaussian_noise(x,y,N=9)
longitude, latitude = utm_proj(NLong, NLat, inverse=True)


#####################################################
#Syncing the radar timestamps with GPS timestamps
org_lat = np.zeros(2093, dtype=np.float64)
org_long = np.zeros(2093, dtype=np.float64)
gps_lat = np.zeros(2093, dtype=np.float64)
gps_long = np.zeros(2093, dtype=np.float64)

gps_timestamps = GPSTimestamp
radar_timestamps = Radar1Timestamp

# use numpy's searchsorted function to find the closest radar timestamps to the GPS timestamps
closest_radar_indices = np.searchsorted(radar_timestamps, gps_timestamps, side='left')

# create an empty list to store the synchronized gps timestamps
gps_sync_index = []

# iterate through the closest radar indices
for i, closest_index in enumerate(closest_radar_indices):
    # check if the closest index is outside of the radar timestamp array bounds
    if closest_index == 0:
        closest_radar_ts = radar_timestamps[0]
    elif closest_index == len(radar_timestamps):
        closest_radar_ts = radar_timestamps[-1]
    else:
        # if the current GPS timestamp is greater than the radar timestamp, take the radar_timestamp before
        if abs(radar_timestamps[closest_index] - gps_timestamps[i]) > abs(
                radar_timestamps[closest_index - 1] - gps_timestamps[i]):
            closest_radar_ts = radar_timestamps[closest_index - 1]
        else:
            closest_radar_ts = radar_timestamps[closest_index]

    if closest_radar_ts == gps_timestamps[i]:
        gps_sync_index.append(i)
# print(len(gps_sync_index))
org_lat = l2[gps_sync_index]
org_long = l1[gps_sync_index]
gps_lat = latitude[gps_sync_index]
gps_long = longitude[gps_sync_index]

##############################################
#load the dx and dy translations.
dx = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\dxdy\\dx_bounds_-2.5_2.5_-2.5_2.5.npy')
dy = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\dxdy\\dy_bounds_-2.5_2.5_-2.5_2.5.npy')

#dx_bounds_-5_5_-5_5
#dy_bounds_-5_5_-5_5
#dy_bounds_-4_4_-4_3
#dx_bounds_-4_4_-4_3
# Plot histograms of the translations
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].hist(dx, bins=64, color='blue', alpha=0.5)
axs[0].set_xlabel('dx translation')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Histogram of dx translations')

axs[1].hist(dy, bins=64, color='green', alpha=0.5)
axs[1].set_xlabel('dy translation')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of dy translations')

plt.show()


##################################################
#getting pixel size in meters

original_resolution_x = 160 / 438
original_resolution_y = 160 / 462
cropped_physical_size_x = 160*original_resolution_x
cropped_physical_size_y = 160*original_resolution_y
pixel_size_x=cropped_physical_size_x/160
pixel_size_y=cropped_physical_size_y/160

utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
x1, y1 = utm_proj(gps_long, gps_lat)
x2, y2 = utm_proj(org_long, org_lat)
dx_m = np.zeros(2093, dtype=np.float64)
dy_m = np.zeros(2093, dtype=np.float64)
dLatm = np.zeros(2093, dtype=np.float64)
dLonm = np.zeros(2093, dtype=np.float64)
for i in range(2093):
    dx_m[i] = dx[i] * pixel_size_x
    dy_m[i] = dy[i] * pixel_size_y
    dLatm[i] = y1[i] + dy_m[i]
    dLonm[i] = x1[i] + dx_m[i]

mse_meters4 = calculate_mse(x2, y2, dLonm, dLatm)
mse_meters5 = calculate_mse(x2, y2, x1, y1)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error((x2, y2), (dLonm, dLatm))
mae1 = mean_absolute_error((x2, y2), (x1, y1))

print("The RMSE for the Original DGPS vs Translated GPS", mse_meters4)
print("The RMSE for the Original DGPS vs GPS", mse_meters5)
print("The MAE for the Original DGPS vs Translated GPS", mae)
print("The MAE for the Original DGPS vs GPS", mae1)

