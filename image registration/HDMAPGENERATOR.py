import numpy as np
import nest_asyncio
import math
from geopy import distance
from pyproj import Proj
from PIL import ImageOps
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation
from natsort import natsorted, realsorted, ns
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
import matplotlib
from statsmodels.tsa.ar_model import AutoReg
matplotlib.use('Agg')




class HDMAPCreator:

    def __init__(self, path_log, path_meta, logname):
        hf1 = h5py.File(path_log + logname + '.h5', 'r')
        self.dset = hf1['sensors/ApplanixDGPS']['position']
        self.dset1 = hf1['sensors/ApplanixDGPS/orientation']
        self.dsetA = hf1.get('sensors/ApplanixDGPS/timestamps')
        self.GPSTimestamp = np.array(self.dsetA)
        hf2 = h5py.File(path_meta + logname + '.h5', 'r')
        self.dsetB = hf2.get('radar_timestamps')
        self.dsetB = np.array(self.dsetB)
        self.Radar1Timestamp = self.dsetB[:, 0]

        # Extract all the Latitudes,Longitude and Orientation(Quaternions)
        self.Long = self.dset[:, 0]  # Longitude
        self.Lat = self.dset[:, 1]  # Latitude
        self.q0 = self.dset1[:, 0]  # Quaternions 1
        self.q1 = self.dset1[:, 1]  # Quaternions 2
        self.q2 = self.dset1[:, 2]  # Quaternions 3
        self.q3 = self.dset1[:, 3]  # Quaternions 4
        self.size = len(self.Long)  # Size of the dataset ususally len(long)
        self.Radar1Timestamp = np.round(self.Radar1Timestamp, 2)  # RadarTime stamps
        self.GPSTimestamp = np.round(self.GPSTimestamp,2)  # GPS TIMESTAMPS Converting the last two decimal places to the nearest one.
        nest_asyncio.apply()

    def quat_to_rotmat(self, q0, q1, q2, q3):
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

    def GPS_data_Transformation(self, Long, Lat, q0, q1, q2, q3):
        RotationM = np.zeros((len(Long), 3, 3), dtype=np.float64)

        H = np.zeros((len(Long), 4, 4), dtype=np.float64)

        for i in range(0, len(q0)):
            RotationM[i, :, :] = self.quat_to_rotmat(q0[i], q1[i], q2[i], q3[i])
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

    def GenerateCalibrated_data(self, Long, Lat, q0, q1, q2, q3):
        F4 = [1, 1, 1, 1]
        NewH1 = np.zeros((len(Long), 4, 4), dtype=np.float64)
        NewF5 = np.zeros((len(Long), 4), dtype=np.float64)

        T = np.eye(4)  # # Creating a 4*4 matrix

        R1 = Rotation.from_euler("zx", [201, 180], degrees=True).as_matrix()  #
        # Rotation and translation parameters of the antenna coordinate system relative to the vehicle coordinate system

        T[:3, :3] = R1  # moving the rotation parameters R1 to test matrix T
        #T1 = T.copy()  # Creating a copy to show the transformation without translations added
        T[:3, 3] = -np.matmul(R1, np.array([-0.335, -0.424, -1.082]))  # Adding translations to the test matrix T

        H1 = self.GPS_data_Transformation(Long, Lat, q0, q1, q2, q3)

        for i in range(0, len(Long)):
            NewH1[i, :, :] = np.matmul(H1[i, :, :], T)  # with translations
            NewF5[i, :] = np.matmul(NewH1[i, :, :], F4)

        utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
        longitude, latitude = utm_proj(NewF5[:, 0], NewF5[:, 1], inverse=True)

        return longitude, latitude

    def add_ar_noise(self,Long, Lat, ar_order=[5], scaling_factor=5):
        # Create an AR model using the given order and coefficients
        ar_model_long = AutoReg(Long, lags=ar_order, trend='n')
        ar_model_lat = AutoReg(Lat, lags=ar_order, trend='n')

        # Fit the AR model to the data
        ar_resul_long = ar_model_long.fit()
        ar_resul_lat = ar_model_lat.fit()

        # use resid function
        ar_resul_long_res = ar_resul_long.resid
        ar_resul_lat_res = ar_resul_lat.resid

        # Scale the AR noise by the desired factor
        scaled_ar_noise_long = scaling_factor * ar_resul_long_res
        scaled_ar_noise_lat = scaling_factor * ar_resul_lat_res

        # removing NAN for lags
        scaled_ar_noise_long = np.nan_to_num(scaled_ar_noise_long, nan=0)
        scaled_ar_noise_lat = np.nan_to_num(scaled_ar_noise_lat, nan=0)

        # Add the scaled AR noise to the longitude and latitude
        x_noisy = Long.copy()
        y_noisy = Lat.copy()


        pad_width = (ar_order[0], 0)  # (before, after) padding widths
        scaled_ar_noise_long = np.pad(scaled_ar_noise_long, pad_width, mode='constant')
        scaled_ar_noise_lat = np.pad(scaled_ar_noise_lat, pad_width, mode='constant')

        np.save(r'C:\Users\bjqb7h\Downloads\Thesis2022\GT TRANSLATIONS\\' + 'dxGTAR', scaled_ar_noise_long)
        np.save(r'C:\Users\bjqb7h\Downloads\Thesis2022\GT TRANSLATIONS\\' + 'dyGTAR', scaled_ar_noise_lat)

        x_noisy = x_noisy + scaled_ar_noise_long
        y_noisy = y_noisy + scaled_ar_noise_lat

        return x_noisy, y_noisy

    def add_gaussian_noise(self,Long, Lat,log):
        cov_matrix = np.array([[0.0000000005, 0], [0, 0.0000000005]])
        noise = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=Long.size)

        # Calculate the magnitude of the noise
        magnitudes = np.linalg.norm(noise, axis=1, keepdims=True)

        # Normalize the noise vectors to have the same direction
        normalized_noise = noise / magnitudes

        # Adjust the magnitude of the noise vectors based on a scaling factor
        scaling_factor = 1.0  # Adjust this value to control the magnitude of the noise
        adjusted_noise = normalized_noise * magnitudes * scaling_factor
        np.save(r'C:\Users\bjqb7h\Downloads\Thesis2022\GT TRANSLATIONS\TRUE NOISE\\' + 'DXGTNEW_'+log, adjusted_noise[:, 0])
        np.save(r'C:\Users\bjqb7h\Downloads\Thesis2022\GT TRANSLATIONS\TRUE NOISE\\' + 'DYGTNEW_'+log, adjusted_noise[:, 1])

        # Add the adjusted noise to the longitude and latitude
        x_noisy = Long + adjusted_noise[:, 0]
        y_noisy = Lat + adjusted_noise[:, 1]

        return x_noisy, y_noisy

    def quat2eulers(self, q0: float, q1: float, q2: float, q3: float):

        # Compute yaw-pitch-roll Euler angles from a quaternion.

        # Args
        # q0: Scalar component of quaternion.
        # q1, q2, q3: Vector components of quaternion.

        # Returns
        # (roll, pitch, yaw) (tuple): 321 Euler angles in radians
        # roll = math.atan2(2 * ((q2 * q3) + (q0 * q1)),q0**2 - q1**2 - q2**2 + q3**2)  # radians
        # pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
        yaw = math.atan2(2 * ((q1 * q2) + (q0 * q3)), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)
        degree = 180 / math.pi
        yaw = yaw * degree
        return yaw

    def CreateAngle(self, q0, q1, q2, q3, size):
        Angle = np.zeros(size, dtype=float)
        for i in range(size):
            Angle[i] = self.quat2eulers(q0[i], q1[i], q2[i], q3[i])
        return Angle

    def BBOX_Generation(self, Lat, Long):
        padding_Lat = 0.001850
        Padding_long = 0.001850
        bbox = (min(Lat) - padding_Lat, max(Lat) + padding_Lat, min(Long) - Padding_long, max(Long) + Padding_long)
        gr = ox.graph_from_bbox(*bbox, network_type='all')
        geo = ox.geometries_from_bbox(*bbox, tags={'building': True})
        return gr, geo, bbox

    def Convert_To_M(self, Lat, Long):
        X, Y, O, P = utm.from_latlon(Lat, Long)
        X1 = X + 80  # 80 meter to match the semseg grid
        Y1 = Y + 80
        NewLat,NewLong = utm.to_latlon(X1, Y1, O, P)
        diff_Lat = NewLat - Lat
        diff_Long = NewLong - Long
        return diff_Long, diff_Lat

    def get_fig(self, bbox=None, size_inches=(15, 15)):

        fig, ax = plt.subplots(frameon=False)
        fig.set_size_inches(*size_inches)
        ax.set_facecolor("green")
        plt.close(fig)
        ox.plot.plot_figure_ground(GRoad, node_size=0, network_type='all', ax=ax, edge_color="black", bgcolor='blue',bbox=bbox,default_width=18)
        ox.plot.plot_footprints(GBuilding, ax=ax, bgcolor='green', bbox=bbox)
        plt.close()
        return fig

    def crop_center(self, pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2,
                                 (img_height - crop_height) // 2,
                                 (img_width + crop_width) // 2,
                                 (img_height + crop_height) // 2))


    def Save_Maps_Sep(self, Lat, Long, DIFF_LAT, DIFF_LONG, i):
        Sep_path_save = 'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS/'
        North = Lat - DIFF_LAT
        South = Lat + DIFF_LAT
        East = Long - DIFF_LONG
        West = Long + DIFF_LONG
        fig = self.get_fig(bbox=(North, South, East, West), size_inches=(6, 6))
        fig.canvas.draw()
        fig_array = np.array(fig.canvas.renderer.buffer_rgba())
        np.save(Sep_path_save + 'Save%d.npy' % i, fig_array)
        #fig.savefig(Sep_path_save + 'Save%d.png' % i, bbox_inches='tight', pad_inches=0)

    def RotateMap_Sep(self, i):
        Sep_Rdirectory = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS'
        Sep_path_Rotate = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS'
        os.chdir(Sep_Rdirectory)
        Narray = np.load(Sep_path_Rotate + '/Save%d.npy' % i)
        Angle = self.CreateAngle(self.q0, self.q1, self.q2, self.q3, self.size)
        # rotated_array = rotate(Narray, angle=-180 - Angle[i], reshape=False)
        rotated_array = imutils.rotate_bound(Narray, angle=-180 - (Angle[i]))
        filename = 'Rotate%d.npy' % i
        np.save(filename, rotated_array)


    def Crop_image_Sep(self, i):
        Crop_directory = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS/'
        #Crop_directory = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS/'
        Crop_save_path = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
        os.chdir(Crop_directory)
        arr = np.load(Crop_directory + 'Rotate%d.npy' % i)
        #arr = np.load(Crop_directory + 'Save%d.npy' % i)
        img = Image.fromarray(np.uint8(arr))
        cropped_img = self.crop_center(img, 160, 160)
        cropped_arr = np.asarray(cropped_img)
        np.save(Crop_save_path + '/Cropped%d.npy' % i, cropped_arr)


if __name__ == '__main__':
    logpath = r'C:/Users/bjqb7h/Downloads/Thesis2022/GPS DATA/GPS DATA/'
    metapath = r'C:/Users/bjqb7h/Downloads/Thesis2022/Meta/NEW META/'
    Crop_save_path = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
    #log = 'AtCityBMW_Applanix-20220221T155149Z890'
    hDF5_PATH = 'C:/Users/bjqb7h/Downloads/Thesis2022/NUMPY/New/New Logs pt 2'

    log_names = ['AtCityBMW_Applanix-20220318T094204Z741', 'AtCityBMW_Applanix-20220318T094401Z320', 'AtCityBMW_Applanix-20220318T094550Z437', 'AtCityBMW_Applanix-20220318T094737Z538', 'AtCityBMW_Applanix-20220318T094925Z141', 'AtCityBMW_Applanix-20220318T095113Z773', 'AtCityBMW_Applanix-20220318T095301Z181', 'AtCityBMW_Applanix-20220318T095448Z233', 'AtCityBMW_Applanix-20220318T095644Z089', 'AtCityBMW_Applanix-20220318T095843Z354', 'AtCityBMW_Applanix-20220318T100037Z829', 'AtCityBMW_Applanix-20220318T100225Z090', 'AtCityBMW_Applanix-20220318T100433Z841', 'AtCityBMW_Applanix-20220318T100631Z941', 'AtCityBMW_Applanix-20220318T100820Z721', 'AtCityBMW_Applanix-20220318T101011Z169', 'AtCityBMW_Applanix-20220318T101201Z633', 'AtCityBMW_Applanix-20220318T101354Z865', 'AtCityBMW_Applanix-20220318T101538Z705', 'AtCityBMW_Applanix-20220318T101728Z289', 'AtCityBMW_Applanix-20220318T101918Z285', 'AtCityBMW_Applanix-20220318T102110Z897', 'AtCityBMW_Applanix-20220318T102258Z061', 'AtCityBMW_Applanix-20220318T102634Z669', 'AtCityBMW_Applanix-20220318T103024Z773', 'AtCityBMW_Applanix-20220318T103220Z261', 'AtCityBMW_Applanix-20220318T103409Z941', 'AtCityBMW_Applanix-20220318T103559Z173', 'AtCityBMW_Applanix-20220318T103739Z649', 'AtCityBMW_Applanix-20220318T104125Z325', 'AtCityBMW_Applanix-20220318T104459Z453', 'AtCityBMW_Applanix-20220318T105057Z890', 'AtCityBMW_Applanix-20220318T105845Z225', 'AtCityBMW_Applanix-20220318T110041Z637', 'AtCityBMW_Applanix-20220318T110817Z249', 'AtCityBMW_Applanix-20220318T111010Z861', 'AtCityBMW_Applanix-20220318T111541Z354', 'AtCityBMW_Applanix-20220318T111728Z374', 'AtCityBMW_Applanix-20220318T111916Z149', 'AtCityBMW_Applanix-20220318T112250Z844', 'AtCityBMW_Applanix-20220318T112442Z710', 'AtCityBMW_Applanix-20220318T112635Z186', 'AtCityBMW_Applanix-20220318T112820Z701', 'AtCityBMW_Applanix-20220318T113015Z354', 'AtCityBMW_Applanix-20220323T085701Z724', 'AtCityBMW_Applanix-20220323T085852Z573', 'AtCityBMW_Applanix-20220323T090050Z537', 'AtCityBMW_Applanix-20220323T090243Z117', 'AtCityBMW_Applanix-20220323T090437Z250', 'AtCityBMW_Applanix-20220323T090630Z689', 'AtCityBMW_Applanix-20220323T090821Z577', 'AtCityBMW_Applanix-20220323T091012Z597', 'AtCityBMW_Applanix-20220323T091159Z705', 'AtCityBMW_Applanix-20220323T091349Z869', 'AtCityBMW_Applanix-20220323T091539Z069', 'AtCityBMW_Applanix-20220323T091728Z820', 'AtCityBMW_Applanix-20220323T091926Z085', 'AtCityBMW_Applanix-20220323T092127Z117', 'AtCityBMW_Applanix-20220323T092316Z445', 'AtCityBMW_Applanix-20220323T092503Z213', 'AtCityBMW_Applanix-20220323T092700Z781', 'AtCityBMW_Applanix-20220323T092846Z669', 'AtCityBMW_Applanix-20220323T093039Z709', 'AtCityBMW_Applanix-20220323T093228Z792', 'AtCityBMW_Applanix-20220323T093418Z297', 'AtCityBMW_Applanix-20220323T093611Z394', 'AtCityBMW_Applanix-20220323T093811Z670', 'AtCityBMW_Applanix-20220323T093957Z861', 'AtCityBMW_Applanix-20220323T094145Z213', 'AtCityBMW_Applanix-20220323T094334Z953', 'AtCityBMW_Applanix-20220323T094529Z813', 'AtCityBMW_Applanix-20220323T094717Z793', 'AtCityBMW_Applanix-20220323T094908Z225', 'AtCityBMW_Applanix-20220323T095115Z350', 'AtCityBMW_Applanix-20220323T095304Z309', 'AtCityBMW_Applanix-20220323T095458Z289', 'AtCityBMW_Applanix-20220323T095652Z949', 'AtCityBMW_Applanix-20220323T095846Z125', 'AtCityBMW_Applanix-20220323T100030Z743', 'AtCityBMW_Applanix-20220323T100220Z293', 'AtCityBMW_Applanix-20220323T100409Z501', 'AtCityBMW_Applanix-20220323T100600Z717', 'AtCityBMW_Applanix-20220323T100749Z653', 'AtCityBMW_Applanix-20220323T100946Z153', 'AtCityBMW_Applanix-20220323T101138Z893', 'AtCityBMW_Applanix-20220323T101332Z493', 'AtCityBMW_Applanix-20220323T101509Z753', 'AtCityBMW_Applanix-20220323T101645Z981', 'AtCityBMW_Applanix-20220323T101834Z585', 'AtCityBMW_Applanix-20220323T102024Z437', 'AtCityBMW_Applanix-20220323T102215Z693', 'AtCityBMW_Applanix-20220323T102405Z673', 'AtCityBMW_Applanix-20220323T102554Z625', 'AtCityBMW_Applanix-20220323T102746Z493', 'AtCityBMW_Applanix-20220323T102937Z449', 'AtCityBMW_Applanix-20220323T103129Z453', 'AtCityBMW_Applanix-20220323T103320Z365', 'AtCityBMW_Applanix-20220323T103511Z433', 'AtCityBMW_Applanix-20220323T103656Z589', 'AtCityBMW_Applanix-20220323T103843Z457', 'AtCityBMW_Applanix-20220323T104033Z041', 'AtCityBMW_Applanix-20220323T104227Z102', 'AtCityBMW_Applanix-20220323T104413Z481', 'AtCityBMW_Applanix-20220323T104604Z673', 'AtCityBMW_Applanix-20220323T104748Z989', 'AtCityBMW_Applanix-20220323T104936Z925', 'AtCityBMW_Applanix-20220323T105123Z233', 'AtCityBMW_Applanix-20220323T105318Z205', 'AtCityBMW_Applanix-20220323T105505Z617']
    for log in log_names:
        tt = HDMAPCreator(path_log=logpath, path_meta=metapath, logname=log)

        CLong, CLat = tt.GenerateCalibrated_data(Long=tt.Long, Lat=tt.Lat, q0=tt.q0, q1=tt.q1, q2=tt.q2, q3=tt.q3)

        Noise_Long, Noise_Lat = tt.add_gaussian_noise(Long=CLong, Lat=CLat,log=log)#for GPS
        #Noise_Long,Noise_Lat = tt.add_ar_noise(Long=CLong, Lat=CLat)

        # Read Lat and Long and generate BBOX for roads and buildings
        GRoad, GBuilding, BBOX = tt.BBOX_Generation(Noise_Lat, Noise_Long) #for GPS
        #GRoad, GBuilding, BBOX = tt.BBOX_Generation(CLat,CLong) #For DGPS
        # Create a bbox of 80m
        #DIFF_LONG, DIFF_LAT = tt.Convert_To_M(Lat=tt.Lat, Long=tt.Long)
        # Creating Angle of rotation
        AngleOFROTATIOn = tt.CreateAngle(q0=tt.q0, q1=tt.q1, q2=tt.q2, q3=tt.q3,size=tt.size)  # Generating Yaw from the 4 quaternion

        for i in trange(tt.GPSTimestamp.size):
            if (tt.GPSTimestamp[i] == tt.Radar1Timestamp[:]).any():
                DIFF_LONG, DIFF_LAT = tt.Convert_To_M(Noise_Lat[i], Noise_Long[i]) #For GPS
                #DIFF_LONG, DIFF_LAT = tt.Convert_To_M(CLat[i], CLong[i]) #For DGPS
                tt.Save_Maps_Sep(Noise_Lat[i], Noise_Long[i],DIFF_LAT,DIFF_LONG, i)#For GPS
                #tt.Save_Maps_Sep(CLat[i], CLong[i], DIFF_LAT, DIFF_LONG, i)#for DGPS
                tt.RotateMap_Sep(i)
                tt.Crop_image_Sep(i)

        # Load all the arrays in the folder and convert them to numpy
        array_files = natsorted(glob.glob(Crop_save_path + '/*.npy'), alg=ns.IGNORECASE)
        numpydata = [np.load(file) for file in array_files]
        numpydata = np.stack(numpydata)
        # Converting the numpy to HDF5
        with h5py.File(hDF5_PATH + '/' + log + 'GPSNOISE1' + '.hdf5', 'w') as hf:
            grp = hf.create_group(log)
            dataset = grp.create_dataset('Image data', data=numpydata)

        dir_name = "C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS"
        test = os.listdir(dir_name)

        for item in test:
            if item.endswith(".npy"):
                os.remove(os.path.join(dir_name, item))

        Sep_path_save = 'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS/'
        test2 = os.listdir(Sep_path_save)
        for item in test2:
            if item.endswith(".npy"):
                os.remove(os.path.join(Sep_path_save, item))

        Crop_save_path = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
        test3 = os.listdir(Crop_save_path)
        for item in test3:
            if item.endswith(".npy"):
                os.remove(os.path.join(Crop_save_path, item))

