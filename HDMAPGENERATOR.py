import numpy as np
import nest_asyncio
import math 
from natsort import natsorted,realsorted, ns
from tqdm.notebook import trange, tqdm
from time import sleep
import osmnx as ox
import torchvision
import shutil
from numpy import asarray
import folium
from IPython import display
import torch
from torchvision import transforms
import os
from os import listdir
import utm
import geopandas
from pathlib import Path
import glob
import concurrent.futures
import geotiler
import torch
import cv2 
import matplotlib.pyplot as plt
import h5py
import htmlwebshot
import imutils
import mplleaflet
from PIL import Image, ImageDraw



class HDMAPCreator:

    def __init__(self,path_log,path_meta,Noise,logname):
        hf1 = h5py.File(path_log+logname+'.h5','r')
        self.dset = hf1['sensors/ApplanixDGPS']['position']
        self.dset1= hf1['sensors/ApplanixDGPS/orientation']
        self.dsetA = hf1.get('sensors/ApplanixDGPS/timestamps')
        self.GPSTimestamp=np.array(self.dsetA)
        hf2 = h5py.File(path_meta+logname+'.h5','r')
        self.dsetB= hf2.get('radar_timestamps')
        self.dsetB=np.array(self.dsetB)
        self.Radar1Timestamp = self.dsetB[:,0]
        
        #Extract all the Latitudes,Longitude and Orientation(Quaternions) 
        self.Long=self.dset[:,0] #Longitude
        self.Lat=self.dset[:,1]  #Latitude
        self.q0=self.dset1[:,0]  #Quaternions 1
        self.q1=self.dset1[:,1]  #Quaternions 2
        self.q2=self.dset1[:,2]  ##Quaternions 3
        self.q3=self.dset1[:,3] ##Quaternions 4
        self.size= len(self.Long) #Size of the dataset ususally len(long)
        self.Radar1Timestamp = np.round(self.Radar1Timestamp,2)#RadarTime stamps 
        self.GPSTimestamp= np.round(self.GPSTimestamp,2)#GPS TIMESTAMPS Converting the last two decimal places to the nearest one.
        nest_asyncio.apply()
        
        
        
    
        
    def Add_Noise(self,Noise_added,Long,Lat):
        #Noise_add=-0.0000150
        n_Long = np.full(Long.size,Noise_added)
        n_Lat =  np.full(Lat.size,Noise_added)
        #New Values of Longitude and Latitude after adding noise 
        Long_n = Long + n_Long
        Lat_n = Lat + n_Lat
        return Long_n,Lat_n


    def quat2eulers(self,q0:float, q1:float, q2:float, q3:float):
    
        #Compute yaw-pitch-roll Euler angles from a quaternion.
    
        #Args
        #q0: Scalar component of quaternion.
        #q1, q2, q3: Vector components of quaternion.
    
        #Returns
        #(roll, pitch, yaw) (tuple): 321 Euler angles in radians
        #roll = math.atan2(2 * ((q2 * q3) + (q0 * q1)),q0**2 - q1**2 - q2**2 + q3**2)  # radians
        #pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
        yaw = math.atan2(2 * ((q1 * q2) + (q0 * q3)),q0**2 + q1**2 - q2**2 - q3**2)
        degree = 180/math.pi
        yaw = yaw *degree
        return yaw

    def CreateAngle(self,q0,q1,q2,q3,size):
        Angle = np.zeros(size,dtype=float)
        for i in range(size):
            Angle[i]=self.quat2eulers(q0[i],q1[i],q2[i],q3[i])
        return Angle

    def BBOX_Generation(self,Lat,Long):
        padding_Lat =0.001750
        Padding_long=0.001750
        bbox = (min(Lat)-padding_Lat,max(Lat)+padding_Lat,min(Long)-Padding_long, max(Long)+Padding_long)
        gr = ox.graph_from_bbox(*bbox, network_type='drive_service')
        geo = ox.geometries_from_bbox(*bbox,tags={'building':True})
        return gr,geo,bbox

    def Convert_To_M(self,Lat,Long):
        X,Y,O,P=utm.from_latlon(Lat, Long)
        X1=X+80#80 meter to match the semseg grid
        Y1=Y+80
        NewLat,NewLong=utm.to_latlon(X1, Y1,O,P)
        diff_Lat = NewLat-Lat
        diff_Long = NewLong-Long
        return diff_Long,diff_Lat
    
    def get_fig(self,bbox=None, size_inches=(15,15)):
        fig, ax = plt.subplots(frameon="False")
        fig.set_size_inches(*size_inches)
        ax.set_facecolor("green")
        plt.close(fig)
        ox.plot.plot_figure_ground(GRoad, node_size=0, network_type='drive_service',ax=ax, edge_color="black", bgcolor='blue', bbox=bbox,default_width=18)
        ox.plot.plot_footprints(GBuilding, ax=ax, bgcolor='green', bbox=bbox)
        return fig
    
    def crop_center(self,pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
    
    def Save_Maps_Sep(self,Lat,Long,DIFF_LAT,DIFF_LONG,i):
        Sep_path_save ='C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS/'
        North = Lat-DIFF_LAT                           
        South = Lat+DIFF_LAT                            
        East  = Long-DIFF_LONG                           
        West  = Long+DIFF_LONG                            
        fig=self.get_fig(bbox=(North,South,East,West),size_inches=(6,6))
        fig.savefig(Sep_path_save+'Save%d.png'%i, bbox_inches='tight', pad_inches = 0)
        
        

    def RotateMap_Sep(self,i):
        Sep_Rdirectory = r'C:\Users\bjqb7h\Downloads\Thesis2022\HD Maps_Rotate\GPSRADAR\RotatedMAPS'
        Sep_path_Rotate = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS'
        os.chdir(Sep_Rdirectory)
        Nimage = cv2.imread(Sep_path_Rotate+ '/Save%d.png' %i)
        Angle = self.CreateAngle(self.q0,self.q1,self.q2,self.q3,self.size)
        imgr = imutils.rotate_bound(Nimage,angle=-180-(Angle[i]))#Read Angle from the H5 file to get the vehicle rotation
        filename ='Rotate%d.png'%i
        cv2.imwrite(filename, imgr) 
    
    def Crop_image_Sep(self,i):
        Crop_directory= r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS'
        Crop_save_path =r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
        os.chdir(Crop_directory)
        im = Image.open(Crop_directory+ '/Rotate%d.png'%i)
        im_new = self.crop_center(im, 160, 160)
        im_new.save(Crop_save_path +'/Cropped%d.png'%i, quality=100)
    

if __name__ == '__main__':
    logpath = r'C:/Users/bjqb7h/Downloads/Thesis2022/GPS DATA/'
    metapath = r'C:/Users/bjqb7h/Downloads/Thesis2022/Meta/'
    Crop_save_path =r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
    log='AtCityBMW_Applanix-20220601T115459Z469'
    hDF5_PATH = 'C:/Users/bjqb7h/Downloads/Thesis2022/NUMPY/New'
    noise1 = 0.0000150
    tt = HDMAPCreator(path_log=logpath,path_meta=metapath,Noise = noise1, logname=log)
    
    Noise_Long,Noise_Lat=tt.Add_Noise(Noise_added=noise1,Long=tt.Long,Lat=tt.Lat) #generating noise in Lat and Long
    #Read Lat and Long and generate BBOX for roads and buildings
    GRoad,GBuilding,BBOX=tt.BBOX_Generation(Noise_Lat,Noise_Long)
    #Create a bbox of 80m
    DIFF_LONG,DIFF_LAT=tt.Convert_To_M(Lat=tt.Lat,Long=tt.Long)
    #Creating Angle of rotation 
    AngleOFROTATIOn = tt.CreateAngle(q0=tt.q0,q1=tt.q1,q2=tt.q2,q3=tt.q3,size=tt.size) #Generating Yaw from the 4 quaternion
    
    for i in trange(tt.GPSTimestamp.size):
        if (tt.GPSTimestamp[i]==tt.Radar1Timestamp[:]).any():
            Noise_Long,Noise_Lat=tt.Add_Noise(Noise_added=noise1,Long=tt.Long,Lat=tt.Lat)
            DIFF_LONG,DIFF_LAT=tt.Convert_To_M(Noise_Lat[i],Noise_Long[i])
            tt.Save_Maps_Sep(Noise_Lat[i],Noise_Long[i],DIFF_LAT,DIFF_LONG,i)
            tt.RotateMap_Sep(i)
            tt.Crop_image_Sep(i)
    #Converting the images to Numpy
    images = [cv2.imread(file) for file in natsorted(glob.glob(Crop_save_path+'/*.png'),alg=ns.REAL)]
    numpydata =asarray(images)
    #Converting the numpy to HDF5
    with h5py.File(hDF5_PATH+'/'+ log+'.hdf5', 'w') as hf:
        grp = hf.create_group(log)
        dataset = grp.create_dataset('Image data', data=numpydata)        






