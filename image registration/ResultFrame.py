


import numpy as np 

import random

from numpy import asarray
from scipy.ndimage import shift
from sklearn import preprocessing
import h5py
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['image.interpolation'] = 'nearest'
np.set_printoptions(precision=5,suppress=True)


path_Semseg = 'C:/Users/bjqb7h/Downloads/Thesis2022/SEMSEGGPS'
log  = '/AtCityBMW_Applanix-20220601T115459Z469'
semsegpath ='C:/Users/bjqb7h/Downloads/Thesis2022/semsegimage'
path_GPS = 'C:/Users/bjqb7h/Downloads/Thesis2022/NUMPY/New/'
log_GPS  = 'AtCityBMW_Applanix-20220601T115459Z469GPSWNSYAW505'
log_DGPS = 'AtCityBMW_Applanix-20220601T115459Z469DGPSWNSYAW101'
NewSemseg ='C:/Users/bjqb7h/Downloads/Thesis2022/Semseg Images'


#Semseg Data
h5f = h5py.File(path_Semseg +log+'.h5','r')
dset = h5f.get('grid_prediction')
print(dset.shape)

#GPS Data
h5fGPS = h5py.File(path_GPS+log_GPS+'.hdf5','r')
dset2 = h5fGPS.get(log+'/Image data')

# DGPS Data
h5fDGPS = h5py.File(path_GPS+log_DGPS+'.hdf5','r')
dset5 = h5fDGPS.get(log+'/Image data')

NewSemseg= h5py.File(NewSemseg+log+'SemsegV6'+'.hdf5','r')
NSem= NewSemseg.get(log+'/Image data')

#Reshaping Semseg Grid prediction
A=len(dset5)
SemsegData = np.reshape(dset,(A,160,160,2))



T = random.randint(0,len(dset5))




#USING GPS HDF5 dATASET
t1_slice = dset2[T,:,:,1]
t1_set = np.flip(dset2[:,:,:,1],axis=1)
t1_slice = np.flip(t1_slice, axis=1)
GPS_data = t1_slice


NewSemseg = np.rot90(SemsegData[T,:,:,1],2)
# NewSemseg=dset[T,:,:]
t2_slice = NewSemseg[:,:]
t2_slice = asarray(t2_slice,dtype=np.float64)
Semseg_data = t2_slice

# #DGPS values
t3_slice = dset5[T,:,:,1]
t3_slice = np.flip(t3_slice, axis=1)
DGPS_data = t3_slice

#
# GPS_data= GPS_data/GPS_data.max()
Semseg_data = Semseg_data/Semseg_data.max()
# DGPS_data= DGPS_data/DGPS_data.max()
RMSEGPS = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\RMSE\\RMSEGPS4.npy')
RMSETGPS = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\RMSE\\RMSETRANS4.npy')

dx_gt = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\GT TRANSLATIONS\\dxT.npy')
dy_gt = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\GT TRANSLATIONS\\dyT.npy')
dx_gt=dx_gt/0.0000045
dy_gt=dy_gt/0.0000050




t1_set_norm = preprocessing.normalize((GPS_data))
t2_set_norm = preprocessing.normalize((Semseg_data))
t3_set_norm = preprocessing.normalize((DGPS_data))




matplotlib.use('TkAgg')


dx = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\dxdy\\DX_NMI22_dx_bounds_-12_12_-12_12.npy')
dy = np.load(f'C:\\Users\\bjqb7h\\Downloads\\Thesis2022\\dxdy\\DY_NMI22_dy_bounds_-12_12_-12_12.npy')



unshifted = shift(t1_set_norm, [dx[T],dy[T]],order=1)

if T>= 1846:
    unshiftedTRUE = shift(t1_set_norm, [-dx_gt[T],dy_gt[T]], order=1)
    print("Frame is more than 1846")
else:
    unshiftedTRUE = shift(t1_set_norm, [dx_gt[T],-dy_gt[T]], order=1)
    print("Frame is less than 1846")

# unshiftedTRUE = shift(t1_set_norm, [dx_gt[T],dy_gt[T]], order=1)




t4_set_norm = preprocessing.normalize((unshifted))

print(T)


# Create a figure with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2)

# Plot the first image in the first row and first column
axs[0, 0].imshow(t3_set_norm,cmap="gray")
axs[0, 0].text(-10, -40, f'True DX={round(dx_gt[T],4)}, True DY ={round(dy_gt[T],4)}', color='blue')
axs[0, 0].set_title('DGPS Map')
print(dx_gt[T],dy_gt[T])
print(RMSEGPS[T],RMSETGPS[T])
# Plot the second image in the first row and second column
axs[0, 1].imshow(t4_set_norm,cmap="gray")
axs[0, 1].text(-10, -40, f'dx={dx[T]}, dy={dy[T]}', color='red')
axs[0, 1].set_title('Translated Map')

# Plot the third image in the second row and first column
axs[1, 0].imshow(t1_set_norm,cmap='gray')
axs[1, 0].set_title('GPS Map')

# Plot the fourth image in the second row and second column
axs[1, 1].imshow(unshiftedTRUE,cmap='gray')
axs[1, 1].text(-60, -40, f'RMSE TGPS={round(RMSETGPS[T],4)}, RMSE GPS={round(RMSEGPS[T],4)}', color='blue')
axs[1, 1].set_title('GPS MAP with TRUE DX AND DY')

# # Plot the fourth image in the second row and second column
# axs[1, 1].imshow(unshiftedTRUE,cmap='gray')
# axs[1, 1].text(-60, -40, f'RMSE TGPS={round(RMSETGPS[T],4)}, RMSE GPS={round(RMSEGPS[T],4)}', color='blue')
# axs[1, 1].set_title('Translated MAP w TRUE DX AND DY')


# Remove empty subplot in the third row and second column
# axs[2, 1].remove()

plt.tight_layout()

# Adjust the spacing between the subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)
# Save the image in a specific location
#plt.savefig(r'C:\Users\bjqb7h\Downloads\Thesis2022\Results\AtCityBMW_Applanix-20220601T115459Z469\After Calibration\After New SemSeg\Result%d.png'%T)
# Show the figure
plt.show()


