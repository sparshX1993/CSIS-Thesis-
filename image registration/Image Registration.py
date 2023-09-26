import numpy as np
from tqdm.notebook import trange, tqdm
from numpy import asarray
from scipy.ndimage.interpolation import shift
from scipy.optimize import differential_evolution
from sklearn import preprocessing
import scipy.ndimage as snd
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
np.set_printoptions(precision=5,suppress=True)

path_Semseg = 'C:/Users/bjqb7h/Downloads/Thesis2022/SEMSEGGPS'
log  = '/AtCityBMW_Applanix-20220601T114937Z033'
semsegpath ='C:/Users/bjqb7h/Downloads/Thesis2022/semsegimage'
path_GPS = 'C:/Users/bjqb7h/Downloads/Thesis2022/NUMPY/New/'
log_GPS  = 'AtCityBMW_Applanix-20220601T115459Z469GPSWNSYAW51'
log_DGPS = 'AtCityBMW_Applanix-20220601T115459Z469DGPSWNS3'

h5f = h5py.File(path_Semseg +log+'.h5','r')
dset = h5f.get('grid_prediction')

GPS = h5py.File(path_GPS+log_GPS+'.hdf5','r')
GPSDSET = GPS.get(log+'/Image data')

#Importing DGPS MAPS
h5fDGPS = h5py.File(path_GPS+log_DGPS+'.hdf5','r')
dset5 = h5fDGPS.get(log+'/Image data')

A=len(dset5)
SemsegData = np.reshape(dset,(A,160,160,2))

print(GPSDSET.shape)

# NewSemseg= h5py.File(NewSemseg+log+'Semseg'+'.hdf5','r')
# NSem= NewSemseg.get(log+'/Image data')
# print(NSem.shape)


BOUNDS = [(-8.5,8.5), (-8.5,8.5)]  # Bounds (in pixels) supported by mutual information based correlator


def __mutual_information(ref_image_crop, cmp_image, bins=256, normed=False):
    """
    :param ref_image_crop: ndarray, cropped image from the center of reference image, needs to be same size as `cmp_image`
    :param cmp_image: ndarray, comparison image data data
    :param bins: number of histogram bins
    :param normed: return normalized mutual information
    :return: mutual information values
    """
    #use linspace instead of bins
    joint_hist,_,_ = np.histogram2d(ref_image_crop.flatten(), cmp_image.flatten(), bins=bins, range=[(0,1),(0,1)])
    ref_hist = joint_hist.sum(axis=1)
    cmp_hist = joint_hist.sum(axis=0)
    joint_ent = __entropy(joint_hist)
    ref_ent = __entropy(ref_hist)
    cmp_ent = __entropy(cmp_hist)
    mutual_info = ref_ent + cmp_ent - joint_ent
    if normed:
        mutual_info = mutual_info / np.sqrt(ref_ent * cmp_ent)
    return mutual_info

def obj_func(dx_dy, ref_image, cmp_image):
    shifted_img = shift(ref_image, dx_dy)
    return -__mutual_information(shifted_img, cmp_image)


def main_mi_reg(ref_image, cmp_image,bounds=BOUNDS):
    """
    Correlator based onMutual Information Algorithm
    :param ref_image: ndarray, containing reference image data
    :param cmp_image: ndarray, containing comparison image data
    :param bounds: sequence, bounds paramater in scipy.optimize.differential_evolution
    :return: (residual in X, residual in Y, match height)
    """
    opt_res = differential_evolution(obj_func, bounds, args=(ref_image, cmp_image), init="latinhypercube", seed=123)
    (dx, dy), match_height = -opt_res.x, -opt_res.fun
    return dx, dy, match_height





def __entropy(img_hist):
    """
    :param img_hist: Array containing image histogram
    :return: image entropy
    """
    img_hist = img_hist / float(np.sum(img_hist))
    img_hist = img_hist[np.nonzero(img_hist)]
    return -np.sum(img_hist * np.log2(img_hist))


def mi_reg_sa(ref_image, cmp_image, bounds=BOUNDS):
    """
    Using Signal annealing instead of differential evolution to find the minima
    :param ref_image: ndarray, containing reference image data
    :param cmp_image: ndarray, containing comparison image data
    :param bounds: sequence, bounds paramater in scipy.optimize.differential_evolution
    :return: (residual in X, residual in Y, match height)
    """
    obj_func = lambda dx_dy: -__mutual_information(shift(ref_image, dx_dy), cmp_image)
    results = dual_annealing(obj_func,bounds,seed=123,maxiter=50)
    (dx_sa, dy_sa), match_height_sa = -results.x,- results.fun
    return dx_sa, dy_sa, match_height_sa

def rmse(gt, pred):
    # Calculate the squared error between the two images
    squared_error = (gt - pred)**2
    # Sum the squared error over all pixels
    sum_squared_error = np.sum(squared_error)
    # Calculate the number of pixels
    num_pixels = gt.shape[0] * gt.shape[1]
    # Calculate the mean squared error
    mse = sum_squared_error / num_pixels
    # Calculate the root mean squared error
    rmse = np.sqrt(mse)
    return rmse

def fancy_xy_trans_slice(img_slice, x_y_trans):
    """ Return copy of `img_slice` translated by `x_y_trans` voxels
    Parameters
     ----------
     img_slice : array shape (M, N)
         2D image to transform with translation `x_vox_trans`
     x_y_trans : float
         Number of pixels (voxels) to translate `img_slice`; can be
         positive or negative, and does not need to be integer value.
     """
    # Resample image using bilinear interpolation (order=1)
    x_y_trans = np.array(x_y_trans)
    trans_slice = snd.affine_transform(img_slice, [1, 1],  -x_y_trans, order=1)
    return trans_slice


dy = np.zeros((A,1),dtype =np.float64)
dx = np.zeros((A,1),dtype =np.float64)
# t1_set_norm= np.zeros((A,160,160),dtype =np.float64)
# t2_set_norm= np.zeros((A,160,160),dtype =np.float64)
# t3_set_norm= np.zeros((A,160,160),dtype =np.float64)
# unshifted= np.zeros((A,160,160),dtype =np.float64)
# semseg_merged = np.max(SemsegData, axis=-1)

# GPS_data = np.flip(GPSDSET[:, :, :, 1], axis=1)
# Semseg_data = np.rot90(semseg_merged[:, :, :], 2)
# Semseg_data = asarray(Semseg_data, dtype=np.float64)
# DGPS_data = np.flip(dset5[:, :, :, 1], axis=1)

dy1 = np.zeros((A,1),dtype =np.float64)
dx1 = np.zeros((A,1),dtype =np.float64)

for T in trange(A):
    t1_slice = GPSDSET[T, :, :, 1]
    t1_slice = np.flip(t1_slice, axis=1)
    GPS_data = t1_slice

    #NewSemseg = np.rot90(NSem[T, :, :, 0], 2)

    t3_slice = dset5[T, :, :, 1]
    t3_slice = np.flip(t3_slice, axis=1)
    DGPS_data = t3_slice

    t1_set_norm = preprocessing.normalize((GPS_data))
    #t2_set_norm = preprocessing.normalize((NewSemseg))
    t3_set_norm = preprocessing.normalize((DGPS_data))

    dx, dy, MI = main_mi_reg(t1_set_norm, t3_set_norm)
    dx1[T, :] = dx
    dy1[T, :] = dy





file_names = [f"dx_bounds_{BOUNDS[0][0]}_{BOUNDS[0][1]}_"
              f"{BOUNDS[1][0]}_{BOUNDS[1][1]}.npy",
              f"dy_bounds_{BOUNDS[0][0]}_{BOUNDS[0][1]}_"
              f"{BOUNDS[1][0]}_{BOUNDS[1][1]}.npy"]

np.save(r'C:\Users\bjqb7h\Downloads\Thesis2022\dxdy\\'+file_names[0],dx)
np.save(r'C:\Users\bjqb7h\Downloads\Thesis2022\dxdy\\'+file_names[1],dy)