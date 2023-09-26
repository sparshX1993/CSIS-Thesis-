
from utils import data_loading_funcs as load_func

import numpy as np
import os
import h5py


base_mat = np.eye(4)  # Base matrix

logpath = r'C:\Users\bjqb7h\Downloads\Thesis2022\GT TRANSLATIONS\TRUE NOISE'  # Path to the folder containing the dx and dy files
output_dir = r'C:\Users\bjqb7h\Downloads\Thesis2022\GT TRANSLATIONS\SYNC TARGET'  # Path to the directory where targets will be saved
path_meta = r'C:\Users\bjqb7h\Downloads\Thesis2022\Meta\New Meta'
data_path = r'C:\Users\bjqb7h\Downloads\Thesis2022\GPS DATA\GPS DATA'


def load_dx_dy(logpath, logname):
    dx_filename = f"DXGTNEW_{logname}.npy"
    dy_filename = f"DYGTNEW_{logname}.npy"

    dx_file = os.path.join(logpath, dx_filename)
    dy_file = os.path.join(logpath, dy_filename)

    dx = np.load(dx_file,'r')
    dy = np.load(dy_file,'r')
    return dx, dy

def sync_radar_dgps(logname,path_meta,data_path,dx,dy):
    logname=logname.replace(" ", "")
    print(logname)
    path_timestamps = 'sensors/ApplanixDGPS/timestamps'
    #hf1 = h5py.File(data_path + '\\'+logname + '.h5', 'r')
    hf1 = h5py.File(os.path.join(data_path, logname + '.h5'), 'r')
    dsetA = hf1.get(path_timestamps)
    GPSTimestamp = np.array(dsetA)

    hf2 = h5py.File(path_meta + '\\'+logname + '.h5', 'r')
    dsetB = hf2.get('radar_timestamps')
    dsetB = np.array(dsetB)
    Radar1Timestamp = dsetB[:, 0]

    Radar1Timestamp = np.round(Radar1Timestamp, 2)  # RadarTime stamps
    GPSTimestamp = np.round(GPSTimestamp, 2)  # GPS


    dgps_sync = []
    for i in range(len(GPSTimestamp)):
        # Check if a match is found with any radar timestamp
        if (GPSTimestamp[i] == Radar1Timestamp[:]).any():
            # Append index to dgps_sync array
            dgps_sync.append(i)

    dx=dx[dgps_sync]
    dy=dy[dgps_sync]

    return dx,dy

def generate_target(dx, dy):
    target_val = np.zeros((len(dx), 6), dtype=np.float32)
    gt_mats = []  # List to store target matrices for each input
    for i in range(len(dx)):
        gt_mat = np.eye(4)
        gt_mat[0, 3] = dx[i] / 0.000014
        gt_mat[1, 3] = dy[i] / 0.000017
        mat_diff = gt_mat.dot(np.linalg.inv(base_mat))
        target_val[i, :] = load_func.decompose_matrix_degree(mat_diff)

    target_val=target_val[:,:2]
    return target_val

# Iterate over files in the folder
for filename in os.listdir(logpath):
    if filename.endswith('.npy'):
        logname = os.path.splitext(filename)[0]

        if logname.startswith('DXGTNEW_'):
            logname = logname[len('DXGTNEW_'):]
        elif logname.startswith('DYGTNEW_'):
            logname = logname[len('DYGTNEW_'):]
            print("dy log encountered endng program")
            break  # End the program if the log name starts with DYGTNEW_

        dx, dy = load_dx_dy(logpath, logname)
        print("Length of dx and dy before sync", len(dx))

        dx,dy = sync_radar_dgps(logname,path_meta,data_path,dx,dy)
        print("Length of dx and dy after sync",len(dx))
        target = generate_target(dx, dy)
        print(target.shape)
        print(target)



        # Create the output file path for the current log
        output_file = os.path.join(output_dir, logname + '.npy')

        # Save the target array as a NumPy binary file
        np.save(output_file, target)

        print(f"Target saved for log {logname}: {output_file}")
