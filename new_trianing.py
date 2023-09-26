import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from networks import generators as gens
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from sklearn import preprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
import h5py
from tqdm import tqdm
import glob

#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description='Model training with argparse')
parser.add_argument('--gps_folder', type=str, required=True,
                    help='Path to the folder containing the GPS logs')
parser.add_argument('--validation_folder', type=str, required=True,
                    help='Path to the folder containing the validation logs')
parser.add_argument('--semseg_folder', type=str, required=True,
                    help='Path to the folder containing the semseg logs')
parser.add_argument('--labels_folder', type=str, required=True,
                    help='Path to the folder containing the labels')
parser.add_argument('--model_save_path', type=str, required=True,
                    help='Path to save the trained model')
parser.add_argument('--losses_save_path', type=str, required=True,
                    help='Path to the folder to save training and validation loss values')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate for training')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of epochs for training')

args = parser.parse_args()

# Extract values from arguments
gps_logs_folder = args.gps_folder
validation_folder = args.validation_folder
semseg_logs_folder = args.semseg_folder
labels_folder = args.labels_folder
Model_save_path = args.model_save_path
Losses_save_path = args.losses_save_path
lr = args.lr
num_epochs = args.num_epochs

class MR_TRUS_4D(Dataset):
    def __init__(self, gps_folder, semseg_folder, labels_folder):
        self.gps_files = glob.glob(os.path.join(gps_folder, '*.hdf5'))
        self.semseg_files = semseg_folder
        self.labels_files = glob.glob(os.path.join(labels_folder, '*.npy'))
        self.num_logs = len(self.gps_files)
        print("TOTAL NO. OF FILES IN THE TRAINING FOLDER ",self.num_logs)


    def __len__(self):
        total_samples = 0
        for gps_file in self.gps_files:
            with h5py.File(gps_file, 'r') as gps_data:
                base_log = os.path.basename(gps_file)
                log_name = base_log.split('GPS')[0]
                num_samples = gps_data[log_name]['Image data'].shape[0]
                total_samples += num_samples
        return total_samples

    def __getitem__(self, idx):
        # Calculate the log index and the index within the log
        log_idx = 0
        for f in self.gps_files:
            with h5py.File(f, 'r') as gps_file:
                base_log = os.path.basename(f)
                log_name = base_log.split('GPS')[0]
                num_samples = gps_file[log_name]['Image data'].shape[0]
            if idx < num_samples:
                break
            idx -= num_samples
            log_idx += 1

        # Load the GPS and Semseg data for the current log
        with h5py.File(self.gps_files[log_idx], 'r') as gps_file:
            base_log = os.path.basename(self.gps_files[log_idx])
            log_name = base_log.split('GPS')[0]
            gps_data = gps_file[log_name]['Image data'][:, :, :, 1]
            print("CURRENT LOG NAME ",log_name)




        semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
        with h5py.File(semseg_file, 'r') as semseg_file:
            sem_data = semseg_file['grid_prediction'][()]

        A=len(gps_data)
        sem_data = sem_data[:A]
        sem_data = np.reshape(sem_data, (A, 160, 160, 3))
        sem_data = sem_data[:,:,:,2]
        # Assuming gps_data and sem_norm have shapes (A, 160, 160)
        # Flatten the data for normalization
        # Flatten the data to compute min and max values from the entire dataset
        flattened_gps_data = gps_data.flatten()
        flattened_sem_data = sem_data.flatten()

        # Compute the min and max values from the entire dataset
        global_min_gps = np.min(flattened_gps_data)
        global_max_gps = np.max(flattened_gps_data)
        global_min_sem = np.min(flattened_sem_data)
        global_max_sem = np.max(flattened_sem_data)


        # Normalize GPS data
        normalized_gps_data = (gps_data - global_min_gps) / (global_max_gps - global_min_gps)

        # Normalize Semseg data
        normalized_sem_data = (sem_data - global_min_sem) / (global_max_sem - global_min_sem)

        # print(normalized_gps_data.max(), normalized_gps_data.min())
        # print(normalized_sem_data.max(), normalized_sem_data.min())

        # Merge GPS and Semseg images into a single feature with two channels
        feat_shape = tuple(list(gps_data.shape) + [2])
        features = np.zeros(feat_shape)
        features[:, :, :, 0] = normalized_gps_data
        features[:, :, :, 1] = normalized_sem_data

        # Load the labels for the current log
        label_file = os.path.join(labels_folder, (log_name+'.npy'))
        labels_data = np.load(label_file)

        features_sample = features[idx]
        labels_sample = labels_data[idx]


        # Convert features and label to torch tensors
        features_sample = torch.from_numpy(features_sample)
        labels_sample = torch.from_numpy(labels_sample)


        return features_sample, labels_sample

class ValD(Dataset):
    def __init__(self, validation_folder, semseg_folder, labels_folder):
        self.val_files = glob.glob(os.path.join(validation_folder, '*.hdf5'))
        self.semseg_files = semseg_folder
        self.labels_files = glob.glob(os.path.join(labels_folder, '*.npy'))
        self.num_logs = len(self.val_files)

    def __len__(self):
        total_samples = 0
        for val_file in self.val_files:
            with h5py.File(val_file, 'r') as val_data:
                base_log = os.path.basename(val_file)
                log_name = base_log.split('GPS')[0]
                num_samples = val_data[log_name]['Image data'].shape[0]
                total_samples += num_samples
        return total_samples

    def __getitem__(self, idx):
        # Calculate the log index and the index within the log
        log_idx = 0
        for f in self.val_files:
            with h5py.File(f, 'r') as val_file:
                base_log = os.path.basename(f)
                log_name = base_log.split('GPS')[0]
                num_samples = val_file[log_name]['Image data'].shape[0]
            if idx < num_samples:
                break
            idx -= num_samples
            log_idx += 1

        # Load the GPS and Semseg data for the current log
        with h5py.File(self.val_files[log_idx], 'r') as val_file:
            base_log = os.path.basename(self.val_files[log_idx])
            log_name = base_log.split('GPS')[0]
            val_data = val_file[log_name]['Image data'][:, :, :, 1]
            min_value = np.min(val_data)
            max_value = np.max(val_data)

            # Next, normalize the GPS data array to the range [0, 1]
            normalized_val_data = (val_data - min_value) / (max_value - min_value)

        semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
        with h5py.File(semseg_file, 'r') as semseg_file:
            sem_data = semseg_file['grid_prediction'][()]

        A = len(val_data)
        sem_data = sem_data[:A]
        sem_data = np.reshape(sem_data, (A, 160, 160, 3))
        sem_data = sem_data[:, :, :, 2]
        min_value_sem = np.min(sem_data)
        max_value_sem = np.max(sem_data)

        # Next, normalize the GPS data array to the range [0, 1]
        normalized_sem_data = (sem_data - min_value_sem) / (max_value_sem - min_value_sem)

        # Merge GPS and Semseg images into a single feature with two channels
        feat_shape = tuple(list(val_data.shape) + [2])
        features = np.zeros(feat_shape)
        features[:, :, :, 0] = normalized_val_data
        features[:, :, :, 1] = normalized_sem_data

        # Load the labels for the current log
        label_file = os.path.join(labels_folder, (log_name + '.npy'))
        labels_data = np.load(label_file)

        # Select the corresponding sample from the log
        features_sample = features[idx]
        labels_sample = labels_data[idx]

        # Convert features and label to torch tensors
        features_sample = torch.from_numpy(features_sample)
        labels_sample = torch.from_numpy(labels_sample)


        return features_sample, labels_sample


# gps_logs_folder = r'C:\Users\bjqb7h\Downloads\Thesis2022\NUMPY\New\New Logs\tr'  # Specify the path to the folder containing the GPS logs
# validation_folder =r'C:\Users\bjqb7h\Downloads\Thesis2022\NUMPY\New\New Logs\vl'
# test_folder =r'C:\Users\bjqb7h\Downloads\Thesis2022\NUMPY\New\New Logs\tt'
# Model_save_path="C:/Users/bjqb7h/Downloads/Thesis2022/Results/Model save/"
# Losses_save_path ="C:/Users/bjqb7h/Downloads/Thesis2022/Results/Losses"

# semseg_logs_folder = r'C:\Users\bjqb7h\Downloads\Thesis2022\Semantic Segmentaiton Logs'  # Specify the path to the folder containing the semseg logs
# labels_folder = r'C:\Users\bjqb7h\Downloads\Thesis2022\GT TRANSLATIONS\SYNC TARGET'  # Specify the path to the folder containing the labels

dataset_T = MR_TRUS_4D(gps_logs_folder, semseg_logs_folder, labels_folder)
dataset_V = ValD(validation_folder, semseg_logs_folder, labels_folder)


# train_sampler = SubsetRandomSampler(train_idx)
# val_sampler = SubsetRandomSampler(val_idx)
# test_sampler = SubsetRandomSampler(test_idx)

train_dataloader = DataLoader(dataset_T, batch_size=32, shuffle=False)
val_dataloader = DataLoader(dataset_V, batch_size=32, shuffle=False)
#test_dataloader = DataLoader(dataset_T, batch_size=16, shuffle=False)
loss = nn.MSELoss()
lr = 0.001

model = gens.AttentionReg()
# Load the previously trained model state dictionary
#saved_model_path = r"C:/Users/bjqb7h/Downloads/Thesis2022/Results/Model save/New Model 2.pth"
#model.load_state_dict(torch.load(saved_model_path))

# Set the model to training mode
model.train()
train_losses = []
val_losses = []
num_epochs = 10
print('Learning rate = {}'.format(lr))


optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,7], gamma=0.3)

print('Starting model training...')
for epoch in range(num_epochs):
    start = time.time()
    print(start)
    # Print the current learning rate
    train_loss_step = list()
    with tqdm(total=len(train_dataloader), unit="batch") as pbar:  # tqdm progress bar
        for batch_idx, (x, y) in enumerate(train_dataloader):

            x = torch.reshape(x, (2, -1, 160, 160, 1))
            x = x.double()
            y = y.double()
            model = model.double()
            optimizer.zero_grad()
            y_hat = model(x)
            train_loss = loss(y, y_hat)
            train_loss.backward()
            optimizer.step()
            train_loss_step.append(train_loss.detach())
            train_losses.append(np.mean(train_loss_step))
            print(train_loss)
            pbar.set_postfix({"Train Loss": np.mean(train_loss_step)})
            pbar.update()
    if epoch % 2 == 0:
        val_loss_step = list()
        with torch.no_grad():
            with tqdm(total=len(val_dataloader), unit="batch") as pbar:
                for batch_idx, (val_x, val_y) in enumerate(val_dataloader):
                    val_x = torch.reshape(val_x, (2, -1, 160, 160, 1))
                    val_x = val_x.double()
                    val_y = val_y.double()
                    val_y_hat = model(val_x)
                    # Update the learning rate based on the validation loss
                    val_loss = loss(val_y, val_y_hat)
                    val_loss_step.append(val_loss)
                    val_losses.append(np.mean(val_loss_step))
                    pbar.set_postfix({"Validation Loss": np.mean(val_loss_step)})
                    pbar.update()
        stop = time.time()
        print(f'*' * 100)
        print(f'Epoch:{epoch}\t Time:{(stop-start)/60}minutes\t Train loss:{np.mean(train_loss_step)}\t Validation loss:{np.mean(val_loss_step)}')
print('Training complete')
# Save the trained model
model_filename = "New Model 4.pth"
save_path = os.path.join(Model_save_path, model_filename)
torch.save(model.state_dict(), save_path)


# Convert the lists to NumPy arrays
train_losses_array = np.array(train_losses)
val_losses_array = np.array(val_losses)

# Save the NumPy arrays as .npy files
np.save('train_losses_model4.npy', train_losses_array)
np.save('val_losses_model4.npy', val_losses_array)
