import torch
import shutil
import torch.cuda
import argparse
import torch.nn as nn
import psutil
import datetime
import torch.optim as optim
from networks import generators as gens
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.profiler import profile, record_function, ProfilerActivity
import torch.optim.lr_scheduler as lr_scheduler
from sklearn import preprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
import h5py
from tqdm import tqdm
import glob
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_logs_without_suffix(folder_path, suffix):
    logs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(suffix):
            logs.append(filename[:-len(suffix)])
    return logs

def verify_logs_presence(logs_to_verify, semantic_logs):
    missing_logs = []
    for log in logs_to_verify:
        if log not in semantic_logs:
            missing_logs.append(log)
    return missing_logs

def normalize_data(flattened_sem_data,flattened_gps_data,gps_data,sem_data):
    # Compute the min and max values from the entire dataset
    global_min_gps = np.min(flattened_gps_data)
    global_max_gps = np.max(flattened_gps_data)
    global_min_sem = np.min(flattened_sem_data)
    global_max_sem = np.max(flattened_sem_data)

    # Normalize GPS data
    normalized_gps_data = (gps_data - global_min_gps) / (global_max_gps - global_min_gps)

    # Normalize Semseg data
    normalized_sem_data = (sem_data - global_min_sem) / (global_max_sem - global_min_sem)

    return normalized_gps_data,normalized_sem_data

class MR_TRUS_4D(Dataset):
    def __init__(self, gps_folder, semseg_folder, labels_folder):
        self.gps_files = glob.glob(os.path.join(gps_folder, '*.hdf5'))
        self.semseg_files = semseg_folder   
        self.labels_folder = labels_folder
        self.num_logs = len(self.gps_files)
        print("TOTAL GPS FILES ", self.num_logs)
        
    def __len__(self):
        total_samples = 0
        for gps_file in self.gps_files:
            with h5py.File(gps_file, 'r') as gps_data:
                base_log = os.path.basename(gps_file)
                log_name = base_log.split('GPS')[0]
                num_samples_gps = gps_data[log_name]['Image data'].shape[0]
                # Get the corresponding Semseg log file
                semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
                with h5py.File(semseg_file, 'r') as semseg_file_data:
                    num_samples_sem = semseg_file_data['grid_prediction'].shape[0]
                # Choose the minimum between GPS and Semseg log lengths
                num_samples = min(num_samples_gps, num_samples_sem)
                total_samples += num_samples
        return total_samples

    def __getitem__(self, idx):
        log_idx = 0
        for f in self.gps_files:
            with h5py.File(f, 'r') as gps_file:
                base_log = os.path.basename(f)
                log_name = base_log.split('GPS')[0]
                num_samples_gps = gps_file[log_name]['Image data'].shape[0]
                semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
                with h5py.File(semseg_file, 'r') as semseg_file_data:
                    num_samples_sem = semseg_file_data['grid_prediction'].shape[0]
                num_samples = min(num_samples_gps, num_samples_sem)
            if idx < num_samples:
                break
            idx -= num_samples
            log_idx += 1

        # Load the GPS and Semseg data for the current log
        with h5py.File(self.gps_files[log_idx], 'r') as gps_file:
            base_log = os.path.basename(self.gps_files[log_idx])
            log_name = base_log.split('GPS')[0]
            gps_data = gps_file[log_name]['Image data'][:, :, :, 1]
            
        semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
        with h5py.File(semseg_file, 'r') as semseg_file:
            sem_data = semseg_file['grid_prediction'][()]

        A = len(gps_data)
        B = len(sem_data)

        if B < A:
            gps_data = gps_data[:B]
            A = B

        sem_data = sem_data[:A]
        sem_data = np.reshape(sem_data, (A, 160, 160, 3))
        sem_data = sem_data[:, :, :, 1]
        
        flattened_gps_data = gps_data.flatten()
        flattened_sem_data = sem_data.flatten()

        normalized_gps_data,normalized_sem_data=normalize_data(flattened_gps_data,flattened_sem_data,gps_data,sem_data)

        #Merge GPS and Semseg images into a single feature with two channels
        feat_shape = tuple(list(gps_data.shape) + [2])
        features = np.zeros(feat_shape)
        features[:, :, :, 0] = normalized_gps_data
        features[:, :, :, 1] = normalized_sem_data

        # Load the labels for the current log
        label_file = os.path.join(self.labels_folder, (log_name + '.npy'))
        labels_data = np.load(label_file)

        features_sample = features[idx]
        labels_sample = labels_data[idx]

        features_sample = torch.from_numpy(features_sample)
        labels_sample = torch.from_numpy(labels_sample)

        return features_sample, labels_sample

class ValD(Dataset):
    def __init__(self, validation_folder, semseg_folder, labels_folder):
        self.val_files = glob.glob(os.path.join(validation_folder, '*.hdf5'))
        self.semseg_files = semseg_folder
        self.labels_folder = labels_folder
        self.num_logs = len(self.val_files)
        print("TOTAL VAL FILES ",self.num_logs)

    def __len__(self):
        total_samples = 0
        for val_file in self.val_files:
            with h5py.File(val_file, 'r') as val_data:
                base_log = os.path.basename(val_file)
                log_name = base_log.split('GPS')[0]
                num_samples_val = val_data[log_name]['Image data'].shape[0]
                # Get the corresponding Semseg log file
                semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
                with h5py.File(semseg_file, 'r') as semseg_file_data:
                    num_samples_sem = semseg_file_data['grid_prediction'].shape[0]
                # Choose the minimum between GPS and Semseg log lengths
                num_samples = min(num_samples_val, num_samples_sem)
                total_samples += num_samples
        return total_samples

    def __getitem__(self, idx):
        log_idx = 0
        for f in self.val_files:
            with h5py.File(f, 'r') as val_file:
                base_log = os.path.basename(f)
                log_name = base_log.split('GPS')[0]
                num_samples_val = val_file[log_name]['Image data'].shape[0]
                semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
                with h5py.File(semseg_file, 'r') as semseg_file_data:
                    num_samples_sem = semseg_file_data['grid_prediction'].shape[0]
                num_samples = min(num_samples_val, num_samples_sem)
            if idx < num_samples:
                break
            idx -= num_samples
            log_idx += 1

        # Load the GPS and Semseg data for the current log
        with h5py.File(self.val_files[log_idx], 'r') as val_file:
            base_log = os.path.basename(self.val_files[log_idx])
            log_name = base_log.split('GPS')[0]
            val_data = val_file[log_name]['Image data'][:, :, :, 1]
            
        semseg_file = os.path.join(self.semseg_files, f'{log_name}.h5')
        with h5py.File(semseg_file, 'r') as semseg_file:
            sem_data = semseg_file['grid_prediction'][()]

        A = len(val_data)
        B = len(sem_data)

        if B < A:
            val_data = val_data[:B]
            A = B

        sem_data = sem_data[:A]
        sem_data = np.reshape(sem_data, (A, 160, 160, 3))
        sem_data = sem_data[:, :, :, 2]
        
        flattened_val_data = val_data.flatten()
        flattened_sem_data = sem_data.flatten()

        normalized_val_data,normalized_sem_data=normalize_data(flattened_val_data,flattened_sem_data,val_data,sem_data)

        #Merge GPS and Semseg images into a single feature with two channels
        feat_shape = tuple(list(val_data.shape) + [2])
        features = np.zeros(feat_shape)
        features[:, :, :, 0] = normalized_val_data
        features[:, :, :, 1] = normalized_sem_data

        # Load the labels for the current log
        label_file = os.path.join(self.labels_folder, (log_name + '.npy'))
        labels_data = np.load(label_file)

        features_sample = features[idx]
        labels_sample = labels_data[idx]

        features_sample = torch.from_numpy(features_sample)
        labels_sample = torch.from_numpy(labels_sample)

        return features_sample, labels_sample


def main(gps_folder, validation_folder, semseg_folder, labels_folder, model_save_path, losses_save_path, lr, num_epochs):

    training_folder = gps_folder
    validation_folder = validation_folder
    semantic_folder = semseg_folder
    suffix = "GPSNOISE1.hdf5"

    training_logs = get_logs_without_suffix(training_folder, suffix)
    validation_logs = get_logs_without_suffix(validation_folder, suffix)
    semantic_logs = get_logs_without_suffix(semantic_folder, ".h5")

    missing_training_logs = verify_logs_presence(training_logs, semantic_logs)
    missing_validation_logs = verify_logs_presence(validation_logs, semantic_logs)

    if missing_training_logs or missing_validation_logs:
        error_message = "Error: Some logs are missing!\n"
        if missing_training_logs:
            error_message += "Missing logs in training folder: {}\n".format(missing_training_logs)
        if missing_validation_logs:
            error_message += "Missing logs in validation folder: {}".format(missing_validation_logs)
        raise ValueError(error_message)
    else:
        print("All logs in training and validation folders are present in the semantic folder.")

    dataset_T = MR_TRUS_4D(gps_folder, semseg_folder, labels_folder)
    dataset_V = ValD(validation_folder, semseg_folder, labels_folder)
    train_dataloader = DataLoader(dataset_T, batch_size=8, shuffle=False,num_workers=5,pin_memory= True)
    val_dataloader = DataLoader(dataset_V, batch_size=8, shuffle=False,num_workers=5,pin_memory= True)
    
    
    torch.cuda.synchronize()
    print("the cuda device is ",device)
    
    loss = nn.MSELoss()
    model1 = gens.AttentionReg()
    # Calculate the total number of parameters
    #total_params = sum(p.numel() for p in model1.parameters())

    #print(f"Total number of parameters: {total_params}")
    # Load the trained model state dictionary
    #saved_model_path = "/private_shared/ModelSAVE/New_Model_777.pth"
    #model1.load_state_dict(torch.load(saved_model_path))
    model1.to(device)

    
    optimizer = optim.Adam(model1.parameters(), lr=lr)
    

     
    model1.train()
    train_losses = []
    val_losses = []

    print('Learning rate = {}'.format(lr))
    print('Starting model training...')

    for epoch in range(num_epochs):
        start = time.time()
        print(start)
        print("TOTAL LOGS IN TRAINING", len(train_dataloader))
        # Train the model
        train_loss_epoch = []
        with tqdm(total=len(train_dataloader), unit="batch") as pbar:
            for batch_idx, (x, y) in enumerate(train_dataloader):
                print("TRAINING CURRENTLY")
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = torch.reshape(x, (2, -1, 160, 160, 1)) #double check the shape in generators.py
                optimizer.zero_grad()
                x = x.float()

                # Forward Pass
                y_hat = model1(x)

                #Compute Loss
                train_loss = loss(y, y_hat)

                #Backpropogation
                train_loss.backward()
                optimizer.step()

                train_loss_epoch.append(train_loss.detach().cpu())
                #train_losses.append(np.mean([loss.item() for loss in train_loss_step])) 
                print(train_loss)
                pbar.set_postfix({"Train Loss": np.mean([loss.item() for loss in train_loss_epoch])})  # Display on CPU
                pbar.update()

        
        avg_train_loss = np.mean([loss.item() for loss in train_loss_epoch])
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}")      
        # Validate the model
        if epoch % 2 == 0:
            val_loss_step = list()
            print("TOTAL LOGS IN VALIDATION ",len(val_dataloader))
            with torch.no_grad():
                with tqdm(total=len(val_dataloader), unit="batch") as pbar:
                    for batch_idx, (val_x, val_y) in enumerate(val_dataloader):
                        val_x = val_x.to(device,non_blocking=True)
                        val_y = val_y.to(device,non_blocking=True)
                        val_x = torch.reshape(val_x, (2, -1, 160, 160, 1))
                        val_x = val_x.float()
                        val_y_hat = model1(val_x)
                        val_loss = loss(val_y, val_y_hat)
                        val_loss_step.append(val_loss.detach().cpu()) 
                        #val_losses.append(np.mean([loss.item() for loss in val_loss_step]))  # Compute mean on CPU
                        pbar.set_postfix({"Validation Loss": np.mean([loss.item() for loss in val_loss_step])})
                        pbar.update()
            #val_epoch_loss = np.mean(val_loss_step)
            #val_losses.append(val_epoch_loss)
            stop = time.time()
            print(f'*' * 100)
            print(f'Epoch:{epoch}\t Time:{(stop-start)/60}minutes\t Train loss:{np.mean(train_loss_epoch)}\t Validation loss:{np.mean(val_loss_step)}')
        
        # Save the trained model at the end of each epoch
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"Model_epoch{epoch + 1}_{current_datetime}.pth"
        save_path = os.path.join(model_save_path, model_filename)
        torch.save(model1.state_dict(), save_path)
        
    print('Training complete')



if __name__ == "__main__":
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
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs for training')

    args = parser.parse_args()

    # Call the main function with provided arguments
    main(args.gps_folder, args.validation_folder, args.semseg_folder, args.labels_folder, args.model_save_path, args.losses_save_path, args.lr, args.num_epochs)
