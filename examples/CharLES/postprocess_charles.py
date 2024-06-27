import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import warnings
import math
import argparse

def read_data(fname, pcloud_fname):
    # Read the data
    print(fname)
    with open(fname, 'r') as file:
        lines = file.readlines()
    
    # Skip the first two lines and read the third line
    desc_line = lines[2]
    
    # Read the data
    data = np.array([np.fromstring(line, sep=' ') for line in lines[3:]]).T
    tstep = data[0, :]
    t = data[1, :]
    # count = data[2, :]
    data = data[3:, :]
    Nsaved = data.shape[1]
    
    # Reshape data
    pcloud = pd.read_csv(pcloud_fname, sep='\s+', header=None)
    
    # Point cloud sweeps z, then sweeps y, then sweeps x
    x = np.unique(pcloud.iloc[:, 0])
    y = np.unique(pcloud.iloc[:, 1])
    z = np.unique(pcloud.iloc[:, 2])
    
    Nx = len(x)
    Ny = len(y)
    Nz = len(z)
    
    data_unreshaped = data
    
    data_out = np.zeros((Nx, Ny, Nz, Nsaved))
    
    for isav in range(Nsaved):
        # Data for one saved time
        data_isav = data[:, isav]
        # Reshape x
        Nbx = Ny * Nz
        data_x = np.zeros((Nx, Nbx))
        for ix in range(Nx):
            data_x[ix, :] = data_isav[ix * Nbx:(ix + 1) * Nbx]
        # Reshape y
        data_y = np.zeros((Nx, Ny, Nz))
        for iy in range(Ny):
            data_y[:, iy, :] = data_x[:, iy * Nz:(iy + 1) * Nz]
        data_out[:, :, :, isav] = data_y
    
    return data_out, t, tstep, x, y, z, data_unreshaped



# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some values.')
parser.add_argument('N_ens', type=int, help='First integer value')
parser.add_argument('statistics', type=int, help='Second integer value')
parser.add_argument('iteration', type=int, help='Third integer value')
args = parser.parse_args()
N_ens = args.N_ens
statistics = args.statistics
iteration = args.iteration


# Base directory path for the data (will need to rewrite directory here)
base_dir = r'/home/ctrsp-2024/youngin/PadeOps/data/half_channel_concurrent/padeops_runs'
restart_idx = 0
start_idx = 3
end_idx = 5

# Point cloud (need to specify this)
pcloud_fname = r'C:\Users\miles\Documents\out\summerproject\point_cloud.txt'

# Initialize lists to store cumulative results
Ubar_list = []

# Loop through runs
for run_num in range(1, N_ens+1):
    print(run_num)
    budget_dir = f'{base_dir}/{iteration}/{run_num}'
    # budget_dir = r"C:/Users/miles/Documents/out/summerproject/probesABL"
    input_fname = f'{budget_dir}/ABLpoints.COMP(AVG(U),0)'
    print(input_fname)
    data_out, t, tstep, x, y, z, _  = read_data(input_fname, pcloud_fname)

    U_tavg = (data_out[:,:,:,end_idx] * t[end_idx] - data_out[:,:,:,start_idx] * t[start_idx]) / (t[end_idx] - t[start_idx])
    Ubar = np.mean(U_tavg, axis=(0,2))
 
    # Add to cumulative lists
    Ubar_list.append(Ubar)


# Convert lists to numpy arrays if desired
Ubar_array = np.array(Ubar_list)

# Define the path for the CSV file
csv_file_path = os.path.join(os.getcwd(), "g_ens_temp.csv")
# Save the array to a CSV file
np.savetxt(csv_file_path, Ubar_array, delimiter=",", fmt='%.15f')
