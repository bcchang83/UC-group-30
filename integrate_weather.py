import numpy as np
import pandas as pd
from scipy.io import savemat
from datetime import datetime, timedelta


# Input files
# All files
us101_files = [
    "us-101/trajectories-0750am-0805am.txt",
    "us-101/trajectories-0805am-0820am.txt",
    "us-101/trajectories-0820am-0835am.txt",
]
i80_files = [
    "i-80/trajectories-0400-0415.txt",
    "i-80/trajectories-0500-0515.txt",
    "i-80/trajectories-0515-0530.txt",
]

files = us101_files+i80_files

w_files = ['weather/Hollywood Freeway, 2005-06-15.csv']*3 +\
         ['weather/Bay area Emeryville Calif... 2005-04-13 to 2005-04-14.csv']*3

column_names = [
    "Vehicle_ID",
    "Frame_ID",
    "Total_Frames",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Global_X",
    "Global_Y",
    "v_Length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
]





# Load data and add dataset id
print("Loading data...")
for i in range(len(files)):
    traj = pd.read_csv(files[i], sep='\s+', header=None, names=column_names)  # Fixed delimiter
    # Weather data processing
    weather = pd.read_csv(w_files[i],
    usecols=['datetime', 'precip', 'windspeed', 'visibility']
    )
    
    weather['datetime'] = pd.to_datetime(weather['datetime'])  # Ensure 'datetime' is in datetime format
    weather = weather.set_index('datetime').sort_index()  # Sort weather data by datetime

    traj['trajectory_time'] = pd.to_datetime(traj['Global_Time'], unit='ms') - pd.Timedelta(hours=7)

    # Sort trajectory times for merging
    traj = traj.sort_values('trajectory_time')

    # Use merge_asof to align trajectory times with weather data
    traj = pd.merge_asof(
        traj,#[['trajectory_time']],  # Only the trajectory times
        weather.reset_index().rename(columns={'datetime': 'trajectory_time'}),
        on='trajectory_time'
    )

    traj.to_csv(f'{files[i][:-4]}_weather.csv', index=False)
