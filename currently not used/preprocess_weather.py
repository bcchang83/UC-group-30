import numpy as np
import pandas as pd
from scipy.io import savemat
from datetime import datetime, timedelta


# Input files
us101_files = ["trajectories-0750am-0805am.txt"]

files = us101_files

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
data = []
for i, file in enumerate(files):
    traj = pd.read_csv(file, sep='\s+', header=None, names=column_names)  # Fixed delimiter
    traj.insert(0, "DatasetId", i + 1)
    data.append(traj)

# Combine all trajectory data into a single DataFrame
data = pd.concat(data, ignore_index=True)

# Weather data processing
weather = pd.read_csv(
    'weather/Hollywood Freeway, 2005-06-15.csv',
    usecols=['datetime', 'precip', 'windspeed', 'visibility']
)
weather['datetime'] = pd.to_datetime(weather['datetime'])  # Ensure 'datetime' is in datetime format
weather = weather.set_index('datetime').sort_index()  # Sort weather data by datetime

# Convert Global_Time to datetime (already elapsed since 1970-01-01)
# data['trajectory_time'] = pd.to_datetime(data['Global_Time'], unit='ms')
# data['trajectory_time'] = data['Global_Time'].apply(
#     lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=ms)
# )
data['trajectory_time'] = pd.to_datetime(data['Global_Time'], unit='ms') - pd.Timedelta(hours=7)


# Sort trajectory times for merging
data = data.sort_values('trajectory_time')

# Use merge_asof to align trajectory times with weather data
data = pd.merge_asof(
    data,#[['trajectory_time']],  # Only the trajectory times
    weather.reset_index().rename(columns={'datetime': 'trajectory_time'}),
    on='trajectory_time'
)

# Join mapped weather back to the original trajectory data
#cresult = pd.concat([data, mapped_weather.drop(columns=['trajectory_time'])], axis=1)

# Result: Trajectory data with interpolated weather
print(data.head())

