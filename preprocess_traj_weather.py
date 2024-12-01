import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Input files
us101_files = ["archive/trajectories-0750am-0805am.txt"]

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
print("Loading trajectory data...")
data = []
for i, file in enumerate(files):
    traj = pd.read_csv(file, sep='\s+', header=None, names=column_names)
    traj.insert(0, "DatasetId", i + 1)
    data.append(traj)

# Combine all trajectory data into a single DataFrame
data = pd.concat(data, ignore_index=True)

# Convert Global_Time to datetime
data['trajectory_time'] = pd.to_datetime(data['Global_Time'], unit='ms') - pd.Timedelta(hours=7)

# Load weather data
print("Loading weather data...")
weather = pd.read_csv(
    'Hollywood Freeway, 2005-06-15 to 2005-06-16.csv',
    usecols=['datetime', 'precip', 'windspeed', 'visibility']
)
weather['datetime'] = pd.to_datetime(weather['datetime'])  # Ensure 'datetime' is in datetime format
weather = weather.set_index('datetime').sort_index()  # Sort weather data by datetime


# Sort trajectory data by trajectory_time
data = data.sort_values('trajectory_time')

# Sort weather data by datetime (already set as index but ensure it's sorted)
weather = weather.sort_index()

# Merge weather data into trajectory data
print("Aligning weather data with trajectory data...")
mapped_weather = pd.merge_asof(
    data[['trajectory_time']].sort_values('trajectory_time'),  # Ensure sorted keys
    weather.reset_index().rename(columns={'datetime': 'trajectory_time'}).sort_values('trajectory_time'),
    on='trajectory_time'
)

data = pd.concat([data, mapped_weather.drop(columns=['trajectory_time'])], axis=1)

# Process trajectory data further
print("Processing trajectory data...")
traj = [
    data[['DatasetId', 'Vehicle_ID', 'Frame_ID', 'Local_Y', 'Lane_ID', 'precip', 'windspeed', 'visibility']].to_numpy(dtype=np.float32)
]

# Add grid-based processing logic (this is a placeholder for your actual logic)
for i, t in enumerate(traj):
    veh_ids = np.unique(t[:, 1])
    for veh_id in veh_ids:
        veh_traj = t[t[:, 1] == veh_id]

# Split into train, validation, and test sets
print("Splitting into train, validation, and test sets...")
traj_all = np.vstack(traj)
traj_tr, traj_val, traj_ts = [], [], []

for i in range(1):
    max_id = int(traj_all[traj_all[:, 0] == i + 1, 1].max())
    ul1, ul2 = int(0.7 * max_id), int(0.8 * max_id)

    traj_tr.append(traj_all[(traj_all[:, 0] == i + 1) & (traj_all[:, 1] <= ul1)])
    traj_val.append(
        traj_all[
            (traj_all[:, 0] == i + 1) & (traj_all[:, 1] > ul1) & (traj_all[:, 1] <= ul2)
        ]
    )
    traj_ts.append(traj_all[(traj_all[:, 0] == i + 1) & (traj_all[:, 1] > ul2)])

# Organize tracks and save
print("Saving combined trajectory and weather data...")

# Save .npy files with trajectory and weather data combined
np.save("TrainSet_one.npy", {"traj": np.vstack(traj_tr)})
np.save("ValSet_one.npy", {"traj": np.vstack(traj_val)})
np.save("TestSet_one.npy", {"traj": np.vstack(traj_ts)})

print("Done.")
