import numpy as np
import pandas as pd
from scipy.io import savemat

# Input files

# Example file
# us101_files = ["trajectories-0750am-0805am.txt"]
# files = us101_files

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
files = us101_files + i80_files

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
    traj = pd.read_csv(file, delim_whitespace=True, header=None, names=column_names)
    # Fix the warning
    # traj = pd.read_csv(file, sep='\s+', header=None, names=column_names)

    traj.insert(0, "DatasetId", i + 1)
    data.append(traj.to_numpy(dtype=np.float32))

# Parse fields
print("Parsing fields...")
veh_trajs = [{} for _ in range(len(data))]  # Create an empty dictionary per dataset
veh_times = [{} for _ in range(len(data))]

# Select and process relevant columns
traj = [
    d[:, [0, 1, 2, 5, 6, 14]] for d in data
]  # Take Dataset ID, Vehicle ID, Frame ID, Local X Y, Lane ID
for i in range(len(traj)):
    if i < 3:  # For datasets 0, 1, 2 (US-101 files), any Lane_ID >= 6 is set to 6
        traj[i][traj[i][:, 5] >= 6, 5] = 6

# Create empty columns for future (2 for behaviors, 13 * 3 for spatial grid)
for i in range(len(traj)):
    new_cols = np.full((traj[i].shape[0], 2 + 13 * 3), 0)
    traj[i] = np.hstack((traj[i], new_cols))

for i, t in enumerate(traj):
    veh_ids = np.unique(t[:, 1])
    for v_id in veh_ids:
        veh_trajs[i][str(int(v_id))] = t[t[:, 1] == v_id]

    time_frames = np.unique(t[:, 2])
    for tf in time_frames:
        veh_times[i][str(int(tf))] = t[t[:, 2] == tf]

    for row in t:
        time, ds_id, veh_id, lane = row[2], row[0], row[1], int(row[5])
        veh_traj = veh_trajs[i][str(int(veh_id))]
        ind = np.where(veh_traj[:, 2] == time)[0][0]

        # Get lateral maneuver
        ub = min(len(veh_traj) - 1, ind + 40)  # upper bound for time (4 seconds)
        lb = max(0, ind - 40)  # lower bound for time (-4 seconds)

        if veh_traj[ub, 5] > veh_traj[ind, 5] or veh_traj[ind, 5] > veh_traj[lb, 5]:
            row[6] = 3  # turn right
        elif veh_traj[ub, 5] < veh_traj[ind, 5] or veh_traj[ind, 5] < veh_traj[lb, 5]:
            row[6] = 2  # turn left
        else:
            row[6] = 1  # no change direction

        # Get longitudinal maneuver
        ub = min(len(veh_traj) - 1, ind + 50)
        lb = max(0, ind - 30)

        # Keep speed
        if ub == ind or lb == ind:
            row[7] = 1
        else:
            v_hist = (veh_traj[ind, 4] - veh_traj[lb, 4] + 1e-6) / (ind - lb + 1e-6)
            v_fut = (veh_traj[ub, 4] - veh_traj[ind, 4] + 1e-6) / (ub - ind + 1e-6)
            row[7] = 2 if v_fut / v_hist < 0.8 else 1

        # Populate grid locations
        t_frame = veh_times[i][str(int(time))]
        frame_ego = t_frame[t_frame[:, 5] == lane]
        frame_l = t_frame[t_frame[:, 5] == lane - 1]
        frame_r = t_frame[t_frame[:, 5] == lane + 1]

        def update_grid(frames, start_idx):
            for f in frames:
                y = f[4] - row[4]
                if abs(y) < 90:
                    grid_ind = start_idx + int(round((y + 90) / 15))
                    row[8 + grid_ind] = f[1]

        update_grid(frame_l, 0)
        update_grid(frame_ego, 13)
        update_grid(frame_r, 26)

# Split into train, validation, and test sets
print("Splitting into train, validation, and test sets...")
traj_all = np.vstack(traj)
traj_tr, traj_val, traj_ts = [], [], []

for i in range(6):
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
print("Saving mat files...")

def create_tracks(traj_set):
    max_veh_id = int(max([traj[:, 1].max() for traj in traj_set]))
    max_ds_id = len(traj_set)
    # tracks = np.full((max_ds_id, max_veh_id), None, dtype=object)
    tracks = np.array([[None for x in range(31)] * max_veh_id for _ in range(max_ds_id)])

    for ds_id, traj in enumerate(traj_set, start=1):
        unique_ids = np.unique(traj[:, 1])
        for veh_id in unique_ids:
            track = traj[traj[:, 1] == veh_id][:, 2:5].T
            tracks[ds_id - 1, int(veh_id) - 1] = track

    return tracks

# Train, Validation, Test tracks
tracks_tr = create_tracks(traj_tr)
tracks_val = create_tracks(traj_val)
tracks_ts = create_tracks(traj_ts)

# def filter_edge_cases(traj, tracks):
#     """
#     Filter edge cases from trajectory data based on the 3-second history condition.
#     """
#     inds = np.zeros(len(traj), dtype=bool)

#     for k in range(len(traj)):
#         dataset_id = k#traj[k, 0].astype(int)  # Dataset ID
#         vehicle_id = traj[k,:,1].astype(int)  # Vehicle ID
#         time_frame = traj[k,:,2]  # Current time frame
#         veh_idx = vehicle_id - 1

#         track = tracks[dataset_id][
#             veh_idx
#         ]  
#         # Ensure track has at least 31 frames
#         # if track is not None and len(track[0]) > 30:
#             # Fetch track for dataset ID and vehicle ID; 30 frames = 3 seconds
#         if track[0][30] <= time_frame and track[0][-1] > time_frame + 1:
#             inds[k] = True

#     return traj[inds]

# traj_tr = np.array(traj_tr)
# traj_val = np.array(traj_val)
# traj_ts = np.array(traj_ts)
# # Apply filtering
# print("Filtering edge cases...")
# filter_traj_tr = filter_edge_cases(traj_tr, tracks_tr)
# filter_traj_val = filter_edge_cases(traj_val, tracks_val)
# filter_traj_ts = filter_edge_cases(traj_ts, tracks_ts)

# Save .mat files
np.save("TrainSet.npy", {"traj": np.vstack(traj_tr), "tracks": tracks_tr})
np.save("ValSet.npy", {"traj": np.vstack(traj_val), "tracks": tracks_val})
np.save("TestSet.npy", {"traj": np.vstack(traj_ts), "tracks": tracks_ts})
print("Done.")
