import numpy as np
import scipy.io
from collections import defaultdict

# Paths to raw input files
us101_1 = 'archive/trajectories-0750am-0805am.txt'

# Load and preprocess data
print('Loading data...')
files = [us101_1]
traj = []

for idx, file in enumerate(files):
    data = np.loadtxt(file)
    dataset_id = (idx + 1) * np.ones((data.shape[0], 1))
    data = np.hstack((dataset_id, data)).astype(np.float32)
    data = data[:, [0, 1, 2, 5, 6, 14]]  # Select specific fields
    if idx < 3:  # For US-101 datasets
        data[data[:, 5] >= 6, 5] = 6
    traj.append(data)

# Combine all trajectories
traj_all = np.vstack(traj)

# Create empty columns for future (2 for behaviors, 13 * 3 for spatial gird)
for i in range(len(traj_all)):
    # new_cols = np.full((traj[i].shape[0], 2 + 13 * 3), None) #should use none or 0?
    new_cols = np.full((traj_all[i].shape[0], 2 + 13 * 3), 0)
    traj_all[i] = np.hstack((traj_all[i], new_cols))
                         
# Function to calculate lateral and longitudinal maneuvers and populate grid
def process_trajectory(traj):
    for k in range(traj.shape[0]):
        time = traj[k, 2]
        lane = traj[k, 5]
        veh_id = traj[k, 1]
        veh_traj = traj[traj[:, 1] == veh_id]

        # Lateral maneuver
        ub = min(len(veh_traj), np.where(veh_traj[:, 2] == time)[0][0] + 40)
        lb = max(0, np.where(veh_traj[:, 2] == time)[0][0] - 40)
        if veh_traj[ub - 1, 5] > veh_traj[lb, 5]:
            traj[k, 6] = 3  # Lane change right
        elif veh_traj[ub - 1, 5] < veh_traj[lb, 5]:
            traj[k, 6] = 2  # Lane change left
        else:
            traj[k, 6] = 1  # Lane keeping

        # Longitudinal maneuver
        ub = min(len(veh_traj), np.where(veh_traj[:, 2] == time)[0][0] + 50)
        lb = max(0, np.where(veh_traj[:, 2] == time)[0][0] - 30)
        v_hist = (veh_traj[np.where(veh_traj[:, 2] == time)[0][0], 4] - veh_traj[lb, 4]) / (np.where(veh_traj[:, 2] == time)[0][0] - lb + 1e-6)
        v_fut = (veh_traj[ub - 1, 4] - veh_traj[np.where(veh_traj[:, 2] == time)[0][0], 4]) / (ub - np.where(veh_traj[:, 2] == time)[0][0] + 1e-6)
        traj[k, 7] = 2 if v_fut / v_hist < 0.8 else 1

        # Populate grid
        grid = np.zeros((39,), dtype=np.float32)
        frame_ego = traj[(traj[:, 2] == time) & (traj[:, 5] == lane)]
        frame_left = traj[(traj[:, 2] == time) & (traj[:, 5] == lane - 1)]
        frame_right = traj[(traj[:, 2] == time) & (traj[:, 5] == lane + 1)]

        def populate_grid(frame, start_index):
            for veh in frame:
                y_diff = veh[4] - traj[k, 4]
                if -90 <= y_diff <= 90:
                    grid_idx = start_index + int((y_diff + 90) // 15)
                    grid[grid_idx] = veh[1]  # Neighbor vehicle ID

        populate_grid(frame_left, 0)
        populate_grid(frame_ego, 13)
        populate_grid(frame_right, 26)

        traj[k, 8:] = grid

    return traj

# Apply processing for lateral, longitudinal maneuvers, and grid
traj_all = process_trajectory(traj_all)

# Create `tracks` from the data
def create_tracks(traj_set):
    max_vehicle_id = int(np.max(traj_set[:, 1]))
    max_dataset_id = int(np.max(traj_set[:, 0]))
    tracks = [[None] * max_vehicle_id for _ in range(max_dataset_id)]
    for ds_id in range(1, max_dataset_id + 1):
        ds_traj = traj_set[traj_set[:, 0] == ds_id]
        for veh_id in range(1, max_vehicle_id + 1):
            veh_traj = ds_traj[ds_traj[:, 1] == veh_id]
            if len(veh_traj) > 0:
                tracks[ds_id - 1][veh_id - 1] = veh_traj[:, [2, 3, 4]].T  # Use time, local_x, local_y
    return np.array(tracks, dtype=object)

# Split into train, validation, and test sets
print('Splitting into train, validation and test sets...')
traj_tr, traj_val, traj_ts = [], [], []
tracks_tr, tracks_val, tracks_ts = None, None, None

for idx in range(1):
    dataset_traj = traj_all[traj_all[:, 0] == idx + 1]
    ul1 = round(0.7 * np.max(dataset_traj[:, 1]))
    ul2 = round(0.8 * np.max(dataset_traj[:, 1]))

    train_set = dataset_traj[dataset_traj[:, 1] <= ul1]
    val_set = dataset_traj[(dataset_traj[:, 1] > ul1) & (dataset_traj[:, 1] <= ul2)]
    test_set = dataset_traj[dataset_traj[:, 1] > ul2]

    traj_tr.append(train_set)
    traj_val.append(val_set)
    traj_ts.append(test_set)

# Generate tracks for each dataset
tracks_tr = create_tracks(np.vstack(traj_tr))
tracks_val = create_tracks(np.vstack(traj_val))
tracks_ts = create_tracks(np.vstack(traj_ts))

# Save datasets with trajectories and tracks
print('Saving datasets...')
np.save("TrainSet_one.npy", {"traj": np.vstack(traj_tr), "tracks": tracks_tr})
np.save("ValSet_one.npy", {"traj": np.vstack(traj_val), "tracks": tracks_val})
np.save("TestSet_one.npy", {"traj": np.vstack(traj_ts), "tracks": tracks_ts})
print('Process completed.')
