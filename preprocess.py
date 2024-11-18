import numpy as np
import pandas as pd
from scipy.io import savemat

# Input files
us101_files = [
    "raw/us-101/trajectories-0750am-0805am.txt",
    "raw/us-101/trajectories-0805am-0820am.txt",
    "raw/us-101/trajectories-0820am-0835am.txt",
]
i80_files = [
    "raw/i-80/trajectories-0400-0415.txt",
    "raw/i-80/trajectories-0500-0515.txt",
    "raw/i-80/trajectories-0515-0530.txt",
]
files = us101_files + i80_files

# Load data and add dataset id
print("Loading data...")
data = []
for i, file in enumerate(files):
    traj = pd.read_csv(file, delim_whitespace=True, header=None)
    traj.insert(0, "DatasetId", i + 1)
    data.append(traj.to_numpy(dtype=np.float32))

# Select and process relevant columns
traj = [d[:, [0, 1, 2, 5, 6, 14]] for d in data]
for i in range(len(traj)):
    if i < 3:
        traj[i][traj[i][:, 5] >= 6, 5] = 6


# Parse fields
print("Parsing fields...")
veh_trajs = [{} for _ in range(len(traj))]
veh_times = [{} for _ in range(len(traj))]

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
        ub = min(len(veh_traj), ind + 40)
        lb = max(0, ind - 40)
        if veh_traj[ub, 5] > veh_traj[ind, 5] or veh_traj[ind, 5] > veh_traj[lb, 5]:
            row[6] = 3
        elif veh_traj[ub, 5] < veh_traj[ind, 5] or veh_traj[ind, 5] < veh_traj[lb, 5]:
            row[6] = 2
        else:
            row[6] = 1

        # Get longitudinal maneuver
        ub = min(len(veh_traj), ind + 50)
        lb = max(0, ind - 30)
        if ub == ind or lb == ind:
            row[7] = 1
        else:
            v_hist = (veh_traj[ind, 4] - veh_traj[lb, 4]) / (ind - lb)
            v_fut = (veh_traj[ub, 4] - veh_traj[ind, 4]) / (ub - ind)
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
    tracks = {}
    for i in range(6):
        traj_by_id = traj_set[i]
        unique_ids = np.unique(traj_by_id[:, 1])
        for uid in unique_ids:
            track = traj_by_id[traj_by_id[:, 1] == uid][:, 2:5].T
            tracks.setdefault(i + 1, {})[int(uid)] = track
    return tracks


# Train, Validation, Test tracks
tracks_tr = create_tracks(traj_tr)
tracks_val = create_tracks(traj_val)
tracks_ts = create_tracks(traj_ts)

# Save .mat files
savemat("TrainSet.mat", {"traj": np.vstack(traj_tr), "tracks": tracks_tr})
savemat("ValSet.mat", {"traj": np.vstack(traj_val), "tracks": tracks_val})
savemat("TestSet.mat", {"traj": np.vstack(traj_ts), "tracks": tracks_ts})

print("Done.")
