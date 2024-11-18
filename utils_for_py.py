from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class ngsimDataset(Dataset):
    def __init__(self, data_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        """
        Initialize the dataset with trajectory data and parameters.
        
        Args:
        - data_file: Path to the .npy file containing the processed trajectory and track data.
        - t_h: Length of track history in frames.
        - t_f: Length of predicted trajectory in frames.
        - d_s: Downsampling rate for sequences.
        - enc_size: Size of the encoder LSTM.
        - grid_size: Size of the spatial context grid (rows, columns).
        """
        data = np.load(data_file, allow_pickle=True).item()
        self.D = data['traj']  # Trajectories
        self.T = data['tracks']  # Tracks
        self.t_h = t_h  # Length of track history
        self.t_f = t_f  # Length of predicted trajectory
        self.d_s = d_s  # Downsampling rate
        self.enc_size = enc_size  # Encoder size
        self.grid_size = grid_size  # Grid size

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        # Extract data for a specific trajectory sample
        ds_id, veh_id, time = self.D[idx, 0].astype(int), self.D[idx, 1].astype(int), self.D[idx, 2]
        grid = self.D[idx, 8:]  # Social grid for neighboring vehicles
        neighbors = []

        # Get history and future trajectories for the target vehicle
        hist = self.getHistory(veh_id, time, veh_id, ds_id)
        fut = self.getFuture(veh_id, time, ds_id)

        # Get histories for neighboring vehicles
        for nbr_id in grid:
            neighbors.append(self.getHistory(int(nbr_id), time, veh_id, ds_id))

        # Encode maneuvers (lateral and longitudinal as one-hot vectors)
        lat_enc = np.zeros(3)
        lon_enc = np.zeros(2)
        lat_enc[int(self.D[idx, 6] - 1)] = 1  # Lateral maneuver
        lon_enc[int(self.D[idx, 7] - 1)] = 1  # Longitudinal maneuver

        return hist, fut, neighbors, lat_enc, lon_enc

    def getHistory(self, vehId, t, refVehId, dsId):
        """
        Retrieve the trajectory history for a vehicle relative to a reference vehicle.

        Args:
        - vehId: ID of the target vehicle.
        - t: Current frame time.
        - refVehId: ID of the reference vehicle (for spatial alignment).
        - dsId: Dataset ID.

        Returns:
        - ndarray: History of the vehicle's trajectory (x, y).
        """
        if vehId == 0:
            return np.empty((0, 2))

        # Get tracks for the reference and target vehicles
        ref_track = self.T[dsId][refVehId].T
        veh_track = self.T[dsId][vehId].T

        if len(veh_track) == 0 or t not in veh_track[:, 0]:
            return np.empty((0, 2))

        ref_pos = ref_track[ref_track[:, 0] == t][0, 1:3]
        hist_start = max(0, np.where(veh_track[:, 0] == t)[0][0] - self.t_h)
        hist_end = np.where(veh_track[:, 0] == t)[0][0] + 1
        hist = veh_track[hist_start:hist_end:self.d_s, 1:3] - ref_pos

        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty((0, 2))
        return hist

    def getFuture(self, vehId, t, dsId):
        """
        Retrieve the future trajectory for a vehicle.

        Args:
        - vehId: ID of the target vehicle.
        - t: Current frame time.
        - dsId: Dataset ID.

        Returns:
        - ndarray: Future trajectory (x, y).
        """
        veh_track = self.T[dsId][vehId].T

        if t not in veh_track[:, 0]:
            return np.empty((0, 2))

        fut_start = np.where(veh_track[:, 0] == t)[0][0] + self.d_s
        fut_end = min(len(veh_track), fut_start + self.t_f)
        ref_pos = veh_track[np.where(veh_track[:, 0] == t)][0, 1:3]
        fut = veh_track[fut_start:fut_end:self.d_s, 1:3] - ref_pos

        return fut

    def collate_fn(self, samples):
        """
        Custom collate function for batching data with varying lengths.

        Args:
        - samples: List of samples from `__getitem__`.

        Returns:
        - Batches of history, neighbors, masks, and maneuver encodings.
        """
        max_hist_len = self.t_h // self.d_s + 1
        max_fut_len = self.t_f // self.d_s

        hist_batch = torch.zeros((max_hist_len, len(samples), 2))
        fut_batch = torch.zeros((max_fut_len, len(samples), 2))
        op_mask_batch = torch.zeros((max_fut_len, len(samples), 2))
        lat_enc_batch = torch.zeros((len(samples), 3))
        lon_enc_batch = torch.zeros((len(samples), 2))
        nbr_batch = []
        nbr_masks = []

        for i, (hist, fut, nbrs, lat_enc, lon_enc) in enumerate(samples):
            hist_len = len(hist)
            fut_len = len(fut)
            hist_batch[:hist_len, i, :] = torch.tensor(hist, dtype=torch.float32)
            fut_batch[:fut_len, i, :] = torch.tensor(fut, dtype=torch.float32)
            op_mask_batch[:fut_len, i, :] = 1
            lat_enc_batch[i, :] = torch.tensor(lat_enc, dtype=torch.float32)
            lon_enc_batch[i, :] = torch.tensor(lon_enc, dtype=torch.float32)

            # Neighbors
            nbr_hist = torch.zeros((max_hist_len, len(nbrs), 2))
            for j, nbr in enumerate(nbrs):
                if len(nbr) > 0:
                    nbr_hist[:len(nbr), j, :] = torch.tensor(nbr, dtype=torch.float32)
            nbr_batch.append(nbr_hist)

        return hist_batch, nbr_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch
