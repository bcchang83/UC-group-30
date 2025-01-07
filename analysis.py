import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

us101_files = [
    "us-101/trajectories-0750am-0805am_weather_5_features.csv",
    "us-101/trajectories-0805am-0820am_weather_5_features.csv",
    "us-101/trajectories-0820am-0835am_weather_5_features.csv",
]
i80_files = [
    "i-80/trajectories-0400-0415_weather_5_features.csv",
    "i-80/trajectories-0500-0515_weather_5_features.csv",
    "i-80/trajectories-0515-0530_weather_5_features.csv",
]
files = us101_files + i80_files

print("Loading data...")
data = pd.DataFrame()
for i, file in enumerate(files):
    traj = pd.read_csv(file)
    traj = traj.drop("trajectory_time", axis=1)

    # Fix the warning
    # traj = pd.read_csv(file, sep='\s+', header=None, names=column_names)
    traj.insert(0, "DatasetId", i + 1)
    traj["Vehicle_ID"] += i * 10000
    data = pd.concat([data, traj])

analysis = {
    "vehicle_id": [],
    "change_dir": [],
    "slow_down": [],
    "high_visibility": [],
    "high_humidity": [],
    "high_windspeed": [],
}
for vid in data.Vehicle_ID.unique():

    change_dir = 0
    slow = 0
    vis = 0
    hum = 0
    wind = 0

    veh_traj = data[data.Vehicle_ID == vid]

    # check direction
    if len(veh_traj["Lane_ID"].unique()) > 1:
        change_dir = 1

    # group weather condition
    if (veh_traj["visibility"] > 5).any():
        vis = 1
    if (veh_traj["humidity"] > 60).any():
        hum = 1
    if (veh_traj["windspeed"] > 10).any():
        wind = 1

    # check speed
    veh_traj = veh_traj.to_numpy(dtype=np.float32)
    for idx in range(len(veh_traj[:, 2])):
        ub = min(len(veh_traj) - 1, idx + 50)
        lb = max(0, idx - 30)

        if ub != idx and lb != idx:
            # if ub == idx or lb == idx:
            #     slow = 0
            # else:
            v_hist = (veh_traj[idx, 6] - veh_traj[lb, 6] + 1e-6) / (idx - lb + 1e-6)
            v_fut = (veh_traj[ub, 6] - veh_traj[idx, 6] + 1e-6) / (ub - idx + 1e-6)
            if v_fut / v_hist < 0.8:
                slow = 1
                break

    analysis["vehicle_id"].append(vid)
    analysis["change_dir"].append(change_dir)
    analysis["slow_down"].append(slow)
    analysis["high_visibility"].append(vis)
    analysis["high_humidity"].append(hum)
    analysis["high_windspeed"].append(wind)

analysis = pd.DataFrame(analysis)

analysis["high_visibility"] = analysis["high_visibility"].map(
    {0: "Low Visibility", 1: "High Visibility"}
)
analysis["high_humidity"] = analysis["high_humidity"].map(
    {0: "Low Humidity", 1: "High Humidity"}
)
analysis["high_windspeed"] = analysis["high_windspeed"].map(
    {0: "Low Wind Speed", 1: "High Wind Speed"}
)
plt.figure()
sns.barplot(x="high_visibility", y="change_dir", data=analysis)
plt.title("Impact of High Visibility on Lane Changing")
plt.show()
plt.figure()
sns.barplot(x="high_visibility", y="slow_down", data=analysis)
plt.title("Impact of High Visibility on Slowing Down")
plt.show()
plt.figure()
sns.barplot(x="high_humidity", y="change_dir", data=analysis)
plt.title("Impact of High Humidity on Lane Changing")
plt.show()
plt.figure()
sns.barplot(x="high_humidity", y="slow_down", data=analysis)
plt.title("Impact of High Humidity on Slowing Down")
plt.show()
plt.figure()
sns.barplot(x="high_windspeed", y="change_dir", data=analysis)
plt.title("Impact of High Wind Speed on Lane Changing")
plt.show()
plt.figure()
sns.barplot(x="high_windspeed", y="slow_down", data=analysis)
plt.title("Impact of High Wind Speed on Slowing Down")
plt.show()
