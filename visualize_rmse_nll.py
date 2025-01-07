import seaborn as sns
import matplotlib.pyplot as plt

# Data
rmse = [0.7115, 1.5052, 2.4360, 3.5530, 4.9117]
rmse_w = [0.6954, 1.4869, 2.4022, 3.4955, 4.8337]
rmse_w5 = [0.6904, 1.4672, 2.3686, 3.4654, 4.7684]
nll = [1.6924, 3.1566, 3.9732, 4.5956, 5.1193]
nll_w = [1.6962, 3.1504, 3.9723, 4.5815, 5.0880]
nll_w5 = [1.6611, 3.1117, 3.9496, 4.5683, 5.0853]

# Set Seaborn style
sns.set(style="whitegrid")

# Create a figure and axes
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# Plot RMSE
ax[0].plot(rmse, marker="o", label="RMSE", color="blue", linestyle="--")
ax[0].plot(rmse_w, marker="s", label="RMSE Weather", color="orange", linestyle="-")
ax[0].plot(rmse_w5, marker="^", label="RMSE Weather-5", color="purple", linestyle=":")
ax[0].set_title("Root Mean Squared Error (RMSE)", fontsize=14)
ax[0].set_xlabel("Index", fontsize=12)
ax[0].set_ylabel("RMSE Value", fontsize=12)
ax[0].legend(fontsize=12)
ax[0].grid(True)

# Plot NLL
ax[1].plot(nll, marker="o", label="NLL", color="green", linestyle="--")
ax[1].plot(nll_w, marker="s", label="NLL Weather", color="red", linestyle="-")
ax[1].plot(nll_w5, marker="^", label="NLL Weather-5", color="purple", linestyle=":")
ax[1].set_title("Negative Log Likelihood (NLL)", fontsize=14)
ax[1].set_xlabel("Index", fontsize=12)
ax[1].set_ylabel("NLL Value", fontsize=12)
ax[1].legend(fontsize=12)
ax[1].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
