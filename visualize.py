import os
import pickle

import matplotlib.pyplot as plt

# dataset_dir = "/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_tokyo"
dataset_dir = "/home/caoruixiang/datasets_mnt/scalenet"

traj_dir = os.path.join(dataset_dir, "trajectories")

for traj_name in sorted(os.listdir(traj_dir)):
    traj_path = os.path.join(traj_dir, traj_name, "traj_est.pkl")
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    traj = traj["raw"]

    # Plot quat raw value in the "raw" [N, 7] array [w, p, q, r]
    plt.figure(figsize=(24, 8))
    for i in range(3, 7):
        plt.plot(traj[:, i])

    # Plot
    plt.title(traj_name)
    plt.xlabel("index")
    plt.ylabel("quat raw value")
    plt.legend(["w", "p", "q", "r"])
    plt.xlim([0, len(traj)])
    plt.ylim([-1, 1])
    plt.show()

