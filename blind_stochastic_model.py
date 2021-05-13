import robofish.io
import numpy as np
import fish_models
from pathlib import Path
import importlib


bins = 1
raycast_options = {
    "n_fish_bins": bins,
    "n_wall_raycasts": bins,
    "fov_angle_fish_bins": 2 * np.pi,
    "fov_angle_wall_raycasts": 2 * np.pi,
    "world_bounds": ([-50, -50], [50, 50]),
    # "far_plane": 142,
}

# Raycast in cm
raycast = fish_models.gym_interface.Raycast(**raycast_options)

# IO Files in cm, actions in m/s
data_folder = Path("data/live_female_female/train")
dset = fish_models.datasets.io_dataset.IoDataset(data_folder, raycast,
                                                 output_strings=["poses", "actions", "views"], reduce_dim=2, max_files=1)

######

import importlib
importlib.reload(fish_models)

model = fish_models.models.andi.blind_stochastic_model.BlindStochasticModel(tau=0.01)
model.train(dset)
model.plot()

# not implemented yet
# fish_models.gym_interface.ModelStorage.save_model("model.pt", model, raycast.options)

generator = fish_models.gym_interface.TrackGeneratorGymRaycast(
    model, raycast, dset.world_size, dset.frequency
)

# initial_poses = dset.poses[0, :, 0]
timesteps = 1500
initial_poses = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
track = generator.create_track(2, timesteps, initial_poses=initial_poses)
f = generator.asIoFile(track)
f.save_as("__generated.hdf5")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title("Blind Stochastic Model")
plt.xlim(-50,50)
plt.ylim(-50,50)
for fish_id in range(2):
    plt.plot(
        track[fish_id, :, 0],
        track[fish_id, :, 1],
        label="reconstructed",
    )

plt.show()