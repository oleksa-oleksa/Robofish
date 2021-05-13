import robofish.io
import numpy as np

from pathlib import Path
import fish_models
import matplotlib.pyplot as plt

# Lets use the random model from above
raycast = fish_models.gym_interface.Raycast(
            n_wall_raycasts=5,
            n_fish_bins=4,
            fov_angle_fish_bins=np.pi,
            fov_angle_wall_raycasts=np.pi,
            world_bounds=([-50, -50], [50, 50]),
        )


data_folder = Path("data/live_female_female/train")

dset = fish_models.datasets.io_dataset.IoDataset(
    data_folder,
    raycast,
    output_strings=["poses", "actions", "views"],
    reduce_dim=2,
    max_files=10,
)


plt.figure(figsize=(10,10))
plt.xlim(-50,50)
plt.ylim(-50,50)
plt.plot(dset[:, :, 0], dset[:, :, 1])
plt.show()
