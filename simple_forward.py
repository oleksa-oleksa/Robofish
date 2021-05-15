import robofish.io
import numpy as np

from pathlib import Path
import fish_models
import matplotlib.pyplot as plt


class SimpleForwardModel(fish_models.gym_interface.AbstractRaycastBasedModel):
    def choose_action(self, view: np.ndarray):
        # Return speed and turn from view
        speed = np.random.random() * 20.
        turn = (np.random.random() - 0.5) * 5.
        return speed, turn


model = SimpleForwardModel()

# Lets use the random model from above
raycast = fish_models.gym_interface.Raycast(
            n_wall_raycasts=5,
            n_fish_bins=4,
            fov_angle_fish_bins=np.pi,
            fov_angle_wall_raycasts=np.pi,
            world_bounds=([-50, -50], [50, 50]),
        )

generator = fish_models.gym_interface.TrackGeneratorGymRaycast(
    model, raycast, [100,100], 25
)
track = generator.create_track(n_guppies=2, trackset_len=1000)

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
