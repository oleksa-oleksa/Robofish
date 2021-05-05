import robofish.io
import numpy as np


# Create a new robofish io file
f = robofish.io.File(world_size_cm=[100, 100], frequency_hz=25.0)
f.attrs["experiment_setup"] = "This is a simple example with made up data."

# Create a new robot entity with 10 timesteps.
# Positions and orientations are passed separately in this example.
# Since the orientations have two columns, unit vectors are assumed
# (orientation_x, orientation_y)
f.create_entity(
    category="robot",
    name="robot",
    positions=np.zeros((10, 2)),
    orientations=np.ones((10, 2)) * [0, 1],
)

# Create a new fish entity with 10 timesteps.
# In this case, we pass positions and orientations together (x, y, rad).
# Since it is a 3 column array, orientations in radiants are assumed.
poses = np.zeros((10, 3))
poses[:, 0] = np.arange(-5, 5)
poses[:, 1] = np.arange(-5, 5)
poses[:, 2] = np.arange(0, 2 * np.pi, step=2 * np.pi / 10)
fish = f.create_entity("fish", poses=poses)
fish.attrs["species"] = "My rotating spaghetti fish"
fish.attrs["fish_standard_length_cm"] = 10

# Some possibilities to access the data
print(f"The file:\n{f}")
print(
    f"Poses Shape:\t{f.entity_poses_rad.shape}.\t"
    + "Representing(entities, timesteps, pose dimensions (x, y, ori)"
)
print(f"The actions of one Fish, (timesteps, (speed, turn)):\n{fish.speed_turn}")
print(f"Fish poses with calculated orientations:\n{fish.poses_calc_ori_rad}")

# Save the file
f.save_as("example.hdf5")