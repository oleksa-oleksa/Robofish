from sklearn.mixture import GaussianMixture
import numpy as np
from pathlib import Path

import fish_models
import robofish.io
import random


class EMAlgorithmFishModel(fish_models.gym_interface.AbstractRaycastBasedModel):
    """
    Representation of a Gaussian mixture model probability distribution.
    GMMs are probabilistic models that assume all the data points are generated
    from a mixture of several Gaussian distributions with unknown parameters.

    The EM algorithm is an iterative approach that cycles between two modes:
    E-Step. Estimate the missing variables in the dataset.
    M-Step. Maximize the parameters of the model in the presence of the data.
    """

    def __init__(self):
        self.model = GaussianMixture(
            n_components=16, covariance_type="full", tol=0.01, max_iter=1000
        )
        self.clusters = {}

    def choose_action(self, view: np.ndarray):
        """
        Gets the mean value for speed and turn from the bunch of saved actions
        belonging to a particular cluster that is predicted by predic-functions

        To avoid a circle pit that fishes are tending to reach in a few minutes with
        clustering model, a stochastic behavioural added to slighthy modify
        predicted speed and turn.

        At the end of calcultations the distance to a wall ahead will be checked and
        in a case the wall is near the calculated turn will be modified once again to
        avoid fish to be stucked in the corner.

        Parameters
        ---------
        view : array_like
            The observations of the virtual fish

        Returns
        ---------
        speed, turn : tuple
            predicted speed/turn after calculations mentioned above
        """

        cluster_id = self.model.predict([view])

        # ------
        # Stochastic part to avoid fishes' circle pit

        speeds = []
        turns = []

        # puts speeds and turns in an separate array
        for i in range(len(self.clusters[str(cluster_id[0])])):
            speeds = speeds + [self.clusters[str(cluster_id[0])][i][0]]
            turns = turns + [self.clusters[str(cluster_id[0])][i][1]]

        # determined stochastic values
        bins = 100
        max_speed = 20
        max_turn = 20

        speeds = np.clip(speeds, 0, max_speed)
        turns = np.clip(turns, -1 * max_turn, max_turn)

        self.speed_bins = np.linspace(0, max_speed, bins)
        self.turn_bins = np.linspace(-1 * max_turn, max_turn, bins)

        self.speed_hist = np.bincount(
            np.digitize(speeds, self.speed_bins), minlength=bins
        )[:bins]
        self.turn_hist = np.bincount(
            np.digitize(turns, self.turn_bins), minlength=bins
        )[:bins]

        self.speed_hist = np.array(self.speed_hist) / np.sum(self.speed_hist)
        self.turn_hist = np.array(self.turn_hist) / np.sum(self.turn_hist)

        speed = np.random.choice(self.speed_bins, p=self.speed_hist)
        turn = np.random.choice(self.turn_bins, p=self.turn_hist)

        # -----

        # turn correction for walls avoidance
        turn = self.avoid_walls(view, turn)

        return speed, turn

    def avoid_walls(self, view, turn):
        """
        Forces to turn a fish in a random direction
        if in a view's raycast of the walls
        a wall in the front of a fish is detected to near

        Parameters
        ---------
        view : array_like
            The observations of the virtual fish
        turn : float
            Turn predicted by a model that is to modify

        Returns
        ---------
        turn : float
            Original or modified turn depending on the wall distance
        """
        param = random.randint(-5, 5)

        if param == 0:
            param = random.randint(5, 11)

        if view[6] > 0.9:
            return param * np.pi
        else:
            return turn

    def fit(self, dset):
        """
        Runs the two steps of EM Algorithm
        until the average log-likelihood converges.

        Saves the clusters labels and actions to obtain them by predict-step
        """
        actions = dset[:]["actions"]
        views = dset[:]["views"]

        # Train
        self.model.fit(views)

        # collect clustercenters as dictionary
        for center in range(len(self.model.means_)):
            self.clusters[str(center)] = []

        # assign data to clustercenters
        for point in dset:
            prediction = self.model.predict([point["views"]])[0]
            self.clusters[str(prediction)] += [point["actions"]]

        # print(self.clusters[str(3)])
