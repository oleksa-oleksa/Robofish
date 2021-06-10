import fish_models
from sklearn.cluster import MeanShift
import sklearn.cluster as cluster
import sklearn.metrics as mt
import numpy as np
import random


class MeanShiftFishModel(fish_models.gym_interface.AbstractRaycastBasedModel):
    """
    Mean shift clustering using a flat kernel.

    MeanShift clustering aims to discover blobs in a smooth density of samples.
    It is a centroid based algorithm, which works by updating candidates for centroids
    to be the mean of the points within a given region. These candidates are then filtered
    in a post-processing stage to eliminate near-duplicates to form the final set of centroids.

    Labelling a new sample is performed by finding the nearest centroid for a given sample.
    """

    def __init__(self):
        """
        self.clustermodel : sklearn.cluster.MeanShift
            Mean shift clustering using a flat kernel.

        self.clusters : Array of all calculated clusters

        NOTE: access to cluster centers with self.clustermodel.cluster_centers_

        """
        self.clustermodel = cluster.MeanShift(bandwidth=5, bin_seeding=True, max_iter=1)
        self.clusters = {}

    def choose_action(self, view: np.ndarray):
        """
        Call a function to a predict an action using
        a raycast observation and a ML model

        Parameters
        ---------
        view: numpy.ndarray
            The view of the fish of lenght n + k,
            consists of n_fish_bin values for fish
            and k_wall_raycasts for the walls (here 5).

        Returns
        ---------
        speed, turn : float, float
            Using a raycast observation and a ML model
            the speed [cm/s] and turn [rad/s] is returned
        """
        action = self.predict_action(view)
        return action[0][0], action[0][1]

    def predict_action(self, view: np.ndarray):
        """
        Measures the distance between data points to determine how data points are close.

        Parameters
        ---------
        view: numpy.ndarray
            The view of the fish of lenght n + k,
            consists of n_fish_bin values for fish
            and k_wall_raycasts for the walls (here 5).

        Returns
        ---------
        Tuple[float, float]
            Using a raycast observation and a ML model returns
            the Tuple of speed [cm/s] and turn [rad/s]
        """
        clustermodel = self.clustermodel
        clusters = self.clusters

        # clustermodel.predict([view]) returns the index for list clustermodel.cluster_centers_
        prediction = clustermodel.cluster_centers_[clustermodel.predict([view])][0]
        choice = random.sample(list(clusters[str(prediction)]), 1)
        return choice

    def train(self, dset):
        """
        Perform Mean shift clustering.
        Calculates and saves cluster_centers into self.clustermodel.cluster_centers_
        to be used by a predict-function

        Parameters
        ---------
        dset: array-like of shape (n_samples, n_features)
            Samples to cluster.

        Returns
        ---------
        None

        """

        clustermodel = self.clustermodel
        clusters = self.clusters

        actions = dset[:]["actions"]
        views = dset[:]["views"]

        # Train
        clustermodel.fit(views)

        # collect clustercenters as dictionary
        for center in clustermodel.cluster_centers_:
            clusters[str(center)] = []

        # assign data to clustercenters
        for point in dset:
            prediction = clustermodel.cluster_centers_[
                clustermodel.predict([point["views"]])
            ][0]
            clusters[str(prediction)] += [point["actions"]]
