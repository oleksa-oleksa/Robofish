import fish_models
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random


class RandomForestFishModel(fish_models.gym_interface.AbstractRaycastBasedModel):
    def __init__(self):
        """
        Parameters
        ---------
        n_estimators : int, default=100
            The number of trees in the forest.
        max_depth : int, default=None
            The maximum depth of the tree.
            If None, then nodes are expanded until all leaves are pure
            or until all leaves contain less than min_samples_split samples.
        random_state : int, RandomState instance or None, default=None
            Controls both the randomness of the bootstrapping
            of the samples used when building trees

        """
        self.clf = RandomForestRegressor(
            n_estimators=255, max_depth=None, random_state=None
        )

    def choose_action(self, view: np.ndarray):
        """
        Predict regression target for given view of a virtual fish.
        The predicted regression target of an input sample is computed
        as the mean predicted regression targets of the trees in the forest.
        """

        prediction = self.clf.predict([view])

        speed = prediction[0][0]
        turn = prediction[0][1]

        return speed, turn

    def fit(self, dset):
        """
        Build a forest of trees from the given fish dset

        Dset will be devided into the training input samples aka views
        and target values aka tuple of actions as real numbers in regression
        """

        views = dset[:]["views"]
        actions = dset[:]["actions"]

        # Fit
        self.clf.fit(views, actions)

        print("Fit done")
