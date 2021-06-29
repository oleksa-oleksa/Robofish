import fish_models
import numpy as np


class KNNFishModel(fish_models.gym_interface.AbstractRaycastBasedModel):
    """
    kNN classifier determines the class of a data point by majority voting principle.

    If k is set to n, the classes of n closest points are checked.
    Prediction is done according to the majority class.
    Similarly, kNN regression takes the mean value of n closest points.
    """

    def __init__(self, k):
        self.k = k

    def choose_action(self, view: np.ndarray):

        speed, turn = self.predict(view, self.k)

        return speed, turn

    def euclidean_distance(self, x_1, x_2):
        """
        Measures the distance between data points to determine how data points are close.
        Euclidean distance is used for distance measurement and it is calculated
        using the square of the difference between x and y coordinates of the points.

        Parameters
        ---------
        x1, x2 : array_like
            Elements to find the distance between.

        Returns
        ---------
        distance : ndarray
            Distance scalar
        """
        return np.sum((x_1 - x_2) ** 2, axis=1)

    def fit(self, dset):
        """
        Storing the training set's speed for separate class predictions.
        """
        self.X = dset[:]["views"]
        self.y = dset[:]["actions"]

    def predict(self, view, k):

        """
        Predict the class of a data point by majority voting principle.

        Parameters
        ---------
        view : array_like
            Test or created by TrackGeneratorGymRaycast datapoint

        k : int
            The number of k nearest neighbors to be used to determine a class of test datapoint

        Returns
        ---------
        predictions : list
            predicted speed/turn as the mean value of k closest points
        """
        # getting distance for each prticular data point
        distances = self.euclidean_distance(self.X, view)
        # getting indexes of k first minimal elements
        idx = np.argpartition(distances, self.k)[: self.k]
        # taking labels (views) by indexes
        votes = self.y[idx]
        # mean for speed and turn
        prediction = np.mean(votes, axis=0)
        return prediction[0], prediction[1]
