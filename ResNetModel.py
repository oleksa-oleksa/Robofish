import fish_models
import fish_models.datasets.io_dataset as io_dataset
import numpy as np
import sys
from pathlib import Path
import time
from typing import Union, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pickle

"""
Idea of this model is to train a small CNN in order to later predict a speed/turn bin for a given view.
With this prediction we then can pick a random sample of the predicted bin.
"""


class CNNModel(nn.Module):
    def __init__(
        self,
        n_fish_bins: int = 3,
        n_wall_bins: int = 3,
        n_speed_bins: int = 21,
        n_turn_bins: int = 21,
        savemodel: bool = False,
        path=Path("output/stochastic_CNN_model"),
    ):
        super(CNNModel, self).__init__()
        """
        The construcor provides the structure of our neural network.
        The Structure of our network for example with 
        n_fish_bins = n_wall_bins = 31 
        and 
        n_speed_bins = n_turn_bins = 63 
        looks like:
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Linear-1               [-1, 1, 93]            5,859
                    Conv1d-2               [-1, 16, 93]              64
                    LeakyReLU-3            [-1, 16, 93]               0
                    BatchNorm1d-4          [-1, 16, 93]              32
                    Conv1d-5               [-1, 32, 93]           1,568
                    LeakyReLU-6            [-1, 32, 93]               0
                    BatchNorm1d-7          [-1, 32, 93]              64
                    Conv1d-8               [-1, 16, 93]           1,552
                    LeakyReLU-9            [-1, 16, 93]               0
                    BatchNorm1d-10         [-1, 16, 93]              32
                    Conv1d-11              [-1, 1, 93]               49
                    Flatten-12             [-1, 93]                   0
                    Linear-13              [-1, 126]             11,844
        ================================================================
        Total params: 21,064
        Trainable params: 21,064
        Non-trainable params: 0
        
        The sizes of each layer vary based on the sizes of all bins.

        Parameters
        ---------
        n_fish_bins : positive integer
            number of values in a given view, that represent 'fishbins'
        n_wall_bins : positive integer
            number of values in a given view, that represent 'wallbins'
        n_speed_bins : positive integer
            number of speed bins
        n_turn_bins : positive integer
            number of turn bins
        savemodel : Boolean
            decides, wether the progress of training shall be saved every epoch
            change self.path to your desired path
        """
        (
            self.n_fish_bins,
            self.n_wall_bins,
            self.n_speed_bins,
            self.n_turn_bins,
            self.savemodel,
            self.path,
        ) = (n_fish_bins, n_wall_bins, n_speed_bins, n_turn_bins, savemodel, path)
        n_view_bins = n_fish_bins + n_wall_bins

        self.losses = []
        self.mean_losses = []
        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(
                            in_features=n_view_bins,
                            out_features=3 * n_view_bins,
                        ),
                    ),
                    (
                        "conv1",
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu1", nn.LeakyReLU()),
                    ("bn1", nn.BatchNorm1d(16)),
                    (
                        "conv2",
                        nn.Conv1d(
                            in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu2", nn.LeakyReLU()),
                    ("bn2", nn.BatchNorm1d(32)),
                    (
                        "conv3",
                        nn.Conv1d(
                            in_channels=32,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu3", nn.LeakyReLU()),
                    ("bn3", nn.BatchNorm1d(16)),
                    (
                        "conv4",
                        nn.Conv1d(
                            in_channels=16,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("flt1", nn.Flatten()),
                    (
                        "fc2",
                        nn.Linear(
                            in_features=3 * n_view_bins,
                            out_features=n_speed_bins + n_turn_bins,
                        ),
                    ),
                ]
            )
        )
        self.double()

    def forward(self, x: np.array):
        """
        Parameters
        ---------
        x : np.array
            view data we want to process

        Returns
        ---------
        Arrays of Propabilities for speed bins and turn bins concatenated

        Example:
        For n_fish_bins = n_wall_bins = 3 and n_speed_bins = n_turn_bins = 2
        The input looks like [0.2,0.9,0,0,0.4,0.2]
            where fish_bins = (0.2,0.9,0) and wall_bins = (0,0.4,0.2)
        and the output looks like [0.8,0.2,-0.1,0.7]
            where the propabilities for speed_bins = (0.8,0.2) and turn_bins = (-0.1,0.7).
            Note that these are not yet normalized and will be done so later on.
        """
        b, c, *_ = x.shape
        x = x.reshape([b, 1, self.n_wall_bins + self.n_fish_bins])

        return self.body(x)

    def fit(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        data: Union[DataLoader, Tuple[DataLoader]],
        max_epochs: int,
        cuda: bool = False,
    ):
        """
        Here happens the training of our neural network.

        Parameters
        ---------
        model: nn.Module
            The pytorch model that is to be trained.
        optimizer: optim.Optimizer
            The optimizer used in training.
        data: Union[DataLoader, Tuple[DataLoader]]
            Two dataloaders for training and test data.
        max_epochs: int
            Number of training epochs to go through.
        cuda : bool = False
            Decides wether the training shall run on a gpu, if possible.

        Returns
        ---------
        losses : array
            Losses for each batch went through.
        mean_losses : array
            Mean losses for every epoch
        Both are stored as attributes CNNStochasticmodel
        """

        # Check if we have a data and testloader
        use_test = False
        if isinstance(data, DataLoader):
            train_loader = data
        elif isinstance(data, tuple):
            if len(data) == 2:
                train_loader, test_loader = data
                if not isinstance(train_loader, DataLoader):
                    raise TypeError(
                        f"Expected 1st entry DataLoader, but got {type(train_loader)}!"
                    )
                if not isinstance(test_loader, DataLoader):
                    raise TypeError(
                        f"Expected 2nd entry DataLoader, but got {type(test_loader)}!"
                    )
                use_test = True
            else:
                raise ValueError(f"Expected tuple of length 2, but got {len(data)}!")
        # For our loss we compute crossentropyloss and kullback-leibler divergence and add them up.
        logmax = nn.LogSoftmax()
        kld = nn.KLDivLoss()
        criterion = nn.CrossEntropyLoss()
        model.train()
        losses = []
        mean_losses = []
        n_speed_bins = self.n_speed_bins
        n_turn_bins = self.n_turn_bins

        batch_total = len(train_loader)
        for epoch in range(max_epochs):
            samples_total = 0
            samples_correct = 0
            test_samples_total = 0
            test_samples_correct = 0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = batch

                if cuda:
                    x, y = x.cuda(), y.cuda()

                output = model(x)
                # Compute logpropabilites from output of our network
                log_output1 = logmax(output[:, :n_speed_bins])
                log_output2 = logmax(output[:, n_speed_bins:])
                # Compute loss for crossentropyloss and kullback-leibler divergence
                # for speed and turn bins each (we have to split up output and labels,
                # see datastructure in docstring for forward function)
                loss = (
                    criterion(output[:, :n_speed_bins], y[:, 0])
                    + criterion(output[:, n_speed_bins:], y[:, 1])
                    + kld(
                        log_output1,
                        nn.functional.one_hot(y[:, 0], n_speed_bins).double(),
                    )
                    + kld(
                        log_output2,
                        nn.functional.one_hot(y[:, 1], n_turn_bins).double(),
                    )
                )
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                mean_losses.append(np.mean(losses))

                if use_test:
                    # Same procedure for testdata
                    model.eval()
                    batch = next(iter(test_loader))
                    test_x, test_y = batch
                    if cuda:
                        test_x, test_y = test_x.cuda(), test_y.cuda()

                    test_output = model(test_x)
                    test_log_output1 = logmax(test_output[:, :n_speed_bins])
                    test_log_output2 = logmax(test_output[:, n_speed_bins:])
                    test_loss = (
                        criterion(test_output[:, :n_speed_bins], test_y[:, 0])
                        + criterion(test_output[:, n_speed_bins:], test_y[:, 1])
                        + kld(
                            test_log_output1,
                            nn.functional.one_hot(test_y[:, 0], n_speed_bins).double(),
                        )
                        + kld(
                            test_log_output2,
                            nn.functional.one_hot(test_y[:, 1], n_turn_bins).double(),
                        )
                    )

                    model.train()

                    sys.stdout.write(
                        f"\rEpoch: {epoch:2}/{max_epochs:2} Step: {batch_idx:2}/{batch_total:2} Loss: {torch.mean(torch.tensor(losses)):10.6f} Test loss: {torch.mean(torch.tensor(test_loss)):10.6f}"
                    )
                else:
                    sys.stdout.write(
                        f"\rEpoch: {epoch:2}/{max_epochs:2} Step: {batch_idx:2}/{batch_total:2} Loss: {loss.item():10.6f} "
                    )

            if self.savemodel:
                self.action_savemodel()

            # Deposit losses and mean_losses as attributes in CNNStochasticmodel so they can be accesed
            # even after incomplete training.
            self.losses = losses
            self.mean_losses = mean_losses
        return losses, mean_losses

    def action_savemodel(self):
        """
        Saves this class in a file so progress in training isn't lost.
        """
        with open(self.path, "wb") as file:
            pickle.dump(self, file)


class CNNStochasticmodel(fish_models.gym_interface.AbstractModel):
    def __init__(
        self,
        speed_bins,
        turn_bins,
        n_fish_bins=3,
        n_wall_bins=3,
        n_speed_bins=21,
        n_turn_bins=21,
        savemodel=False,
        path=Path("output/stochastic_CNN_model"),
    ):
        """A stochastic model using discrete actions and a sklearn MLPClassifier.

        Parameters
        ---------
        Most of these parameters will be given to the constructor of our neural network.
        speed_bins :
            The array with the float borders between speed bins
        turn_bins :
            The array with the float borders between turn bins
        n_fish_bins : positive integer
            number of values in a given view, that represent 'fishbins'
        n_wall_bins : positive integer
            number of values in a given view, that represent 'wallbins'
        n_speed_bins : positive integer
            number of speed bins
        n_turn_bins : positive integer
            number of turn bins
        savemodel : Boolean
            decides, wether the progress of training shall be saved every epoch
            change self.path to your desired path
        """

        self.speed_bins = speed_bins
        self.turn_bins = turn_bins

        self.clf = CNNModel(
            n_fish_bins=n_fish_bins,
            n_wall_bins=n_wall_bins,
            n_speed_bins=n_speed_bins,
            n_turn_bins=n_turn_bins,
            savemodel=savemodel,
            path=path,
        )

    def predict_proba(self, view: np.ndarray):
        """
        Returns the output of our neural network for a given input.

        Parameters
        ---------
        view : np.array
            view data we want to process.

        Returns
        ---------
        Arrays of Propabilities for speed bins and turn bins concatenated

        Example:
        For n_fish_bins = n_wall_bins = 3 and n_speed_bins = n_turn_bins = 2
        The input looks like [0.2,0.9,0,0,0.4,0.2]
            where fish_bins = (0.2,0.9,0) and wall_bins = (0,0.4,0.2)
        and the output looks like [0.8,0.2,-0.1,0.7]
            where the propabilities for speed_bins = (0.8,0.2) and turn_bins = (-0.1,0.7).
            Note that these are not yet normalized and will be done so later on.
        """
        output = self.clf(torch.tensor(np.array([view]).astype(np.double)))
        return output.detach().numpy()

    def loadmodel(self):
        """
        Loads the model from a file so progress in training can be retained.
        """
        with open(Path("output/stochastic_CNN_model"), "rb") as file:
            self.clf = pickle.load(file)

    def train(
        self,
        dset,
        test_dset,
        n_training_iterations=200,
        train_batch_size=64,
        test_batch_size=16,
        learning_rate=1e-3,
    ):
        """
        Train the classifier

        Parameters
        ---------
        dset : fish_models.datasets.io_dataset.IoDataset:
            An IoDataset with at least output_strings ["actions_binned", "views"]
        test_dset : fish_models.datasets.io_dataset.IoDataset:
            An IoDataset with at least output_strings ["actions_binned", "views"]
        n_training_iterations : positive int
            Number of epochs to go through in training.
        train_batch_size : int
            Batchsize for trainingdataloader.
        test_batch_size : int
            Batchsize for testdataloader.
        learning_rate : float
            Learning rate for optimizer.
        """
        # self.poses_storage = []

        train_loader = torch.utils.data.DataLoader(
            dset,
            collate_fn=io_dataset.IODatasetPytorchDataloaderCollateFN(
                ["views", "actions_binned"], [torch.float64, torch.long]
            ),
            batch_size=train_batch_size,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dset,
            collate_fn=io_dataset.IODatasetPytorchDataloaderCollateFN(
                ["views", "actions_binned"], [torch.float64, torch.long]
            ),
            batch_size=test_batch_size,
        )

        start = time.time()

        fc_optim = optim.Adam(self.clf.parameters(), lr=learning_rate)
        self.fc_losses, self.mean_losses = self.clf.fit(
            self.clf, fc_optim, (train_loader, test_loader), n_training_iterations
        )

        end = time.time()

        print(f"\nTraining took {end-start}s!")

    def choose_action(self, view: np.ndarray) -> Tuple[float, float]:
        """
        Parameters
        ---------
        view : np.ndarray
            view data we want to process.

        Returns
        ---------
        Tuple of speed and turn values.
        """
        probabilities = self.predict_proba([view])
        speed_probabilities = probabilities[0, : len(self.speed_bins)]
        turn_probabilities = probabilities[0, len(self.speed_bins) :]

        # Shift ouput of our value so that no negative values exist.
        speed_probabilities += np.abs(np.min(speed_probabilities)) + 0.000001
        turn_probabilities += np.abs(np.min(turn_probabilities)) + 0.000001

        # Normalize
        speed_probabilities /= np.sum(speed_probabilities)
        turn_probabilities /= np.sum(turn_probabilities)

        speed = np.random.choice(self.speed_bins, p=speed_probabilities)
        turn = np.random.choice(self.turn_bins, p=turn_probabilities)

        # Double sampling (sample inside the chosen bin)
        speed += np.random.random() * np.diff(self.speed_bins)[0]
        turn += np.random.random() * np.diff(self.turn_bins)[0]

        return [speed, turn]
