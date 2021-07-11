import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import os
import time
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import grad
from torch.nn import Parameter
cuda = torch.cuda.is_available()
import pickle as pk
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision.transforms import transforms
from typing import Union,Tuple
import fish_models
from fish_models import utils
import robofish.io

# Test training
n_training_iterations = 10
n_files = 5
n_timesteps = None
n_speed_bins = 21
n_turn_bins = 21
n_view_bins = 5
n_timesteps_simulation = 2000
batch_size = 64
n_epochs = 50
fishes = 2

# IO Files in cm, actions in m/s
data_folder = Path("data/live_female_female/train")
test_data_folder = Path("data/live_female_female/test")

"""
This model uses a deep convolutional neural network in order to later predict 
a speed/turn bin for a given view.
"""


class ResNetBlock(nn.Module):
    def __init__(self):
        """
        ResNet inner block with skip connection (residual blocks)
        Residual blocks allow the flow of memory (or information) from initial layers to last layers.
        """
        super(ResNetBlock, self).__init__()
        self.kernel_size = 3

        # structure
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=self.kernel_size, stride=2, padding=6)
        self.relu4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(16)

        # 1x1 conv filters can be used to change the dimensionality in the filter space.
        self.identity_upsample_first = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32))
        self.identity_upsample_second = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16))

    def forward(self, x):
        """
        The forward function computes output Tensors from input Tensors.
        This block used multiply times
        Created building block of a ResNet called a residual block or identity block.

        Parameters
        ---------
        x:
            input Tensor
        """
        # save first identity
        identity1 = x.clone()

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        # sum point
        # 1. Upsample identity size
        identity1 = self.identity_upsample_first(identity1)
        # 2. Summarize (skip connection)
        x = x + identity1
        # 3. save second identity and change size to be able to summurize in the next sum point
        identity_upsampled = self.identity_upsample_second(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        # sum point
        x = x + identity_upsampled
        return x


class ResNet(nn.Module):
    # layers is the list telling us how much to reuse the block in each block
    def __init__(self, block, N, batch_size):
        """
        ResNet is a short name for Residual Network.
        A deep convolutional neural network,
        Several layers are stacked and are trained to the task at hand.
        The network learns several low/mid/high level features at the end of its layers.

        The main innovation of ResNet is the skip connection.
        Deep networks often suffer from vanishing gradients,
        ie: as the model backpropagates, the gradient gets smaller and smaller.

        This allows to stack additional layers and build a deeper network.

        Parameters
        ---------
        block:
            nn.Module for inner ResNet structure
        N:
            Number of inner blocks of deep model, default = 1
        batch_size:
            the additional dimension for batching

        """
        super(ResNet, self).__init__()
        self.in_channels = 1
        self.kernel_size = 3
        self.start_identity = None
        self.batch_small = 2
        self.N = N
        self.batch_size = batch_size

        # Begin
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        # nn.BatchNorm2d 4 expects 4D inputs in shape of [batch, channel, height, width]
        self.bn1 = nn.BatchNorm1d(16)

        # ResNet Layers
        # repeat 3 times for i in range(2)
        # out_channel is equal to 16 * 2 * i => 16*2^0, 16*2^1, 16*2^2 => 16, 32 and 64 out_channels
        self.layer1 = self.make_layer(block, self.N, out_channels=2)
        self.layer2 = self.make_layer(block, self.N, out_channels=4)
        self.layer3 = self.make_layer(block, self.N, out_channels=8)

        # End
        self.conv_end = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.flt_end = nn.Flatten()
        self.fc_end = nn.Linear(in_features=10, out_features=n_speed_bins + n_turn_bins)
        self.double()

    def forward(self, x):
        """
        The forward function computes output Tensors from input Tensors.
        The implementation of this function allows to make a custom calculation
        when passing Tensors containing input data

        Parameters
        ---------
        x:
            input Tensor
        """
        # Begin
        # [batch_size, seq_length] as input data -> add the channel dimension using unsqueeze

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        # ResNet Layers forward step
        # loops inside every layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # end of inner blocks

        # Tail of the ResNet
        x = self.conv_end(x)
        x = self.flt_end(x)
        x = self.fc_end(x)

        return x

    def make_layer(self, block, num_residual_blocks, out_channels):
        """
        Builds a repetitive blocks with increasing number of channels

        Parameters
        ---------
        x:
            input Tensor
        """

        identity_downsample = None
        layers = []

        print(f'Extern loop START')
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels * 2

        # intern loops for layers and skip connections
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            print(f'Inner layer {n} is created!')

        print(f'Extern loop END')
        return nn.Sequential(*layers)
