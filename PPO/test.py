"""
For testing functions
"""

import os
import cv2
import numpy as np
import torch

from tqdm import tqdm

from neural_networks import ValueNetwork
from data import load_array_from_file


def get_TD_error(valueNet: ValueNetwork, cur_states_folder_path, reward_arr, discount_factor=0.9):
    """
    Path to all the states stored as images. We use the valueNetwork to generate the TD error.
    1. go through the states, one by one load the states according to timestep sequence
    2. generate the Value estimates
    3. TD error δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
    """

    images_filenames = sorted(os.listdir(cur_states_folder_path), key=lambda x: int(x.split('.')[0]))

    value_est_arr = []

    for i in tqdm(range(len(images_filenames)), desc="generate value est"):
        img_filename = images_filenames[i]
        img_filepath = os.path.join(cur_states_folder_path, img_filename)
        state = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED) # make sure load all 4 channels
        # print(state.shape)
        with torch.no_grad():
            value_est = valueNet.forward(state)
            value_est_arr.append(value_est.item())

    # convert to np for efficient calculation
    # remove last reward as it will not be part of a transition
    reward_arr_np = np.array(reward_arr[:-1])
    cur_state_value_est_np = np.array(value_est_arr[:-1])
    next_state_value_est_np = np.array(value_est_arr[1:])

    TD_error_np = reward_arr_np + discount_factor * next_state_value_est_np - cur_state_value_est_np
    print(TD_error_np)

    return TD_error_np


if __name__ == "__main__":


    r_arr = load_array_from_file("./data/0/reward.txt")

    image = cv2.imread("./data/0/cur_states/0.png", cv2.IMREAD_UNCHANGED)
    vNet = ValueNetwork()

    get_TD_error(vNet, "./data/0/cur_states", r_arr)

