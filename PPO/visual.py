"""
code here helps to visualise the data
"""
import cv2
import os.path

import torch

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import pearsonr

from data import load_array_from_file

def display_image(image_array, window_name="Image", wait_time=0):
    """
    Display an image array using OpenCV

    Args:
        image_array: numpy array representing the image
        window_name: name of the display window
        wait_time: time to wait (0 = wait for key press, >0 = milliseconds)
    """
    # Display the image
    cv2.imshow(window_name, image_array)

    # Wait for key press or specified time
    cv2.waitKey(wait_time)

    # Clean up (optional - removes the window)
    cv2.destroyAllWindows()



"""
Reinforcement Learning Visualization and Evaluation
--------------------------------------------------
This module provides utilities for:
1. Visualizing training metrics from saved data files
2. Evaluating trained policies in a rendered environment
3. Displaying performance metrics

The visualization functions help analyze the training progress by plotting
various metrics like policy loss, value loss, and rewards over time.

The simulation function allows for visual evaluation of the trained policy
by loading model weights and running the policy in a rendered environment.
"""


def plot_progress_data(data, save_plot=False, plot_file_title=None):
    # After training is complete, plot the loss graph

    data = np.array(data)

    if data.ndim < 2:
        data = [data, ]

    plt.figure(figsize=(10, 6))

    for cur_data in data:
        plt.plot(cur_data)
        plt.legend()
    plt.title(f'{plot_file_title} over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)


    if save_plot:
        if plot_file_title is None:
            plot_file_title = "plot.png"
        plt.savefig(plot_file_title)
    plt.show()


def plot_correlation_graph(rewards, q_totals, save_plot=False, plot_file_title=None):

    if not isinstance(rewards, np.ndarray):
        rewards = np.array(rewards)

    if not isinstance(q_totals, np.ndarray):
        q_totals = np.array(q_totals)

    print(f"reward size {rewards.shape} and q_total size {q_totals.shape}")

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(rewards, q_totals, alpha=0.5)
    plt.xlabel('Actual Rewards')
    plt.ylabel('Predicted Q-total')
    plt.title('Alignment between Rewards and Q-total Values')

    # Add trend line
    z = np.polyfit(rewards, q_totals, 1)
    p = np.poly1d(z)
    plt.plot(rewards, p(rewards), "r--", alpha=0.8)

    # Calculate correlation coefficient
    corr, p_value = pearsonr(rewards, q_totals)
    plt.annotate(f"Correlation: {corr:.3f} (p={p_value:.3f})",
                 xy=(0.05, 0.95), xycoords='axes fraction')

    plt.grid(True)

    if save_plot:
        if plot_file_title is None:
            plot_file_title = "plot.png"
        plt.savefig(plot_file_title)

    plt.show()

if __name__ == "__main__":

    # reward_arr = load_array_from_file("./demo_reward_overtime.txt")
    # phase_arr = load_array_from_file("./phase.txt")
    # box_arr = load_array_from_file("./boxes_in_bin.txt")

    reward_arr = load_array_from_file("./reward_plot.txt")
    policy_arr = load_array_from_file("./reward_plot.txt")
    value_arr = load_array_from_file("./va.txt")

    plot_progress_data([reward_arr, phase_arr, box_arr])
