


"""
This is where we conduct training of the PPO agent.
1. direct training with randomly initiated PPO agent
2. perform imitation learning, then use PPO
"""

import numpy as np
import torch
import torch.nn.functional as F

from neural_networks import ValueNetwork, PolicyNetwork
from dataCollector import collect_buffer_data, generate_PPO_training_batch, get_reward
from data import append_values_to_file
from visual import plot_progress_data
from env import FinalEnv
from tqdm import tqdm



def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, log_probs_old, advantages,
               returns, clip_ratio=0.2, epochs=10):
    for _ in range(epochs):
        # Get current policy evaluation
        log_probs, entropy = policy_net.evaluate(states, actions)

        # Compute policy ratio (π_θ / π_θ_old)
        ratio = torch.exp(log_probs - log_probs_old)

        # Compute clipped objective
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

        # Add entropy bonus
        policy_loss = policy_loss - 0.01 * entropy.mean()

        # Update policy
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Update value function
        value_pred = value_net(states)
        value_loss = F.mse_loss(value_pred, returns)

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()


def single_epoch_update(policy_net: PolicyNetwork, value_net: ValueNetwork, batch_size=64, data_folder_path="./data/"):
    """
    In each epoch we aim the following things
    1. collect data using the policy network and the replayBuffer
    2. random sample training batches to train both network till convergence.
    :param batch_size: batch size used from random sample to train network
    :param envs: initiated environments for simulation
    :param buffer:
    :param policy_net:
    :param value_net:
    :return:
    """

    """
    collect data for using policy and value network with the initiated envs
    The trajectory tensors contain keys:

    states: Environment states at each timestep, initially shaped [timesteps, trajectories, state_dim]
    actions: Actions taken by the policy, initially shaped [timesteps, trajectories, action_dim]
    rewards: Rewards received after each action, initially shaped [timesteps, trajectories]
    old_probs: Action probabilities from the policy that generated the data, initially shaped [timesteps, trajectories]
    next_states: States after taking actions, initially shaped [timesteps, trajectories, state_dim]
    dones: Boolean flags indicating terminal states, initially shaped [timesteps, trajectories]
    advantages: Computed advantage estimates using GAE, likely shaped [timesteps, trajectories]
    returns: Computed returns (discounted sum of rewards), likely shaped [timesteps, trajectories]
    """
    # trajectory_tensor = generate_PPO_training_batch(data_folder_path, policy_net, value_net, batch_size=batch_size)

    policy_loss = None

    policy_loss_arr = []
    value_loss_arr = []

    convergence_threshold = 1e-4


    for _ in range(10):
        # extract a random batch of data for training.
        batch_training_data_tensor = generate_PPO_training_batch(data_folder_path, policy_net,
                                                                 value_net, batch_size=batch_size)

        # policy network learning
        policy_loss = policy_net.get_loss(batch_training_data_tensor)
        policy_net.optimizer.zero_grad()
        policy_loss.backward()
        policy_net.optimizer.step()

        # value network learning
        value_loss = value_net.get_loss(batch_training_data_tensor)
        value_net.optimizer.zero_grad()
        value_loss.backward()
        value_net.optimizer.step()


        # save the loss value for tracking
        policy_loss_arr.append(policy_loss.detach().item())
        value_loss_arr.append(value_loss.detach().item())

    append_values_to_file(policy_loss_arr, "./policy_loss.txt")
    # plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="policy_loss")

    append_values_to_file(value_loss_arr, "./value_loss.txt")
    # plot_progress_data(value_loss_arr, save_plot=True, plot_file_title="value_loss")

if __name__ == "__main__":

    datapath = "./data/"

    # Create the environment
    env = FinalEnv()

    # define variables
    input_height = 224
    input_width = 300
    output_dim = 512
    action_dim = 24
    interval = 1
    action_bounds = [-1, 1]

    # self.rgbd_nn = RGBDNetwork(input_height, input_width, output_dim)
    pNet = PolicyNetwork(n_states=output_dim,
                                   n_actions=action_dim,
                                   input_height=input_height,
                                   input_width=input_width,
                                   action_bounds=action_bounds)

    pNet.load_weights("./Policy_nn_weight.pth")
    vNet = ValueNetwork()
    vNet.load_weights("./Value_nn_weight.pth")

    trajectory_num = 0
    reward_arr = []

    time_step = 0


    while True:
        
        trajectory_num = collect_buffer_data(env, trajectory_num, total_trajectory_num=5, buffer_size=100, folder_path=datapath)

        if time_step % 100 == 0:
            # every 100 timestep we save the weights
            pNet.save_weights()
            vNet.save_weights()
            append_values_to_file(reward_arr, "./reward_plot.txt")
            reward_arr = []

        single_epoch_update(pNet, vNet, batch_size=64)

        reward = get_reward(env, "./Policy_nn_weight.pth")

        reward_arr.append(reward)

        time_step += 1

        print(f"current time_step {time_step}, modal reward {reward}")

