"""
This is where we implement all the neural networks needed for MAPPO
this includes
1. Policy network that takes in the RGBD and output 12 dim: 3 for position and 9 for rotation matrix for end-effector

"""

import os.path

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from variables import CUR_STATES, ACTIONS


import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class RGBDNetwork(nn.Module):
    def __init__(self, input_height=224, input_width=600, output_dim=512):
        """
        RGBD Neural Network

        Args:
            input_height (int): Height of input images
            input_width (int): Width of input images
            output_dim (int): Dimension of output feature vector
        """
        super(RGBDNetwork, self).__init__()

        print(f"initiate RGBD network with input_height={input_height}, input_width={input_width}, output_dim={output_dim}")

        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim

        # Convolutional layers for feature extraction
        # Input: 4 channels (RGB + Depth)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after convolutions
        self._calculate_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, output_dim)

        self.to(device)

    def _calculate_conv_output_size(self):
        """Calculate the output size after all conv layers"""
        with torch.no_grad():
            x = torch.zeros(1, 4, self.input_height, self.input_width)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool3(F.relu(self.bn4(self.conv4(x))))
            self.conv_output_size = x.numel()

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4, height, width)
                             Channels: [R, G, B, D] where D is depth

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(F.relu(self.bn4(self.conv4(x))))

        print(f"x after convolution {x.shape}")

        # print(f"expected inputs {self.conv_output_size}")

        # Flatten for fully connected layers
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def process_inputs(self, batch):
        """
        a batch is made up of dictionary
        each dict is made up of a cur_state which is RGBD and a (12,) action
        need to forward the RGBD, then concate them
        """

        # Single RGBD array
        if batch.ndim == 3:
            # If shape is (H, W, 4), add batch dimension and permute to (1, 4, H, W)
            if batch.shape[2] == 4:
                RGBD_inputs = torch.tensor(batch.astype(np.float32), device=device).unsqueeze(0).permute(0, 3, 1, 2)
            # If shape is (4, H, W), add batch dimension to get (1, 4, H, W)
            elif batch.shape[0] == 4:
                RGBD_inputs = torch.tensor(batch.astype(np.float32), device=device).unsqueeze(0)
        else:
            RGBD_inputs = torch.tensor(np.array([np.array(transition[CUR_STATES]) for transition in batch])
                                       .astype(np.float32),
                                       device=device).permute(0, 3, 1, 2)

        # Input: [64, 4, 480, 1280]
        # Output: [64, 4, 224, 600]
        RGBD_inputs = F.interpolate(
            RGBD_inputs,
            size=(self.input_height, self.input_width),
            mode='bilinear',
            align_corners=False
        )

        # action_inputs = torch.tensor(np.array([np.array(transition[ACTIONS]) for transition in batch]).astype(np.float32),
        #                            device=device)

        # print(action_inputs)

        # print(f"before processing: {RGBD_inputs.shape}")

        processed_RGBD = self.forward(RGBD_inputs)

        # processed_RGBD = torch.cat([processed_RGBD, action_inputs], dim=1)

        # print(f"after processing: {processed_RGBD.shape}")

        return processed_RGBD



    def save_weights(self, weight_name="rgbdNet_weights.pth"):
        torch.save(self.state_dict(), weight_name)

    def load_weights(self, weight_path):
        if not os.path.exists(weight_path):
            print(f"weight path is invalid {weight_path}")
            return False
        self.load_state_dict(torch.load(weight_path))


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):

        print(f"policy network initiation: input dim {n_states}, output dim {n_actions}")

        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

        self.to(device)

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    # def sample_or_likelihood(self, states):
    #     dist = self(states)
    #     # Reparameterization trick
    #     u = dist.rsample()
    #     action = torch.tanh(u)
    #     log_prob = dist.log_prob(value=u)
    #     # Enforcing action bounds
    #     log_prob -= torch.log(1 - action ** 2 + 1e-6)
    #     log_prob = log_prob.sum(-1, keepdim=True)
    #     return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob

    def sample_or_likelihood(self, states):
        # Get distribution from forward pass
        dist = self(states)

        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)

        # Calculate log probability
        log_prob = dist.log_prob(value=u)

        # Squash correction for tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        # Ensure action_bounds is on the same device as action
        if isinstance(self.action_bounds, (np.ndarray, list)):
            self.action_bounds = torch.tensor(self.action_bounds, dtype=torch.float32).to(states.device)
        elif isinstance(self.action_bounds, torch.Tensor) and self.action_bounds.device != states.device:
            self.action_bounds = self.action_bounds.to(states.device)

        # Split action into two parts: first 3 (no limits) and last 9 (rotation matrix)
        action_first_3 = u[:, :3]  # First 3 dimensions - no bounds
        action_rotation = action[:, 3:]  # Last 9 dimensions - rotation matrix elements

        # Scale first 3 dimensions normally
        # scaled_first_3 = (action_first_3 * self.action_bounds[1][:3]).clamp(
        #     min=self.action_bounds[0][:3],
        #     max=self.action_bounds[1][:3]
        # )

        # For rotation matrix elements, keep them in [-1, 1] range
        # Since tanh already outputs [-1, 1], we can use them directly
        # or apply a smaller scaling if needed
        # scaled_rotation = action_rotation  # Already in [-1, 1] from tanh

        # Optionally, you can enforce stricter bounds for rotation matrix
        scaled_rotation = action_rotation.clamp(min=-1.0, max=1.0)

        # Combine both parts
        scaled_action = torch.cat([action_first_3, scaled_rotation], dim=1)

        return scaled_action, log_prob

    def save_weights(self, weight_name="policyNet_weights.pth"):
        torch.save(self.state_dict(), weight_name)

    def load_weights(self, weight_path):
        if not os.path.exists(weight_path):
            print(f"weight path is invalid {weight_path}")
            return False
        self.load_state_dict(torch.load(weight_path))


