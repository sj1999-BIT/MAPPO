"""
This is where we implement all the neural networks needed for MAPPO
this includes
1. Policy network that takes in the RGBD and output 12 dim: 3 for position and 9 for rotation matrix for end-effector

"""

import os.path
import torch
import torch.optim as optim


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
    def __init__(self, input_channels=4, hidden_dim=512, lr=1e-5):
        """
        More efficient version using adaptive pooling
        """
        super(ValueNetwork, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            # Adaptive pooling to fixed size
            nn.AdaptiveAvgPool2d((4, 4))  # Output: 256 * 4 * 4 = 4096
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

        # Initialize optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device)

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 4, 480, 640)

        Returns:
            value: State value tensor of shape (batch_size, 1)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(device)



        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2)

        # print(f"input dim {x.shape}")


        # Feature extraction
        x = self.features(x)

        # Flatten
        x = x.flatten(start_dim=1)

        # Classification
        value = self.classifier(x)

        return value


    def get_loss(self, trajectory_tensor):
        """
        generate MSE for value network
        :param trajectory_tensor: contains batch data sampled from the generated data
        :param epilson: small value for clipping
        :return:
        """

        cur_state_np = trajectory_tensor['states']
        reward_to_go_tensor = torch.tensor(trajectory_tensor['returns'], dtype=torch.float32).to(device)

        value_tensor = self.forward(cur_state_np)

        # Calculate mean squared error between predicted values and actual returns
        value_loss = torch.mean((value_tensor - reward_to_go_tensor) ** 2)

        return value_loss

    def save_weights(self, weight_name="Value_nn_weight.pth"):
        torch.save(self.state_dict(), weight_name)

    def load_weights(self, weight_path):
        if not os.path.exists(weight_path):
            print(f"weight path is invalid {weight_path}")
            return False
        self.load_state_dict(torch.load(weight_path))


class PolicyNetwork(nn.Module):

    def __init__(self, n_states, n_actions, action_bounds, input_height=224, input_width=300, n_hidden_filters=256,
                 lr=1e-5):

        print(f"policy network initiation: input dim {n_states}, output dim {n_actions}")
        super(PolicyNetwork, self).__init__()

        self.input_height = input_height
        self.input_width = input_width

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
        self.fc3 = nn.Linear(512, n_states)

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

        # Initialize optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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

        # print(f"x before convolution {x.shape}")

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(device)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)

        # Input: [64, 4, 480, 1280]
        # Output: [64, 4, 224, 600]
        x = F.interpolate(
            x,
            size=(self.input_height, self.input_width),
            mode='bilinear',
            align_corners=False
        )

        # print(f"testing for current inputs {x.shape}")
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(F.relu(self.bn4(self.conv4(x))))

        # print(f"x after convolution {x.shape}")

        # print(f"expected inputs {self.conv_output_size}")

        # Flatten for fully connected layers
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        # activation layer
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

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
            self.action_bounds = torch.tensor(self.action_bounds, dtype=torch.float32).to(device)
        elif isinstance(self.action_bounds, torch.Tensor) and self.action_bounds.device != device:
            self.action_bounds = self.action_bounds.to(device)

        # Combine both parts
        # scaled_action = torch.cat([action_first_3, action_first_rotation, action_sec_3, action_sec_rotation], dim=1)
        scaled_action = u.clamp(min=-1.0, max=1.0)

        return scaled_action, log_prob

    def get_action_probability(self, state, action):
        """
        Calculate the probability density of taking a specific action given the state.

        Args:
            state: Environment state tensor
            action: The action tensor for which to compute probability

        Returns:
            prob: The probability density of the action
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        action = action.to(device)

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(device)


        # a Normal distribution from forward pass
        dist = self.forward(state)

        # Get log probability density
        log_prob = dist.log_prob(action)

        # Sum across only the last dimension to get joint probabilities
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)  # Use -1 to always refer to the last dimension

        return log_prob

    def get_loss(self, trajectory_tensor, epilson=0.2):
        """
        buffer data is saved as files. What is passed to this function is just a sample of
        index that we can use to load the state.

        :param trajectory_tensor: contains batch data sampled from the generated data.
        :param epilson: small value for clipping
        :return:
        """

        cur_state_tensor = torch.tensor(trajectory_tensor['states'], dtype=torch.float32)
        action_tensor = torch.tensor(trajectory_tensor['actions'], dtype=torch.float32)
        old_log_probs_tensor = trajectory_tensor['old_log_probs'].to(device)
        advantage_tensor = torch.tensor(trajectory_tensor['advantages'], dtype=torch.float32).to(device)

        # find the new probability of outputing current action given state
        new_log_prob_tensor = self.get_action_probability(cur_state_tensor, action_tensor)
        prob_ratio = torch.exp(new_log_prob_tensor - old_log_probs_tensor).to(device)
        clipped_prob_ratio = torch.clamp(prob_ratio, min=1-epilson, max=1+epilson).to(device)

        # Calculate the surrogate objectives
        surrogate1 = prob_ratio * advantage_tensor
        surrogate2 = clipped_prob_ratio * advantage_tensor

        # Take the minimum of the two surrogate objectives
        # Use negative because we want to maximize the objective, but optimizers minimize
        policy_loss_tensor = -torch.min(surrogate1, surrogate2)

        # Average over all timesteps and trajectories
        policy_loss = policy_loss_tensor.mean()

        return policy_loss

    def save_weights(self, weight_name="Policy_nn_weight.pth"):
        torch.save(self.state_dict(), weight_name)

    def load_weights(self, weight_path):
        if not os.path.exists(weight_path):
            print(f"weight path is invalid {weight_path}")
            return False
        self.load_state_dict(torch.load(weight_path))


