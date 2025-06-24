"""
This is for actual testing of different classes and packages
"""

from env import FinalEnv
from controller import PolicyController

import numpy as np

if __name__ == '__main__':
    # at test time, we will use different random seeds.
    np.random.seed(0)
    env = FinalEnv()

    controller = PolicyController(RGBD_weight_path="MAPPO/weights/IL_initiated_weights/rgbdNet_weights.pth",
                                  policy_weight_path="MAPPO/weights/IL_initiated_weights/policyNet_weights.pth")

    env.run(controller, render=True, render_interval=5, debug=True)
    # at test time, run the following
    # env.run(Solution())
    env.close()