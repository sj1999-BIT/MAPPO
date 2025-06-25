"""
This is for actual testing of different classes and packages
"""

from env import FinalEnv
from controller import SinglePolicyController, DemoController, DummyController

import numpy as np

if __name__ == '__main__':
    # at test time, we will use different random seeds.
    np.random.seed(0)
    env = FinalEnv()
    # env.reset()

    # controller = DummyController()

    controller = DemoController()

    # controller = SinglePolicyController()

    env.run(controller, render=True, render_interval=1, debug=True)
    # at test time, run the following
    # env.run(Solution())
    env.close()