"""
Controllers are a specific wrapper that takes in values from environment and return an action.

Action is 2 array, each 12 output:
    1. first 3 values represent the end effector position
    2. next 9 values represent the rotation matrix for the end effector
"""
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2axangle, qmult, qinverse, mat2quat

import numpy as np

from neural_networks import PolicyNetwork, RGBDNetwork

"""
Helper functions needed by the agents.
"""


def skew_symmetric_matrix(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def G_inv(theta):
    """
   Direct implementation of book formula:
   G(θ) = Iθ + (1-cos θ)[ω] + (θ-sin θ)[ω]²
   """
    theta_norm = np.linalg.norm(theta)
    if theta_norm < 1e-6:
        return np.eye(3)

    # Get unit vector (ω)
    omega = theta / theta_norm
    omega_skew = skew_symmetric_matrix(omega)

    # Compute each term
    term1 = theta_norm * np.eye(3)  # Iθ
    term2 = (1 - np.cos(theta_norm)) * omega_skew  # (1-cos θ)[ω]
    term3 = (theta_norm - np.sin(theta_norm)) * omega_skew @ omega_skew  # (θ-sin θ)[ω]²

    G = term1 + term2 + term3
    return np.linalg.inv(G)


def pose2exp_coordinate(pose):
    # Different variable names and extraction method
    T = pose  # use full transform name
    R = T[:3, :3]
    p = T[:3, 3]

    # Different way to check identity case
    trace_R = np.trace(R)
    if abs(trace_R - 3) < 1e-2:  # slightly different threshold
        # Pure translation case
        w = np.zeros(3)
        v = p
        magnitude = np.sqrt(np.sum(v ** 2))  # different way to compute norm
        return np.concatenate([w, v / magnitude]), magnitude

    # Rotation case - different order and structure
    angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))  # added clip for stability

    # Different way to compute rotation axis
    skew = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))

    # Compute rotation vector
    w = angle * skew

    # Get translation component
    v = G_inv(w) @ p

    # Different way to compute magnitude
    magnitude = np.sqrt(w @ w)

    # Different order of operations in return
    result = np.zeros(6)
    result[:3] = w / magnitude
    result[3:] = v

    return result, magnitude


def pose2mat(pose):
    """You need to implement this function

    You will need to implement this function first before any other functions.
    In this function, you need to convert a (position: pose.p, quaternion: pose.q) into a SE(3) matrix

    You can not directly use external library to transform quaternion into rotation matrix.
    Only numpy can be used here.
    Args:
        pose: sapien Pose object, where Pose.p and Pose.q are position and quaternion respectively

    Hint: the convention of quaternion

    Returns:
        (4, 4) transformation matrix represent the same pose

    """
    pos = pose.p  # 3D position
    q = pose.q  # Quaternion [w,x,y,z]

    # Extract quaternion components
    w, x, y, z = q
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])

    # Ensure orthogonality (clean up numerical errors)
    U, _, Vh = np.linalg.svd(R)
    R = U @ Vh

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return T


def is_box_reachable(box_position):
    """
    Check if the box is within the robot's workspace.
    Args:
        box_position: 3D global position of the box.
    Returns:
        bool: True if the box is reachable, False otherwise.
    """
    # Define the robot's workspace limits (adjust as needed)
    workspace_limits = {
        'x': [-0.30, 0.19],  # X-axis limits
        'y': [-0.40, 0.40],  # Y-axis limits
        'z': [0.0, 0.64]  # Z-axis limits
    }

    # Check if the box is within the workspace limits
    within_x = workspace_limits['x'][0] <= box_position[0] <= workspace_limits['x'][1]
    within_y = workspace_limits['y'][0] <= box_position[1] <= workspace_limits['y'][1]
    # within_z = workspace_limits['z'][0] <= box_position[2] <= workspace_limits['z'][1]

    # return within_x and within_y and within_z
    return within_x and within_y


def get_global_position_from_camera(camera, depth, x, y):
    """
    camera: an camera agent
    depth: the depth obsrevation
    x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
    """
    cm = camera.get_metadata()
    proj, model = cm['projection_matrix'], cm['model_matrix']
    w, h = cm['width'], cm['height']

    # get 0 to 1 coordinate for (x, y) coordinates
    xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

    # get 0 to 1 depth value at (x,y)
    zf = depth[int(y), int(x)]

    # get the -1 to 1 (x,y,z) coordinates
    ndc = np.array([xf, yf, zf, 1]) * 2 - 1

    # transform from image space to view space
    v = np.linalg.inv(proj) @ ndc
    v /= v[3]

    # transform from view space to world space
    v = model @ v

    return v


def compute_joint_velocity_from_twist(robot, twist: np.ndarray) -> np.ndarray:
    """
    This function calculates the joint velocities needed to achieve a given spatial twist at the end effector.

    robot: The robot being controlled
    twist: A 6-dimensional vector representing the spatial twist,(linear&angular velocity)
    """
    assert twist.size == 6
    # compute dense jacobian matrix
    dense_jacobian = robot.get_compute_functions()['spatial_twist_jacobian']()
    end_effector_jacobian = np.zeros([6, robot.dof])
    end_effector_index = 9
    end_effector_jacobian[:3, :] = dense_jacobian[end_effector_index * 6 - 3:end_effector_index * 6, :7]
    end_effector_jacobian[3:6, :] = dense_jacobian[(end_effector_index - 1) * 6:end_effector_index * 6 - 3, :7]
    # pseudo inverse of jacobian
    ee_jacobian_inverse = np.linalg.pinv(end_effector_jacobian)
    # twist to joint velocity
    joint_velocity = ee_jacobian_inverse @ twist
    return joint_velocity


def jacobian_drive(robot, index, target_pose, speed=0.5):
        """
        This function aims to move the robot's end effector to a target pose based on Jacobian matrices,
        which relate joint velocities to end effector velocities

        para: similar to above
        """
        # ee_pose to matrix
        passive_force = robot.get_compute_functions()['passive_force'](True, True, False)
        q_position, q_velocity, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        current_pose = pose2mat(current_pose)
        current_rotation = current_pose[:3, :3]
        current_position = current_pose[:3, 3]
        target_pose = pose2mat(target_pose)

        # transformation from current to target
        pose_difference = np.linalg.inv(current_pose) @ target_pose
        twist_difference, theta_difference = pose2exp_coordinate(pose_difference)
        twist_body_difference = twist_difference * speed

        # compute v with twist
        my_adjoint_matrix = np.zeros((6, 6))
        my_adjoint_matrix[0:3, 0:3] = current_rotation
        my_adjoint_matrix[3:6, 3:6] = current_rotation
        my_adjoint_matrix[3:6, 0:3] = skew_symmetric_matrix(current_position) @ current_rotation
        ee_twist_difference = my_adjoint_matrix @ twist_body_difference
        target_q_velocity = compute_joint_velocity_from_twist(robot, ee_twist_difference)

        robot.set_action(q_position, target_q_velocity, passive_force)

        return [q_position, target_q_velocity, passive_force]


def is_box_reachable(box_position):
    """
    Check if the box is within the robot's workspace.
    Args:
        box_position: 3D global position of the box.
    Returns:
        bool: True if the box is reachable, False otherwise.
    """
    # Define the robot's workspace limits (adjust as needed)
    workspace_limits = {
        'x': [-0.30, 0.20],  # X-axis limits
        'y': [-0.40, 0.40],  # Y-axis limits
        'z': [0.0, 1.5]    # Z-axis limits
    }

    # Check if the box is within the workspace limits
    within_x = workspace_limits['x'][0] <= box_position[0] <= workspace_limits['x'][1]
    within_y = workspace_limits['y'][0] <= box_position[1] <= workspace_limits['y'][1]
    within_z = workspace_limits['z'][0] <= box_position[2] <= workspace_limits['z'][1]

    return within_x and within_y and within_z


def get_offset(current_pose, bin_center):
    offset_x = bin_center[0] - current_pose[0]
    offset_y = bin_center[1] - current_pose[1]
    target_position = [
        current_pose[0],
        current_pose[1] + offset_y,
        current_pose[2]
    ]

    return target_position


def rotate_spade(robot, initial_qpos, rotation_step, total_steps, clockwise=True):
    # Rotate the joint
    direction = 1 if clockwise else -1
    qpos, _, _ = robot.get_observation()
    last_joint_index = robot.dof - 1
    qpos[last_joint_index] = initial_qpos[last_joint_index] + direction * (2 * np.pi) * (rotation_step / total_steps)

    drive_target = qpos
    drive_velocity = [0] * robot.dof
    additional_force = [0] * robot.dof
    robot.set_action(drive_target, drive_velocity, additional_force)

    return [drive_target, drive_velocity, additional_force]


class Controller:

    def __init__(self):
        # PD controller gains for 7-DOF robotic arms
        # Values decrease from base to end-effector joints for stable control
        self.ps = [1000, 800, 600, 600, 200, 200, 100]  # Proportional gains (Kp) for each joint
        self.ds = [1000, 800, 600, 600, 200, 200, 100]  # Derivative gains (Kd) for each joint

        # Flag to track if controllers have been configured
        self.initiated = False
        pass

    def configController(self, r1, r2):
        """
        Configure PD controllers for both robotic arms with predefined gains.

        The PD controllers compute joint torques using:
        τ = Kp * (q_desired - q_current) + Kd * (q̇_desired - q̇_current)

        Higher gains for base joints (1000) provide strong control for larger masses.
        Lower gains for end-effector joints (100) enable gentle, stable manipulation.

        Args:
            r1: Left robotic arm agent
            r2: Right robotic arm agent
        """

        # Apply PD gains to both arms, enables torque computation from position/velocity commands
        r1.configure_controllers(self.ps, self.ds)
        r2.configure_controllers(self.ps, self.ds)

        # Mark controllers as configured for subsequent control commands
        self.isControllerConfigured = True

    def act(self, envData: dict):
        """
        Based on all the date and variables received from env, set action for the robotic arms.
        No need to return anything. Setting the robotic arm action directly.

        Env only calls this function.

        All agent must implement this function.
        Args:
            envData: contains variables and agents in the environment.
        """
        raise NotImplementedError


class DemoController(Controller):
    """
    Implements the original solution that generates the default demostration.
    Each time env calls demo controller, passes everything that this agent needs to judge what
    action needed based on a deterministic control algorithm.
    """

    def __init__(self):
        super().__init__()
        self.phase = 0
        self.drive = 0

    def _init(self, envData: dict):
        """
        Initiate basic variables for the controller
        Args:
            envData: dict of agents and values from environment
        """

        # need to initiate first
        self.configController(envData["left_robot"], envData["right_robot"])
        # determine target bin coordinates
        self.bin_id = envData['bin_id']
        self.box_ids = envData['box_ids']
        self.bin_position = self.locate_bin(envData["top_camera"])

        # set target bin position for r1
        self.r1_target_bin_position = self.bin_position.copy()
        self.r1_target_bin_position[0] += 0.1  # Offset X-axis towards center of bin
        self.r1_target_bin_position[1] += 0.15  # Offset Y-axis towards center of bin
        self.r1_target_bin_position[2] += 0.5  # Offset in Z-axis for safety

        # set target bin position for r2
        self.r2_target_bin_position = self.bin_position.copy()
        self.r2_target_bin_position[0] += 0.1  # Offset X-axis towards center of bin
        self.r2_target_bin_position[1] -= 0.1  # Offset Y-axis towards center of bin
        self.r2_target_bin_position[2] += 0.4  # Offset in Z-axis for safety

        self.rotate_flag = False

        self.target_position = ""

        self.initiated = True

    def act(self, envData: dict):
        if not self.initiated:
            self._init(envData)

        r1, r2 = envData["left_robot"], envData["right_robot"]
        pf_left = f = r1.get_compute_functions()['passive_force'](True, True, False)
        pf_right = f = r2.get_compute_functions()['passive_force'](True, True, False)

        # first phase, move the arm to the starting position
        if self.phase == 0:
            t1 = [2, 1, 0, -1.5, -1, 1, -2]
            t2 = [-2, 1, 0, -1.5, 1, 1, -2]

            r1.set_action(t1, [0] * 7, pf_left)

            r2.set_action(t2, [0] * 7, pf_right)

            if np.allclose(r1.get_observation()[0], t1, 0.05, 0.05) and np.allclose(
                    r2.get_observation()[0], t2, 0.05, 0.05):
                self.phase = 1
                self.counter = 0
                self.selected_x = None

        if self.phase == 1:
            self.counter += 1

            if (self.counter == 1):
                selected = self.pick_box(envData["top_camera"])
                self.selected_x = selected[0]
                if self.selected_x is None:
                    return False

            target_pose_left = Pose([self.selected_x, 0.5, 0.67], euler2quat(np.pi, -np.pi / 3, -np.pi / 2))
            jacobian_drive(r1, 9, target_pose_left)

            target_pose_right = Pose([self.selected_x, -0.5, 0.6], euler2quat(np.pi, -np.pi / 3, np.pi / 2))
            jacobian_drive(r2, 9, target_pose_right)

            if self.counter == 2000 / 5:
                self.phase = 2
                self.counter = 0

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = 0.07
                self.pose_left = Pose(p, q)

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = -0.07
                self.pose_right = Pose(p, q)

        if self.phase == 2:
            self.counter += 1
            jacobian_drive(r1, 9, self.pose_left)
            jacobian_drive(r2, 9, self.pose_right)

            if self.counter == 3000 / 5:
                self.phase = 3

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.1
                q = euler2quat(np.pi, -np.pi / 4, -np.pi / 2)
                self.pose_left = Pose(p, q)

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.0
                q = euler2quat(np.pi, -np.pi / 1.8, np.pi / 2)
                self.pose_right = Pose(p, q)

                self.counter = 0

        if self.phase == 3:
            if self.counter < 500 / 5:
                self.counter += 1
                jacobian_drive(r1, 9, self.pose_left)
                jacobian_drive(r2, 9, self.pose_right)

            elif self.counter < 1500 / 5:
                self.counter += 1
                t1 = [3, 1, 0, -1.5, -1, 1, -2]
                r1.set_action(t1, [0] * 7, pf_left)
                jacobian_drive(r2, 9, self.pose_right)

            else:
                self.phase = 4
                # spade = self.locate_spade_length(c4)
                self.counter = 0

        if self.phase == 4:
            self.counter += 1
            # middle point 1
            if (self.counter < 3000 / 5):
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.5
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                jacobian_drive(r2, 9, Pose(p, q))
            elif (self.counter < 6000 / 5):
                p = self.bin_position.copy()
                p[2] += 0.7
                q = euler2quat(0, -np.pi / 3, 0)
                jacobian_drive(r2, 9, Pose(p, q))

            elif (self.counter == 6000 / 5):
                robot_pose = r2.get_observation()[2][9].p  # Robot's current end effector position
                self.target_position = get_offset(robot_pose, self.bin_position)

            elif (self.counter < 7000 / 5):
                self.r2_target_bin_position.copy()
                p = self.bin_position.copy()
                p[1] -= 0.2
                q = euler2quat(0, -np.pi / 2, 0)
                jacobian_drive(r2, 9, Pose(p, q), speed=0.4)

            elif (self.counter == 7000 / 5):
                robot_pose = r2.get_observation()[2][9].p  # Robot's current end effector position
                robot_pose[1] += -0.2
                self.target_position = robot_pose

            elif (self.counter < 7500 / 5):
                p = r2.get_observation()[2][9].p
                p[1] -= 0.01
                q = euler2quat(0, -np.pi / 2, 0)
                jacobian_drive(r2, 9, Pose(self.target_position, q), speed=0.4)

            else:
                # print("Phase 4d")  # Debug
                self.rotate_flag = True
                # Initialize rotation state if it doesn't exist
                if not hasattr(self, "_rotation_step"):
                    self._rotation_step = 0
                    self._total_steps = 36
                    self._delay_multiplier = 30
                    self._delay_counter = 0
                    self._initial_qpos = r2.get_observation()[0]

                if self._delay_counter < self._delay_multiplier:
                    self._delay_counter += 1
                else:
                    self._delay_counter = 0
                    self._rotation_step += 1

                    rotate_spade(r2, self._initial_qpos, self._rotation_step, self._total_steps)
                    # self.rotate_spade(r1, self._initial_qpos, self._rotation_step, self._total_steps, clockwise=False)
                if self._rotation_step >= self._total_steps:
                    del self._rotation_step
                    del self._total_steps
                    del self._delay_multiplier
                    del self._delay_counter
                    del self._initial_qpos
                    self.phase += 1

        if self.phase == 5: #set an independent phase for return to start
            self.rotate_flag = False
            self.phase = 0

        return True

    ############ HELPER FUNCTIONS ##############################################

    def locate_bin(self, camera):
        """
        Locate the bin's global position using the camera's segmentation and depth data.
        """
        _, depth, segmentation = camera.get_observation()
        bin_pixels = np.where(segmentation == self.bin_id)
        if len(bin_pixels[0]) == 0:
            raise ValueError("Bin not found in the camera's view.")

        # Calculate the center pixel of the bin
        x_center = int((bin_pixels[1].min() + bin_pixels[1].max()) / 2)
        y_center = int((bin_pixels[0].min() + bin_pixels[0].max()) / 2)

        # Convert the pixel to a global position
        bin_global_position = get_global_position_from_camera(camera, depth, x_center, y_center)
        return bin_global_position

    def pick_box(self, c):
        color, depth, segmentation = c.get_observation()

        # Get the global positions of all boxes
        box_positions = {}
        for i in self.box_ids:
            m = np.where(segmentation == i)
            if len(m[0]):
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for y, x in zip(m[0], m[1]):
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                global_position = get_global_position_from_camera(c, depth, x, y)
                box_positions[i] = global_position

        # If no boxes are found, return False
        if not box_positions:
            return [None,None]

        # Count neighbors for each box
        neighbor_counts = {}
        reachable_boxes = []
        for box_id, position in box_positions.items():
            # Check if the box is reachable (within the robot's workspace)
            if is_box_reachable(position):  # Implement this function
                reachable_boxes.append(box_id)
                neighbor_counts[box_id] = 0
                for other_box_id, other_position in box_positions.items():
                    if box_id != other_box_id:
                        distance = np.linalg.norm(position - other_position)
                        if distance < 0.2:  # Neighbor threshold (adjust as needed)
                            neighbor_counts[box_id] += 1

        # If no reachable boxes, return False
        if not reachable_boxes:
            return [None,None]

        # Prioritize boxes with the most neighbors
        prioritized_boxes = sorted(reachable_boxes, key=lambda x: neighbor_counts[x], reverse=True)

        # Return the position of the box with the most neighbors
        best_box_id = prioritized_boxes[0]
        return box_positions[best_box_id]


class PolicyController(Controller):
    """
    This controller utilises a policy network to generate the action based on inputs
    """

    def __init__(self, policy_weight_path=None, RGBD_weight_path=None):
        super().__init__()

        # define variables
        input_height = 224
        input_width = 600
        output_dim = 512
        action_dim = 12
        interval = 1
        action_bounds = [-1, 1]

        self.rgbd_nn = RGBDNetwork(input_height, input_width, output_dim)
        self.policy_nn = PolicyNetwork(n_states=output_dim, n_actions=action_dim, action_bounds=action_bounds)

        if policy_weight_path is not None:
            self.policy_nn.load_weights(policy_weight_path)

        if RGBD_weight_path is not None:
            self.rgbd_nn.load_weights(policy_weight_path)

    def get_next_pose(self, cur_state):
        """
        Given RGBD state, generate the corresponding next position and rotation
        Args:
            cur_state:

        Returns:

        """
        policy_inputs = self.rgbd_nn.process_inputs(cur_state)

        # pass processed batch inputs to pNet to generate the actions
        output_actions_tensor, log_probs = self.policy_nn.sample_or_likelihood(policy_inputs)

        # Convert to numpy and flatten to get 12 values
        values = output_actions_tensor.detach().cpu().numpy().flatten()  # shape: (12,)

        # Split into two parts
        position = values[:3]  # First 3 values for next position of end_effector
        rot_matrix = values[3:].reshape(3, 3)  # Next 9 values as 3x3 rotation matrix
        quat = mat2quat(rot_matrix)

        # print(f"position: {position}, quaternion: {quat}")

        pose = Pose(position, quat)

        return pose



    def act(self, envData: dict):

        # get the input states for each arm
        left_arm_cur_state = np.concatenate((envData["left_view"], envData["top_view"]))
        right_arm_cur_state = np.concatenate((envData["right_view"], envData["top_view"]))

        # get the action given policy
        pose_left = self.get_next_pose(left_arm_cur_state)
        pose_right = self.get_next_pose(right_arm_cur_state)

        # IK for robotic arm action
        jacobian_drive(envData["left_robot"], 9, pose_left)
        jacobian_drive(envData["right_robot"], 9, pose_right)

        return True






