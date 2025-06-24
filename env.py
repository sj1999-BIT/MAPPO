"""
This is where the environment class is defined

Env class is a wrapper that allows us to render the environment in SAPIEN, load specific agents and run them for testing
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, PxrMaterial, OptifuserConfig, SceneConfig
from transforms3d.quaternions import axangle2quat, qmult
from typing import List


import cv2
import numpy as np


from controller import Controller

steel = PxrMaterial()
steel.metallic = 0.9
steel.specular = 0
steel.roughness = 0.3
steel.set_base_color([0.7, 0.8, 0.9, 1])

render_config = OptifuserConfig()
render_config.shadow_map_size = 8192
render_config.shadow_frustum_size = 10
render_config.use_shadow = False
render_config.use_ao = True


class Agent(object):
    def get_observation(self) -> object:
        raise NotImplementedError

    def set_action(self, action: object) -> None:
        raise NotImplementedError

    def get_metadata(self):
        return dict()


class Env(object):
    def step(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def save(self) -> object:
        raise NotImplementedError

    def load(self, saved: object) -> None:
        raise NotImplementedError

    def get_reward(self) -> np.ndarray:
        raise NotImplementedError

    def get_agents(self) -> List[Agent]:
        raise NotImplementedError

    def get_metadata(self):
        return dict()

    def close(self) -> None:
        pass


class SolutionBase(object):
    def init(self, env: Env):
        """
        called before the first time step
        """
        pass

    def act(self, env: Env, current_timestep: int):
        """
        called at each time step
        """
        raise NotImplementedError()


class Robot(Agent):
    def __init__(self, articulation: sapien.Articulation):
        self.robot = articulation
        self.dof = self.robot.dof
        self.active_joints = [j for j in self.robot.get_joints() if j.get_dof() == 1]

    def get_observation(self):
        """
        observation contains 3 parts: qpos, qvel, and the poses for each link
        """
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        poses = [l.pose for l in self.robot.get_links()]
        return qpos, qvel, poses

    def configure_controllers(self, ps, ds):
        """
        Set parameters for the PD controllers for the robot joints
        """
        assert len(ps) == self.dof
        assert len(ds) == self.dof
        for j, p, d in zip(self.active_joints, ps, ds):
            j.set_drive_property(p, d)

    def set_action(self, drive_target, drive_velocity, additional_force):
        """
        action includes 3 parts
        drive_target: qpos target for PD controller
        drive_velocity: qvel target for PD controller
        additional_force: additional qf applied to the joints
        """
        assert len(drive_target) == self.dof
        assert len(drive_velocity) == self.dof
        assert len(additional_force) == self.dof
        for j, t, v in zip(self.active_joints, drive_target, drive_velocity):
            j.set_drive_target(t)
            j.set_drive_velocity_target(v)
        self.robot.set_qf(additional_force)

    def get_metadata(self):
        """
        metadata contains the root link pose and the render ids for each link
        """
        return {
            'root_pose': self.robot.get_root_pose(),
            'link_ids': [l.get_id() for l in self.robot.get_links()]
        }

    def get_compute_functions(self):
        """
        provides various convenience functions
        """
        return {
            'forward_dynamics': self.robot.compute_forward_dynamics,
            'inverse_dynamics': self.robot.compute_inverse_dynamics,
            'adjoint_matrix': self.robot.compute_adjoint_matrix,
            'spatial_twist_jacobian': self.robot.compute_spatial_twist_jacobian,
            'world_cartesian_jacobian': self.robot.compute_world_cartesian_jacobian,
            'manipulator_inertia_matrix': self.robot.compute_manipulator_inertia_matrix,
            'transformation_matrix': self.robot.compute_transformation_matrix,
            'passive_force': self.robot.compute_passive_force,
            'twist_diff_ik': self.robot.compute_twist_diff_ik,
            'cartesian_diff_ik': self.robot.compute_cartesian_diff_ik
        }


class Camera(Agent):
    def __init__(self, camera: sapien.OptifuserCamera):
        self.camera = camera
        self.metadata = None

    def set_action(self, q):
        """
        camera cannot do anything
        """
        pass

    def get_observation(self):
        """
        provides color, depth, and segmentation maps
        """
        self.camera.take_picture()
        return [self.camera.get_color_rgba(), self.camera.get_depth(), self.camera.get_segmentation()]

    def get_metadata(self):
        """
        provide camera parameters
        """
        return {
            'pose': self.camera.get_pose(),
            'near': self.camera.get_near(),
            'far': self.camera.get_far(),
            'width': self.camera.get_width(),
            'height': self.camera.get_height(),
            'fov': self.camera.get_fovy(),
            'projection_matrix': self.camera.get_projection_matrix(),
            'model_matrix': self.camera.get_model_matrix()
        }


class FinalEnv(Env):
    def __init__(self, timestep=1 / 500, frame_skip=5):
        self.global_total_timesteps = 0
        self.global_max_steps = 1000000  # the total time budget is 2000 seconds (in actual testing it will be greater)
        self.local_total_timesteps = 0
        self.local_max_steps = 100000  # each run needs to end after 200 seconds
        self.total_box_genreated = 0
        self.total_box_picked = 0

        self.frame_skip = frame_skip
        self.window = False
        self.engine = sapien.Engine(0, 0.001, 0.005)
        self.renderer = sapien.OptifuserRenderer(config=render_config)
        self.renderer.enable_global_axes(False)
        self.engine.set_renderer(self.renderer)
        self.renderer_controller = sapien.OptifuserController(self.renderer)
        # self.renderer_controller.set_camera_position(-2, 0, 1)
        self.renderer_controller.set_camera_position(-1.3, 0, 4.1)
        # self.renderer_controller.set_camera_rotation(0, -0.5)
        self.renderer_controller.set_camera_rotation(0, -1.4)

        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        self.scene = self.engine.create_scene(config=scene_config)
        self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)
        self.scene.set_shadow_light([0.2, -0.2, -1], [1, 1, 1])
        self.scene.set_ambient_light([0.3, 0.3, 0.3])

        self.build_room()
        self.build_table()
        self.build_camreas()

        self.left_robot: Robot = None
        self.right_robot: Robot = None
        loader = self.scene.create_urdf_loader()
        self.robot_builder = loader.load_file_as_articulation_builder('./assets/robot/panda_spade.urdf')

        self.bin = None
        self.boxes: List[sapien.Actor] = []
        for i in range(10):
            s = np.random.random() * 0.01 + 0.01
            d = self.create_dice([s, s, s])
            self.boxes.append(d)

    def step(self):
        self.scene.step()
        self.scene.update_render()
        self.local_total_timesteps += 1
        self.global_total_timesteps += 1

    def build_room(self):
        b = self.scene.create_actor_builder()
        b.add_box_shape(Pose([0, 0, -0.1]), [2.5, 2.5, 0.1])
        b.add_box_shape(Pose([0, -2.5, 1.5]), [2.5, 0.1, 1.5])
        b.add_box_shape(Pose([0, 2.5, 1.5]), [2.5, 0.1, 1.5])
        b.add_box_shape(Pose([-2.5, 0, 1.5]), [0.1, 2.5, 1.5])
        b.add_box_shape(Pose([2.5, 0, 1.5]), [0.1, 2.5, 1.5])

        b.add_box_visual(Pose([0, 0, -0.1]), [2.5, 2.5, 0.1], [0.2, 0.2, 0.2])
        b.add_box_visual(Pose([0, -2.5, 1.5]), [2.5, 0.1, 1.5], [0.8, 0.8, 0.8])
        b.add_box_visual(Pose([0, 2.5, 1.5]), [2.5, 0.1, 1.5], [0.8, 0.8, 0.8])
        b.add_box_visual(Pose([-2.5, 0, 1.5]), [0.1, 2.5, 1.5], [0.8, 0.8, 0.8])
        b.add_box_visual(Pose([2.5, 0, 1.5]), [0.1, 2.5, 1.5], [0.8, 0.8, 0.8])

        self.room = b.build_static(name="room")

    def build_table(self):
        b = self.scene.create_actor_builder()
        b.add_multiple_convex_shapes_from_file('./assets/bigboard.obj', Pose([-0.3, 0, 0]))
        b.add_visual_from_file('./assets/bigboard.obj', Pose([-0.3, 0, 0]))
        self.table = b.build_static(name="table")

    #####################################
    # attempt to create camera that is attached to the end-effector
    def create_end_effector_camera(self, robot, camera_name, offset_pose=None, show_visual=True):
        """
        Create end effector camera with optional visual representation

        Args:
            robot: Robot object
            camera_name: Name for the camera
            offset_pose: Camera offset from end effector
            show_visual: Whether to show a visual camera model
        """
        if offset_pose is None:
            offset_pose = Pose([0.05, 0, -0.02], axangle2quat([1, 0, 0], np.deg2rad(15)))

        end_effector_link = robot.robot.get_links()[9]

        # Add visual camera model if requested
        if show_visual:
            # Create a small camera visual on the end effector
            builder = self.scene.create_actor_builder()
            # Small box to represent camera body
            builder.add_box_visual(Pose([0, 0, 0]), [0.01, 0.015, 0.01], [0.2, 0.2, 0.2])
            # Small cylinder for lens
            builder.add_capsule_visual(Pose([0.01, 0, 0], axangle2quat([0, 1, 0], np.pi / 2)),
                                       0.005, 0.005, [0.1, 0.1, 0.1])

            camera_visual = builder.build(True, name=f"{camera_name}_visual")

            # # Create joint to attach visual to end effector
            # joint_builder = self.scene.create_actor_builder()
            # joint_builder.set_joint_type(sapien.ArticulationJointType.FIXED)
            # joint_builder.set_parent_link(end_effector_link)
            # joint_builder.set_child_link(camera_visual)
            # joint_builder.set_joint_pose_in_parent(offset_pose)
            # joint_builder.set_joint_pose_in_child(Pose())
            # joint = joint_builder.build()

        # Create the actual camera
        # Calculate field of view parameters
        fovy = np.deg2rad(60)  # 60 degree vertical field of view
        aspect_ratio = 640 / 480
        fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)

        # Create camera with correct SAPIEN signature
        camera = self.scene.add_mounted_camera(
            camera_name,  # name: str
            end_effector_link,  # actor: ActorBase
            offset_pose,  # pose: Pose
            640,  # width: int
            480,  # height: int
            fovx,  # fovx: float
            fovy,  # fovy: float
            0.01,  # near: float
            10.0  # far: float
        )

        return Camera(camera)

    def setup_end_effector_cameras(self):
        """Set up cameras on both robot end effectors"""
        if self.left_robot is not None:
            # Left robot camera - offset slightly to avoid collision
            # correct direction, wrong orientation
            rotation1 = axangle2quat([0, 1, 0], np.deg2rad(-90))
            # Second rotation: 180° around X-axis (flip upside down)
            rotation2 = axangle2quat([1, 0, 0], np.deg2rad(180))
            # Combine rotations
            combined_rotation = qmult(rotation1, rotation2)

            left_offset = Pose([0, 0, 0], combined_rotation)

            self.left_ee_camera = self.create_end_effector_camera(
                self.left_robot, 'left_ee_camera', left_offset, show_visual=True
            )


        if self.right_robot is not None:
            rotation1 = axangle2quat([0, 1, 0], np.deg2rad(-90))
            # Second rotation: 180° around X-axis (flip upside down)
            rotation2 = axangle2quat([1, 0, 0], np.deg2rad(180))
            # Combine rotations
            combined_rotation = qmult(rotation1, rotation2)

            right_offset = Pose([0, 0, 0], combined_rotation)

            self.right_ee_camera = self.create_end_effector_camera(
                self.right_robot, 'right_ee_camera', right_offset, show_visual=True
            )

            b = self.scene.create_actor_builder()
            b.add_visual_from_file('./assets/camera.dae', Pose(), scale=[0.3, 0.3, 0.3])
            cam_front = b.build(True, name="camera")
            pitch = np.deg2rad(-15)
            # cam_front.offset_pose = right_offset
            # cam_front.end_effector_link = self.right_robot.robot.get_links()[9]

            self.right_ee_camera.visual_actor = cam_front

    #######################################

    def build_camreas(self):

        b = self.scene.create_actor_builder()
        b.add_visual_from_file('./assets/camera.dae')
        cam_front = b.build(True, name="camera")
        pitch = np.deg2rad(-15)
        cam_front.set_pose(
            Pose([1, 0, 1], qmult(axangle2quat([0, 0, 1], np.pi), axangle2quat([0, -1, 0], pitch))))
        c = self.scene.add_mounted_camera('front_camera', cam_front, Pose(), 640, 480, 0, np.deg2rad(45), 0.1,
                                          10)
        self.front_camera = Camera(c)
        b = self.scene.create_actor_builder()
        b.add_visual_from_file('./assets/tripod.dae', Pose([0, 0, -1]))
        b.build(True, name="tripod").set_pose(Pose([1, 0, 1]))

        b = self.scene.create_actor_builder()
        b.add_visual_from_file('./assets/camera.dae')
        cam_left = b.build(True, name="camera")
        pitch = np.deg2rad(-15)
        cam_left.set_pose(
            Pose([0, 1.5, 1], qmult(axangle2quat([0, 0, 1], -np.pi / 2), axangle2quat([0, -1, 0], pitch))))
        c = self.scene.add_mounted_camera('left_camera', cam_left, Pose(), 640, 480, 0, np.deg2rad(45), 0.1, 10)

        self.left_camera = Camera(c)
        b = self.scene.create_actor_builder()
        b.add_visual_from_file('./assets/tripod.dae', Pose([0, 0, -1]))
        b.build(True, name="tripod").set_pose(Pose([0, 1.5, 1]))

        b = self.scene.create_actor_builder()
        b.add_visual_from_file('./assets/camera.dae')
        cam_right = b.build(True, name="camera")
        pitch = np.deg2rad(-15)
        cam_right.set_pose(
            Pose([0, -1.5, 1], qmult(axangle2quat([0, 0, 1], np.pi / 2), axangle2quat([0, -1, 0], pitch))))
        c = self.scene.add_mounted_camera('right_camera', cam_right, Pose(), 640, 480, 0, np.deg2rad(45), 0.1, 10)
        self.right_camera = Camera(c)
        b = self.scene.create_actor_builder()
        b.add_visual_from_file('./assets/tripod.dae', Pose([0, 0, -1]))
        b.build(True, name="tripod").set_pose(Pose([0, -1.5, 1]))

        b = self.scene.create_actor_builder()
        b.add_visual_from_file('./assets/camera.dae')
        cam_top = b.build(True, name="camera")
        pitch = np.deg2rad(-90)
        cam_top.set_pose(Pose([-0.5, 0, 3], axangle2quat([0, -1, 0], pitch)))
        c = self.scene.add_mounted_camera('top_camera', cam_top, Pose(), 640, 480, 0, np.deg2rad(45), 0.1, 10)
        self.top_camera = Camera(c)

    def create_bin(self, p, r, size, thickness):
        if self.bin is not None:
            self.scene.remove_actor(self.bin)

        self.bin_size = size
        b = self.scene.create_actor_builder()
        b.add_box_shape(Pose([0, 0, thickness / 2]), [size[0] / 2, size[1] / 2, thickness / 2])
        b.add_box_visual(Pose([0, 0, thickness / 2]), [size[0] / 2, size[1] / 2, thickness / 2],
                         [0.1, 0.8, 1])

        b.add_box_shape(Pose([0, size[1] / 2, size[2] / 2]),
                        [size[0] / 2 + thickness / 2, thickness / 2, size[2] / 2])
        b.add_box_visual(Pose([0, size[1] / 2, size[2] / 2]),
                         [size[0] / 2 + thickness / 2, thickness / 2, size[2] / 2], [0.1, 0.8, 1])
        b.add_box_shape(Pose([0, -size[1] / 2, size[2] / 2]),
                        [size[0] / 2 + thickness / 2, thickness / 2, size[2] / 2])
        b.add_box_visual(Pose([0, -size[1] / 2, size[2] / 2]),
                         [size[0] / 2 + thickness / 2, thickness / 2, size[2] / 2], [0.1, 0.8, 1])

        b.add_box_shape(Pose([size[0] / 2, 0, size[2] / 2]), [thickness / 2, size[1] / 2, size[2] / 2])
        b.add_box_visual(Pose([size[0] / 2, 0, size[2] / 2]), [thickness / 2, size[1] / 2, size[2] / 2],
                         [0.1, 0.8, 1])
        b.add_box_shape(Pose([-size[0] / 2, 0, size[2] / 2]), [thickness / 2, size[1] / 2, size[2] / 2])
        b.add_box_visual(Pose([-size[0] / 2, 0, size[2] / 2]), [thickness / 2, size[1] / 2, size[2] / 2],
                         [0.1, 0.8, 1])

        self.bin = b.build(True, name="bin")
        self.bin.set_pose(Pose(p, axangle2quat([0, 0, 1], r)))

    def check_inside_bin(self, point):
        size = self.bin_size
        [x, y, z] = self.bin.pose.inv().transform(Pose(point)).p
        return -size[0] / 2 < x < size[0] / 2 and \
            -size[1] / 2 < y < size[1] / 2 and \
            0 < z < size[2]

    def create_dice(self, size):
        b = self.scene.create_actor_builder()
        b.add_convex_shape_from_file('./assets/dice.obj', scale=size)
        b.add_visual_from_file('./assets/dice.obj', scale=size)
        return b.build(name="cube")

    def load_robot(self, pose, size, thickness=0.003, offset=0.01):
        x, y, z = size
        lb = self.robot_builder.get_link_builders()
        lb[9].remove_all_shapes()
        lb[9].remove_all_visuals()
        lb[9].add_box_shape(Pose([-x / 2, 0, z / 2 + offset]), [thickness / 2, y / 2, z / 2])
        lb[9].add_box_visual_complex(Pose([-x / 2, 0, z / 2 + offset]), [thickness / 2, y / 2, z / 2], steel)
        lb[9].add_box_shape(Pose([0, y / 2, z / 2 + offset]), [x / 2 + thickness / 2, thickness / 2, z / 2])
        lb[9].add_box_visual_complex(Pose([0, y / 2, z / 2 + offset]),
                                     [x / 2 + thickness / 2, thickness / 2, z / 2], steel)
        lb[9].add_box_shape(Pose([0, -y / 2, z / 2 + offset]), [x / 2 + thickness / 2, thickness / 2, z / 2])
        lb[9].add_box_visual_complex(Pose([0, -y / 2, z / 2 + offset]),
                                     [x / 2 + thickness / 2, thickness / 2, z / 2], steel)
        lb[9].add_box_shape(Pose([0, 0, offset]),
                            [x / 2 + thickness / 2, y / 2 + thickness / 2, thickness / 2])
        lb[9].add_box_visual_complex(Pose([0, 0, offset]),
                                     [x / 2 + thickness / 2, y / 2 + thickness / 2, thickness / 2], steel)

        r = self.robot_builder.build(True)
        r.name = "robot"
        r.set_root_pose(pose)
        return r

    def init(self):
        b = self.scene.create_actor_builder()
        b.add_box_shape(size=[0.02, .5, 1])
        b.add_box_visual(size=[0.02, .5, 1])
        w1 = b.build(True)
        w1.set_pose(Pose([-0.2, 0, 1]))

        b = self.scene.create_actor_builder()
        b.add_box_shape(size=[0.02, .5, 1])
        b.add_box_visual(size=[0.02, .5, 1])
        w2 = b.build(True)
        w2.set_pose(Pose([0.2, 0, 1]))

        b = self.scene.create_actor_builder()
        b.add_box_shape(size=[0.25, 0.02, 1])
        b.add_box_visual(size=[0.25, 0.02, 1])
        w3 = b.build(True)
        w3.set_pose(Pose([0, -0.4, 1]))

        b = self.scene.create_actor_builder()
        b.add_box_shape(size=[0.25, 0.02, 1])
        b.add_box_visual(size=[0.25, 0.02, 1])
        w4 = b.build(True)
        w4.set_pose(Pose([0, 0.4, 1]))

        for i in range(1500):
            self.scene.step()
        self.scene.remove_actor(w1)
        self.scene.remove_actor(w2)
        self.scene.remove_actor(w3)
        self.scene.remove_actor(w4)

    def reset(self):
        self.total_box_genreated += 10
        self.local_total_timesteps = 0
        if self.left_robot:
            self.scene.remove_articulation(self.left_robot.robot)
            self.scene.remove_articulation(self.right_robot.robot)
        base_size = np.array([0.015, 0.07, 0.07])
        size = base_size + np.random.random(3) * np.array([0.01, 0.06, 0.06])
        self.left_robot = Robot(
            self.load_robot(Pose([-.5, .25, 0.6], axangle2quat([0, 0, 1], -np.pi / 2)), size))
        self.left_robot.robot.set_qpos([1.5, 1, 0, -1, 0, 0, 0])
        self.right_robot = Robot(
            self.load_robot(Pose([-.5, -.25, 0.6], axangle2quat([0, 0, 1], np.pi / 2)), size))
        self.right_robot.robot.set_qpos([-1.5, 1, 0, -1, 0, 0, 0])

        self.create_bin([-1.2 + np.random.random() * 0.1,
                         np.random.random() * 0.2 - 0.1, 0.6],
                        np.random.random() * np.pi, [
                            0.2 + np.random.random() * 0.05, 0.3 + np.random.random() * 0.05,
                            0.4 + np.random.random() * 0.05
                        ], 0.015 + np.random.random() * 0.05)

        for d in self.boxes:
            d.set_pose(Pose([np.random.random() * 0.2 - 0.1, np.random.random() * 0.4 - 0.2, 2]))

        # add in 2 more cameras
        self.setup_end_effector_cameras()

        self.init()
        self.scene.update_render()

    def save(self):
        return {
            'boxes': [d.pack() for d in self.boxes],
            'r1': self.left_robot.robot.pack(),
            'r2': self.right_robot.robot.pack(),
        }

    def load(self, saved):
        for d, s in zip(self.boxes, saved['boxes']):
            d.unpack(s)
        self.left_robot.robot.unpack(saved['r1'])
        self.right_robot.robot.unpack(saved['r2'])

    def get_agents(self):
        return [
            self.left_robot, self.right_robot, self.front_camera, self.left_camera, self.right_camera,
            self.top_camera
        ]

    def get_reward(self):
        return sum([self.check_inside_bin(b.pose.p) for b in self.boxes])

    def close(self):
        self.scene = None

    def render_all_cameras(self):
        """Render all camera views in separate windows"""
        if not self.show_camera_views:
            return

        # Get images from all cameras
        cameras = {
            # 'Front Camera': self.front_camera,
            'Left Camera': self.left_ee_camera,
            'Left Camera1': self.right_ee_camera
        }

        for name, camera in cameras.items():
            # Get camera observation
            color_rgba, depth, segmentation = camera.get_observation()

            # Convert RGBA to RGB for display
            color_rgb = color_rgba[:, :, :3]

            # Normalize and convert to uint8
            color_display = (color_rgb * 255).astype(np.uint8)

            # Display the image
            cv2.imshow(name, cv2.cvtColor(color_display, cv2.COLOR_RGB2BGR))

    def get_RGBD_numpy(self, camera: Camera):
        """
        return the RGBD of a given camera
        """
        bgra, depth, segmentation = camera.get_observation()

        bgr = bgra[:, :, :3]

        rgb_np = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        rgb_np = cv2.normalize(rgb_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        depth_normalized_np = np.expand_dims(np.array(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX,
                                                                    dtype=cv2.CV_8U)), axis=-1)

        rgbd_np = np.concatenate([rgb_np, depth_normalized_np], axis=-1)

        return rgbd_np

    ##############################################################
    def get_end_effector_poses(self):
        """
        Get the position and rotation matrix of both robot end effectors.

        Returns:
            dict: Dictionary containing pose information for both end effectors
                {
                    'left_robot': {
                        'position': np.array([x, y, z]),
                        'rotation_matrix': np.array(3x3),
                        'pose': sapien.Pose object
                    },
                    'right_robot': {
                        'position': np.array([x, y, z]),
                        'rotation_matrix': np.array(3x3),
                        'pose': sapien.Pose object
                    }
                }
        """
        end_effector_poses = {}

        # Get left robot end effector pose
        if self.left_robot is not None:
            # Get the end effector link (index 9 based on the code)
            left_ee_link = self.left_robot.robot.get_links()[9]
            left_pose = left_ee_link.get_pose()

            end_effector_poses['left_robot'] = {
                'position': np.array(left_pose.p),
                'rotation_matrix': np.array(left_pose.to_transformation_matrix()[:3, :3]),
                'pose': left_pose
            }
        else:
            end_effector_poses['left_robot'] = None

        # Get right robot end effector pose
        if self.right_robot is not None:
            # Get the end effector link (index 9 based on the code)
            right_ee_link = self.right_robot.robot.get_links()[9]
            right_pose = right_ee_link.get_pose()

            end_effector_poses['right_robot'] = {
                'position': np.array(right_pose.p),
                'rotation_matrix': np.array(right_pose.to_transformation_matrix()[:3, :3]),
                'pose': right_pose
            }
        else:
            end_effector_poses['right_robot'] = None

        return end_effector_poses

    def get_end_effector_positions(self):
        """
        Get only the positions of both robot end effectors.

        Returns:
            dict: Dictionary containing position vectors
                {
                    'left_robot': np.array([x, y, z]) or None,
                    'right_robot': np.array([x, y, z]) or None
                }
        """
        positions = {}

        if self.left_robot is not None:
            left_ee_link = self.left_robot.robot.get_links()[9]
            positions['left_robot'] = np.array(left_ee_link.get_pose().p)
        else:
            positions['left_robot'] = None

        if self.right_robot is not None:
            right_ee_link = self.right_robot.robot.get_links()[9]
            positions['right_robot'] = np.array(right_ee_link.get_pose().p)
        else:
            positions['right_robot'] = None

        return positions

    def get_end_effector_rotation_matrices(self):
        """
        Get only the rotation matrices of both robot end effectors.

        Returns:
            dict: Dictionary containing 3x3 rotation matrices
                {
                    'left_robot': np.array(3x3) or None,
                    'right_robot': np.array(3x3) or None
                }
        """
        rotations = {}

        if self.left_robot is not None:
            left_ee_link = self.left_robot.robot.get_links()[9]
            left_transform = left_ee_link.get_pose().to_transformation_matrix()
            rotations['left_robot'] = np.array(left_transform[:3, :3])
        else:
            rotations['left_robot'] = None

        if self.right_robot is not None:
            right_ee_link = self.right_robot.robot.get_links()[9]
            right_transform = right_ee_link.get_pose().to_transformation_matrix()
            rotations['right_robot'] = np.array(right_transform[:3, :3])
        else:
            rotations['right_robot'] = None

        return rotations

    def print_end_effector_info(self):
        """
        Print current end effector pose information for debugging.
        """
        poses = self.get_end_effector_poses()

        for robot_name, pose_data in poses.items():
            if pose_data is not None:
                print(f"\n{robot_name.upper()} End Effector:")
                print(f"Position: {pose_data['position']}")
                print(f"Rotation Matrix:\n{pose_data['rotation_matrix']}")
            else:
                print(f"\n{robot_name.upper()}: Not initialized")

    def get_end_effector_distance(self):
        """
        Calculate the distance between the two end effectors.

        Returns:
            float: Distance between end effectors, or None if either robot is not initialized
        """
        positions = self.get_end_effector_positions()

        if positions['left_robot'] is not None and positions['right_robot'] is not None:
            return np.linalg.norm(positions['left_robot'] - positions['right_robot'])
        else:
            return None
    ##############################################################

    def get_env_variables(self):
        """
        Get all essential information from the finalEnv object to be passed to controller
        Returns:
        """
        meta = self.get_metadata()
        box_ids = meta['box_ids']
        r1, r2, c1, c2, c3, c4 = self.get_agents()

        left_camera = self.left_ee_camera
        right_camera = self.right_ee_camera
        top_camera = self.top_camera

        # determine target bin coordinates
        bin_id = meta['bin_id']

        return {
            "left_robot": r1,
            "right_robot": r2,
            "left_camera": left_camera,
            "right_camera": right_camera,
            "top_camera": top_camera,
            "left_view": self.get_RGBD_numpy(left_camera),
            "right_view": self.get_RGBD_numpy(right_camera),
            "top_view": self.get_RGBD_numpy(top_camera),
            "bin_id": bin_id,
            "box_ids": box_ids
        }

        # only this function matter

    def run(self, controller: Controller, render=False, render_interval=1, debug=False):
        """Run the environment

        Args:
            solution: your solution
            render: whether to render the scene
            render_interval: if render == True, the interval to render.
                A large value can increase fps, but will result in non-smooth visualization.
            debug: if true, the environment will run for only one episode.
                Otherwise it will keep running until it reaches the max time step.

        """
        if debug:
            self.reset()
            while not self.renderer_controller.should_quit:
                if not controller.act(self.get_env_variables()):
                    break
                for _ in range(self.frame_skip):
                    self.step()
                if render and (self.local_total_timesteps // self.frame_skip) % render_interval == 0:
                    self.render()
            self.end_episode()
            self.print_stat()
            return

        while self.global_total_timesteps < self.global_max_steps:
            # In testing, a fixed sequence of randomly seeds will be set here
            self.reset()
            while True:
                if self.local_total_timesteps >= self.local_max_steps or \
                        self.global_total_timesteps >= self.global_max_steps or \
                        not controller.act(self.get_env_variables()):
                    break
                for _ in range(self.frame_skip):
                    self.step()
                if render and self.local_total_timesteps // self.frame_skip % render_interval == 0:
                    self.render()
            self.end_episode()
            self.print_stat()

        self.print_stat()

    def render(self, show_cameras=False):
        """Enhanced render method with camera view option"""
        # Render main view
        if not self.window:
            self.renderer_controller.show_window()
            self.window = True
        self.renderer_controller.render()

        # Optionally render all camera views
        if show_cameras:
            self.show_camera_views = True
            self.render_all_cameras()
            cv2.waitKey(1)  # Process OpenCV events

    def get_metadata(self):
        return {
            'timestep': self.scene.get_timestep(),
            'box_ids': [b.get_id() for b in self.boxes],
            'bin_id': self.bin.get_id(),
            'frame_skip': self.frame_skip
        }

    def end_episode(self):
        r = self.get_reward()
        self.total_box_picked += r

    def print_stat(self):
        seconds = self.global_total_timesteps * self.scene.get_timestep()
        print('running for {:.2f} seconds, success on {}/{} boxes'.format(seconds, self.total_box_picked,
                                                                          self.total_box_genreated))
        print('success_rate={:.2f}%, efficiency={:.2f}/minute'.format(
            (self.total_box_picked / self.total_box_genreated) * 100, self.total_box_picked / seconds * 60
        ))

