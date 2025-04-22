# system
import sys
# gym
import gymnasium as gym
from gymnasium import spaces
# webots
from typing import Optional, Tuple
from utils import *
from controller import Supervisor, Motor

# running through terminal
sys.path.append('/Applications/Webots.app/Contents/lib/controller/python')


class WebotsWallFollowing(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(WebotsWallFollowing, self).__init__()

        # training requirements debug and logging
        self.left_value = 0                 # for tensorboard
        self.max_episode_timestep = 5000    # Define a maximum number of timestep per episode
        self.episode_timestep = 0           # Initialize current timestep counter
        self.randomize = False              # True for training, False for testing

        # Webots configs
        self.robot: Supervisor = Supervisor()
        if self.robot is None:
            raise ValueError("Robot DEF 'e-puck' not found.")
        self.timestep = int(self.robot.getBasicTimeStep())

        # motors
        self.left_motor: Motor = self.robot.getDevice('left wheel motor')
        self.right_motor: Motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Nodes
        self.robot_node = self.robot.getFromDef("robot")
        self.translation_node = self.robot_node.getField("translation")

        # Action space normalized
        linear_velocity_low = -1
        linear_velocity_high = 1
        angular_velocity_low = -1
        angular_velocity_high = 1
        low = np.array([linear_velocity_low, angular_velocity_low])
        high = np.array([linear_velocity_high, angular_velocity_high])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Sensors Bumper and Lidar
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.robot.step()
        print("is lidar point cloud enable?", self.lidar.isPointCloudEnabled())
        self.bumper = self.robot.getDevice('touch sensor')
        self.bumper.enable(self.timestep)

        # Bounds for observation space
        self.min_range = self.lidar.getMinRange()  # 0.05
        self.max_range = self.lidar.getMaxRange()  # 0.3

        # Observation Space normalized
        num_lidar_readings = self.lidar.getHorizontalResolution()
        print("num_lidar_readings:" + str(num_lidar_readings))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_lidar_readings,),
            dtype=np.float32
        )

        # Initialize previous angular velocity (used in reward)
        self.prev_angular_velocity = 0.0
        self.prev_angular_vel_norm = 0.0
        self.prev_linear_velocity = 0.0

    @staticmethod
    def scale_action(action):
        # calculated from odometry
        max_linear_velocity = 0.1256    # m/s
        max_angular_velocity = 4.8307   # rad/s
        linear_vel = action[0] * max_linear_velocity
        angular_vel = action[1] * max_angular_velocity
        # dtype
        scaled_action = np.array([linear_vel, angular_vel], dtype=np.float32)
        return scaled_action

    def clip_observation(self, observation):
        # First, handle infinities if any
        processed_observation = np.where(np.isinf(observation), self.max_range, observation)
        return processed_observation.astype(np.float32)

    def normalize_observation(self, observation):
        # Normalize to 0 to 1
        normalized_observation = (observation - self.min_range) / (self.max_range - self.min_range)
        # Scale from 0 to 1 to -1 to 1
        scaled_observation = (normalized_observation * 2) - 1
        return scaled_observation.astype(np.float32)

    def step(self, action):
        # apply action
        linear_vel_norm, angular_vel_norm = action
        linear_vel, angular_vel = self.scale_action(action)
        cmd_vel(self.robot, linear_vel, angular_vel)
        self.robot.step(self.timestep)
        print("linear_vel:" + str(linear_vel), " angular_vel:" + str(angular_vel))

        # get Lidar values and clip only
        state = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        state = self.clip_observation(state)
        self.left_value = state[0]
        print("clip state", state)

        # update & reward
        # reward = self.calculate_reward(state, linear_vel, angular_vel)
        reward = self.calculate_reward(state, linear_vel_norm, angular_vel_norm)  # l_vel etc entre -1 e 1

        self.episode_timestep += 1  # Increment timestep with each call to step
        print("episode timestep", self.episode_timestep)

        # check done
        done = self.is_done(state)
        truncated = (self.episode_timestep >= self.max_episode_timestep)
        if done or truncated:
            done = True

        # normalization required and tensorboard logging info
        state = self.normalize_observation(state)
        info = {'current_min_distance': self.left_value}

        return state, reward, done, False, info

    def reset(self, seed: Optional[int] = None, return_info: bool = False) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Randomization or not
        positions = [
            [0.0476146, 0.49136, -0.0000319338],
            [-0.506447, 0.293503, -0.0000319338],
            [-0.129066, 0.0899107, -0.0000319338]
        ]
        initial_position = positions[np.random.choice(len(positions))] if self.randomize else [-0.325, -0.243, -3.08235e-05]
        # maze =  [-0.16, -0.8, -3.08235e-05]

        print(initial_position)
        #   Reset
        self.translation_node.setSFVec3f(initial_position)
        self.robot_node.resetPhysics()
        #self.robot.simulationResetPhysics()
        self.episode_timestep = 0

        # state update
        info = {}
        if return_info:
            info = {'current_min_distance': self.left_value}
        action = np.array([0.0, 0.0], dtype=np.float32)
        return self.step(action)[0], info

    def calculate_reward(self, state, linear_vel_norm, angular_vel_norm, desired_distance=0.15):
        """
        Computes the reward based on the current state, linear velocity norm, and angular velocity norm.

        Parameters:
        - state: Current state containing relevant values.
        - linear_vel_norm: Normalized linear velocity.
        - angular_vel_norm: Normalized angular velocity.
        - desired_distance: The desired distance for calculations (default is 0.15).

        Returns:
        - total_reward: The calculated total reward.
        - for factor = 1 the max reward is 3 and min -3
        """

        if self.bumper.getValue() == 1.0 or (np.min(state) <= (self.min_range + 0.005)):
            return -10

        left_value = state[0]

        def following_reward(x,factor=2):
            # Maximum value is 1, minimum value is -1
            return (-(800 / 9) * (x - desired_distance) ** 2 + 1) * factor

        def linear_velocity_reward(factor=1):
            # Reward based on the linear velocity norm
            return linear_vel_norm * factor

        def angular_smoothness_reward(x, y, factor=1):
            # Smoothness reward calculation
            return (1 - 2 * (x - y) ** 4) * factor

        # Calculate total reward by combining individual rewards
        total_reward = (
                following_reward(left_value) +
                linear_velocity_reward() +
                angular_smoothness_reward(angular_vel_norm, self.prev_angular_vel_norm)
        )

        # update
        self.prev_angular_vel_norm = angular_vel_norm

        return total_reward

    def is_done(self, state):
        # check for collision
        if self.bumper.getValue() == 1.0:
            print("Bumper true")
            return True

        # check for min range to avoid infinity misundertand
        else:
            print("Bumper false")
            for i in state:
                if i <= (self.min_range + 0.005):
                    print("object too close")
                    return True

        # Otherwise, the episode continues
        return False

    def render(self, mode='human'):
        # Implemente a lógica de renderização aqui, se necessário
        pass


