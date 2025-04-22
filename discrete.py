import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple
from controller import Supervisor, Motor
from utils import cmd_vel

class WebotsWallFollowing(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(WebotsWallFollowing, self).__init__()

        # Training requirements debug and logging
        self.left_value = 0                 # For tensorboard
        self.max_episode_timestep = 5000    # Define a maximum number of timesteps per episode
        self.episode_timestep = 0           # Initialize current timestep counter
        self.randomize = False              # True for training, False for testing

        # Webots configs
        self.robot = Supervisor()
        if self.robot is None:
            raise ValueError("Robot 'e-puck' not found.")
        self.timestep = int(self.robot.getBasicTimeStep())

        self.left_motor: Motor = self.robot.getDevice('left wheel motor')
        self.right_motor: Motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Lidar and Bumper Sensors
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.robot.step()
        print("is lidar point cloud enabled?", self.lidar.isPointCloudEnabled())

        # Nodes
        self.robot_node = self.robot.getFromDef("robot")
        self.translation_node = self.robot_node.getField("translation")

        self.robot.step()
        self.initial_state = np.array(self.lidar.getRangeImage(), dtype=np.float32)

        self.bumper = self.robot.getDevice('touch sensor')
        self.bumper.enable(self.timestep)

        # Bounds for observation space
        self.min_range = self.lidar.getMinRange()  # 0.05
        self.max_range = self.lidar.getMaxRange()  # 0.3

        # Action space
        self.action_space = spaces.Discrete(3)

        # Observation space (normalized)
        num_lidar_readings = self.lidar.getHorizontalResolution()
        print("num_lidar_readings:" + str(num_lidar_readings))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_lidar_readings,),
            dtype=np.float32
        )

        self.max_linear_vel = 0.1256
        self.max_angular_vel = 4.8307
        self.previous_action = None

    def decode_action(self, action):
        # Define and execute the actions
        if action == 0:         # Move forward
            print("MOVE FORWARD")
            cmd_vel(self.robot, self.max_linear_vel, 0)
        elif action == 1:       # Turn left
            print("TURN LEFT")
            cmd_vel(self.robot, 0, self.max_angular_vel)
        elif action == 2:       # Turn right
            print("TURN RIGHT")
            cmd_vel(self.robot, 0, -self.max_angular_vel)

    def clip_observation(self, observation):
        #   First, handle infinities if any
        processed_observation = np.where(np.isinf(observation), self.max_range, observation)
        return processed_observation.astype(np.float32)

    def normalize_observation(self, observation):
        # Normalize to 0 to 1
        normalized_observation = (observation - self.min_range) / (self.max_range - self.min_range)

        # Scale from 0 to 1 to -1 to 1
        scaled_observation = (normalized_observation * 2) - 1
        return scaled_observation.astype(np.float32)

    def step(self, action):
        # Apply action
        self.decode_action(action)
        self.robot.step(self.timestep)

        # Get Lidar values and clip them
        state = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        state = self.clip_observation(state)
        self.left_value = state[0]

        # Reward
        reward = self.calculate_reward(state, action)

        # Update
        self.episode_timestep += 1  # Increment timestep with each call to step
        print("episode timestep", self.episode_timestep)

        # Check if done
        done = self.is_done(state)
        truncated = self.episode_timestep >= self.max_episode_timestep
        if done or truncated:
            done = True

        # Normalization required and tensorboard logging info
        state = self.normalize_observation(state)
        info = {'current_min_distance': self.left_value}

        return state, reward, done, False, info

    def reset(self, seed: Optional[int] = None, return_info: bool = False) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Randomization or not
        # positions = [
        #    [0.0476146, 0.49136, -0.0000319338],
        #    [-0.506447, 0.293503, -0.0000319338],
        #    [-0.129066, 0.0899107, -0.0000319338]
        # ]

        positions2 = [
            [-0.158667, -0.35101, -0.0000319338],
            [-0.311151, 0.32963, -0.0000319338],
            [-0.504659, -0.134969, -0.0000319338]
        ]

        initial_position = positions2[np.random.choice(len(positions2))] if self.randomize else [-0.325, -0.243, -3.08235e-05]

        print(initial_position)
        #   Reset
        self.translation_node.setSFVec3f(initial_position)
        self.robot_node.resetPhysics()
        # self.robot.simulationResetPhysics()

       # self.robot.simulationReset()

        self.episode_timestep = 0
        self.robot.step(self.timestep)
        self.previous_action = None
        state = self.initial_state
        state = self.clip_observation(state)
        state = self.normalize_observation(state)

        info = {}
        if return_info:
            info = {'current_min_distance': self.left_value}

        return state, info

    """
    def calculate_reward(self, state, action):
        print("previous_action", self.previous_action)
        print("current_action", action)
        if self.bumper.getValue() == 1.0 or (np.min(state) <= (self.min_range + 0.005)):
            return -100

        desired_distance = 0.15
        optimal_index = 0
        measured_distance = state[optimal_index]
        print("Distância minima (esquerda):", measured_distance)

        # Positive reward for maintaining optimal distance -> Smaller the distance error, higher the reward
        distance_error = (measured_distance - desired_distance) ** 2  # Penalizing the square of the error
        distance_reward = 100 - 1000 * distance_error
        positive_reward = max(0, distance_reward)  # Adding a positive reward when the error is small

        # Penalidade por mudar de direção frequentemente
        not_moving_penalty = 0
        if self.previous_action is not None and action != 0 and self.previous_action != 0:
            not_moving_penalty = -15  # Penalidade pequena para ações de rotação consecutivas

        # Recompensa para movimento contínuo para frente
        moving_forward_reward = 5 if action == 0 else 0

        # Aggregate total reward
        total_reward = positive_reward + not_moving_penalty + moving_forward_reward
        print("Total Reward: ", total_reward)

        # Update previous action
        self.previous_action = action
        return total_reward
    """

    def calculate_reward(self, state, action, desired_distance=0.15):
        """
        Computes the reward based on the current state and current action.

        Parameters:
        - state: Current state containing relevant values.
        - action: The action that was just chosen.
        - desired_distance: The desired distance for calculations (default is 0.15).

        Returns:
        - total_reward: The calculated total reward.
        - for factor = 1 the max reward is 3 and min -3
        """

        if self.bumper.getValue() == 1.0 or (np.min(state) <= (self.min_range + 0.005)):
            return -10

        left_value = state[0]

        def following_reward(x, factor=2):
            # Maximum value is 1, minimum value is -1
            return (-(800 / 9) * (x - desired_distance) ** 2 + 1) * factor


        # Calculate total reward by combining individual rewards
        total_reward = (
                following_reward(left_value)
        )

        # update
        self.previous_action = action

        return total_reward

    def is_done(self, state):
        # Check for collision
        if self.bumper.getValue() == 1.0:
            print("Bumper true")
            return True
        # Check for min range to avoid infinity misunderstanding
        else:
            print("Bumper false")
            for i in state:
                if i <= (self.min_range + 0.005):
                    print("Object too close")
                    return True
        # Otherwise, the episode continues
        return False

    def render(self, mode='human'):
        # Implemente a lógica de renderização aqui, se necessário
        pass



