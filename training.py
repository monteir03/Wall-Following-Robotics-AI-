import os
from stable_baselines3 import DQN, PPO, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from discrete import *  # change to continuous if you to run continuous action space

"""
If you use DQN, you must import discrete action space.
If you use TD3, you must import continuous action space.

Change paths and model calls accordingly to your choice of model and action space.
"""

class LogMinDistanceCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LogMinDistanceCallback, self).__init__(verbose)
        self.left_values = []

    def _on_step(self):
        # Record the minimum distance from the wall
        left_value = self.training_env.get_attr('left_value')[0]
        self.left_values.append(left_value)
        return True

    def _on_rollout_end(self):
        # Compute the mean of minimum distances for the episode
        mean_left_value = np.mean(self.left_values)
        # Log to TensorBoard
        self.logger.record('metrics/mean_left_value', mean_left_value)
        self.left_values = []  # Reset for the next episode


def main():
    # Define directories for saving models and logs
    models_dir = "models/map_X_discrete_random/DQN"   # Verify
    logdir = "logs/map_X_discrete_random/DQN"         # Verify

    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Create and check the environment
    env = WebotsWallFollowing()
    check_env(env, warn=True)

    # Initialize the  model with TensorBoard logging
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir) # Verify

    # Create the custom callback instance
    min_distance_callback = LogMinDistanceCallback()

    # Define training parameters
    time_steps = 5000            # learning timestep per iteration
    total_time_steps = 100000

    # Train the model and save checkpoints
    for i in range(total_time_steps // time_steps):

        # each 5000
        model.learn(
            total_timesteps=time_steps, reset_num_timesteps=False,
            tb_log_name="Discrete_DQN_WorldX",                  # Verify
            callback=min_distance_callback
            )

        # Save the model
        model.save(f"{models_dir}/{time_steps * (i + 1)}")
        print(f"Iteration: {i + 1}, Model saved at: {models_dir}/{time_steps * (i + 1)}")


if __name__ == "__main__":
    main()
