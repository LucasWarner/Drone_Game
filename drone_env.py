import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from drone import Drone

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()\

        self.drone = Drone()

        # State space: [x, y, z, heading, vx, vy, vz, angular_vel]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        # Define action space for all possible movements
        # 0: no action
        # 1: move up     2: move down
        # 3: move left   4: move right
        # 5: move fwd    6: move back
        # 7: rotate left 8: rotate right
        self.action_space = gym.spaces.Discrete(9)

    def reset(self, seed=None):
        # Reset drone to initial state using its own reset method
        self.drone.reset()  # Implement this in your Drone class if not exists
        return self.get_observation(), {}

    def get_observation(self):
        # Use drone's actual state including velocity and acceleration
        return np.array([
            self.drone.x,
            self.drone.y,
            self.drone.z,
            self.drone.heading,
            self.drone.velocity_x,
            self.drone.velocity_y,
            self.drone.velocity_z,
        ], dtype=np.float32)

    def move_up(self):
        self.drone.move_up()

    def move_down(self):
        self.drone.move_down()

    def move_left(self):
        self.drone.move_left()

    def move_right(self):
        self.drone.move_right()

    def move_forward(self):
        self.drone.move_forward()

    def move_backward(self):
        self.drone.move_backward()

    def rotate_left(self):
        self.drone.rotate_left()

    def rotate_right(self):
        self.drone.rotate_right()

    def step(self, action):
        # Apply the action using drone's existing methods
        if action == 1:
            self.move_up()
        elif action == 2:
            self.move_down()
        elif action == 3:
            self.move_left()
        elif action == 4:
            self.move_right()
        elif action == 5:
            self.move_forward()
        elif action == 6:
            self.move_backward()
        elif action == 7:
            self.rotate_left()
        elif action == 8:
            self.rotate_right()

        # Update drone physics (assuming your Drone class has an update method)
        self.drone.update()  # This should handle acceleration/deceleration

        # Get observation after action
        obs = self.get_observation()

        # Calculate reward
        reward = 0  # Define your reward function

        # Check if episode is done
        done = False  # Define your termination conditions

        return obs, reward, done, False, {}
    def render(self, mode='human'):
        self.game.run()