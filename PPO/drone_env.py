import gym
import math
from gym import spaces
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from stable_baselines3 import PPO
from drone import Drone
from obstacle import Obstacle
from spatial_grid import SpatialGrid

ROTATION_SPEED = 3
MAX_VELOCITY = 0.2
ACCELERATION = 0.01
DECELERATION = 0.005
SIZE = 0.5
BOUNDARY_TOLERANCE = 0



class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.goal_position = np.array([50,3,50])
        self.latest_reward = 0
        self.drone = Drone([0, 2, 0], [0.0, 0.0, 0.0], 0)
        self.obstacles = self.generate_obstacles(45)
        self.spatial_grid = SpatialGrid(cell_size=10)
        for obstacle in self.obstacles:
            self.spatial_grid.add_obstacle(obstacle)
        # State space: [x, y, z, heading, vx, vy, vz, angular_vel]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )
        # Define action space for all possible movements
        # 0: no action
        # 1: move up     2: move down
        # 3: move left   4: move right
        # 5: move fwd    6: move back
        # 7: rotate left 8: rotate right
        self.action_space = gym.spaces.Discrete(9)
    """
    def reset(self, seed=None):
    # Reset drone to initial state using its own reset method
    self.drone.reset()  # Implement this in your Drone class if not exists
    return self.get_observation(), {}
    """

    def check_collision(self, radius):

        drone_min_x, drone_max_x = self.drone.position[0] - (SIZE / 4 + BOUNDARY_TOLERANCE), \
                                   self.drone.position[0] + (SIZE / 4 + BOUNDARY_TOLERANCE)
        drone_min_y, drone_max_y = self.drone.position[1], self.drone.position[1] + (
                SIZE / 2 + BOUNDARY_TOLERANCE)
        drone_min_z, drone_max_z = self.drone.position[2] - (SIZE / 2 + BOUNDARY_TOLERANCE), \
                                   self.drone.position[2] + (SIZE / 2 + BOUNDARY_TOLERANCE)

        obstacle_min_x, obstacle_max_x = -100, 100
        obstacle_min_y, obstacle_max_y = -1, 0
        obstacle_min_z, obstacle_max_z = -100, 100

        if (drone_min_x <= obstacle_max_x and drone_max_x >= obstacle_min_x) and \
                (drone_min_y <= obstacle_max_y and drone_max_y >= obstacle_min_y) and \
                (drone_min_z <= obstacle_max_z and drone_max_z >= obstacle_min_z):
            return True

        # Assuming `detect_nearby_obstacles` returns a list of Obstacle objects
        obstacles = self.spatial_grid.detect_nearby_obstacles(self.drone, SIZE, radius)

        for obstacle in obstacles:
            # Access the position and size attributes directly from the Obstacle object
            obstacle_x, obstacle_y, obstacle_z = obstacle.position
            obstacle_size = obstacle.size

            obstacle_min_x, obstacle_max_x = obstacle_x - (obstacle_size / 2 + BOUNDARY_TOLERANCE), obstacle_x + (
                    obstacle_size / 2 + BOUNDARY_TOLERANCE)
            obstacle_min_y, obstacle_max_y = 0, obstacle_y + BOUNDARY_TOLERANCE
            obstacle_min_z, obstacle_max_z = obstacle_z - (obstacle_size / 2 + BOUNDARY_TOLERANCE), obstacle_z + (
                    obstacle_size / 2 + BOUNDARY_TOLERANCE)

            if (drone_min_x <= obstacle_max_x and drone_max_x >= obstacle_min_x) and \
                    (drone_min_y <= obstacle_max_y and drone_max_y >= obstacle_min_y) and \
                    (drone_min_z <= obstacle_max_z and drone_max_z >= obstacle_min_z):
                return True
        return False

    def get_observation(self):
        # Use drone's actual state including velocity and acceleration
        return np.array([self.drone.position[0],self.drone.position[1],self.drone.position[2],
                self.drone.heading,
                self.drone.velocity[0], self.drone.velocity[1], self.drone.velocity[2]])
    def do_nothing(self):
        self.drone.update_position(DECELERATION, False, self.check_collision(5))
    def move_up(self):
        delta_v = [self.drone.velocity[0], self.drone.velocity[1] + ACCELERATION, self.drone.velocity[2]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(5))

    def move_down(self):
        delta_v = [self.drone.velocity[0], self.drone.velocity[1] - ACCELERATION, self.drone.velocity[2]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(5))

    def move_left(self, rotation_matrix):
        delta_a = np.matmul(rotation_matrix, [-ACCELERATION, 0])
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(5))

    def move_right(self, rotation_matrix):
        delta_a = np.matmul(rotation_matrix, [ACCELERATION, 0])
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(5))

    def move_forward(self, rotation_matrix):
        delta_a = np.matmul(rotation_matrix, [0, -ACCELERATION]).tolist()
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(5))

    def move_backward(self, rotation_matrix):
        delta_a = np.matmul(rotation_matrix, [0, ACCELERATION])
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(5))

    def rotate_left(self):
        self.drone.heading -= ROTATION_SPEED
        self.drone.update_position(DECELERATION, False, self.check_collision(5))

    def rotate_right(self):
        self.drone.heading += ROTATION_SPEED
        self.drone.update_position(DECELERATION, False, self.check_collision(5))

    def step(self, action):
        rotation_matrix = np.array([[np.cos(np.radians(self.drone.heading)), -np.sin(np.radians(self.drone.heading))],
                                    [np.sin(np.radians(self.drone.heading)), np.cos(np.radians(self.drone.heading))]])
        reward = self.calculate_reward(action)
        self.latest_reward = reward
        if self.latest_reward == 100 or self.latest_reward == -10 or self.latest_reward == -12 or self.latest_reward == -14 or self.latest_reward == -16:
            action = 0
        elif self.latest_reward == -18:
            action = 1
        elif self.latest_reward == -8:
            action = 2
        #Apply the action using drone's existing methods
        if action == 0:
            self.do_nothing()
        elif action == 1:
            self.move_up()
        elif action == 2:
            self.move_down()
        elif action == 3:
            self.move_left(rotation_matrix)
        elif action == 4:
            self.move_right(rotation_matrix)
        elif action == 5:
            self.move_forward(rotation_matrix)
        elif action == 6:
            self.move_backward(rotation_matrix)
        elif action == 7:
            self.rotate_left()
        elif action == 8:
            self.rotate_right()

        reward = self.calculate_reward(action)
        # Get observation after action
        obs = self.get_observation()
        # Check if episode is done
        done = False  # Define your termination conditions
        info = {}

        return obs, reward, done, False, {}
    def calculate_reward(self, action):
        # Calculate and return the reward for the given action
        reward = 0  # Define your reward function
        # Calculate the direction to the goal
        prev_distance_to_goal = self.goal_position - self.drone.last_position
        prev_distance_to_goal = np.array([prev_distance_to_goal[0], prev_distance_to_goal[2]])
        distance_to_goal = self.goal_position - self.drone.position
        distance_to_goal = np.array([distance_to_goal[0], distance_to_goal[2]])
        direction_to_goal = distance_to_goal / np.linalg.norm(distance_to_goal)  # Normalize the vector
        print(direction_to_goal)
        # Calculate the drone's heading vector
        heading_radians = np.radians(self.drone.heading)
        drone_heading_vector = np.array([-np.cos(heading_radians), np.sin(heading_radians)])
        drone_heading_vector /= np.linalg.norm(drone_heading_vector)
        print(drone_heading_vector)
        # Calculate alignment using dot product and velocity using norm
        alignment = np.dot(drone_heading_vector, direction_to_goal)
        velocity_magnitude = np.linalg.norm(self.drone.velocity)
        # Reward for heading towards the goal
        reward += alignment * 3  # Scale the reward as needed
        reward += pow(np.linalg.norm(distance_to_goal), -1) * 5
        # if alignment correct increase reward
        print(self.drone.position)
        if np.linalg.norm(distance_to_goal) - np.linalg.norm(prev_distance_to_goal) < -.2:
            reward += pow(np.linalg.norm(distance_to_goal) - np.linalg.norm(prev_distance_to_goal),-1) * 20
        else:
            reward += -.1
        # if velocity less then 55% correct then decrease reward
        if np.dot([self.drone.velocity[0], self.drone.velocity[2]], direction_to_goal) < .55:
            reward += -velocity_magnitude
            print("not correct velocity vector")
        # Reward for reaching the goal
        if np.linalg.norm(self.drone.position - self.goal_position) < SIZE:
            return 100.0
        # if drone is too low command it to go up
        if self.drone.position[1] < 2:
            print("drone too low!")
            return -18.0
        # if alignment and velocity vector are correct increase reward
        if np.dot([self.drone.velocity[0], self.drone.velocity[2]], direction_to_goal) > .75:
            reward += velocity_magnitude * 5
            print("proceeding in correct direction")
        # if drone is too high make it go down
        if self.drone.position[1] > 10:
            print("drone too high!")
            return -8.0
            # Favor action 8
        if self.drone.position[0] > 90:
            print("drone too far!")
            return -10.0
            # Favor action 8
        if self.drone.position[0] < -90:
            print("drone too far!")
            return -12.0
            # Favor action 8
        if self.drone.position[2] > 90:
            print("drone too far!")
            return -14.0
            # Favor action 8
        if self.drone.position[2] < -90:
            print("drone too far!")
            return -16.0
            # Favor action 8
        if self.spatial_grid.nearby_obstacles(self.drone, SIZE, .4):
            print("nearby obstacle detected")
            reward += -pow(self.spatial_grid.nearby_obstacle_distance(self.drone, .4),-1) * 20
        if alignment > 0.99:
            print("correct alignment")
            reward += alignment * 4

        print(reward)
        return reward

    def get_latest_reward(self):
        # Return the latest reward
        return self.latest_reward

    def render(self, mode='human'):
        self.game.run()

    @staticmethod
    def draw_ground_plane():
        glColor4f(0.5, 0.5, 0.5, 0.5)
        glBegin(GL_QUADS)
        glVertex3f(-100, 0, -100)
        glVertex3f(100, 0, -100)
        glVertex3f(100, 0, 100)
        glVertex3f(-100, 0, 100)
        glEnd()

    @staticmethod
    def draw_goal_box(vertices, edges):
        glColor4f(1, 1, 0, 1)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    @staticmethod
    def generate_obstacles(num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            size = random.uniform(0.5, 2.5)
            height = random.uniform(2, 10.0)
            x = random.uniform(-50, 50)
            z = random.uniform(-50, 50)
            obstacle = Obstacle([x, height, z], size)
            obstacles.append(obstacle)
        return obstacles