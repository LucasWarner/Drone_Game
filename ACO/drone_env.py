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
        self.is_goal_reached = False
        self.latest_reward = 0
        self.previous_action = 0
        self.drone = Drone([0, 3, 0], [0.0, 0.0, 0.0], 0)
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
    def calculate_new_position(self, action):
        reward, done = self.step(action)
        return self.drone.position, reward
    def calculate_rm(self):
        return np.array([[np.cos(np.radians(self.drone.heading)), -np.sin(np.radians(self.drone.heading))],
                                    [np.sin(np.radians(self.drone.heading)), np.cos(np.radians(self.drone.heading))]])
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

    def get_observation(self, ant):
        # Use drone's actual state including velocity and acceleration
        return np.array([self.drone.position[0],self.drone.position[1],self.drone.position[2],
                self.drone.heading,
                self.drone.velocity[0], self.drone.velocity[1], self.drone.velocity[2]])
    def do_nothing(self):
        self.drone.update_position(DECELERATION, False, self.check_collision(.4))
    def move_up(self):
        delta_v = [self.drone.velocity[0], self.drone.velocity[1] + ACCELERATION, self.drone.velocity[2]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(.4))

    def move_down(self):
        delta_v = [self.drone.velocity[0], self.drone.velocity[1] - ACCELERATION, self.drone.velocity[2]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(.4))

    def move_left(self):
        rotation_matrix = self.calculate_rm()
        delta_a = np.matmul(rotation_matrix, [-ACCELERATION, 0])
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(.4))

    def move_right(self):
        rotation_matrix = self.calculate_rm()
        delta_a = np.matmul(rotation_matrix, [ACCELERATION, 0])
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(.4))

    def move_forward(self):
        rotation_matrix = self.calculate_rm()
        delta_a = np.matmul(rotation_matrix, [0, -ACCELERATION]).tolist()
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(.4))

    def move_backward(self):
        rotation_matrix = self.calculate_rm()
        delta_a = np.matmul(rotation_matrix, [0, ACCELERATION])
        delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
        self.drone.update_velocity(delta_v, MAX_VELOCITY)
        self.drone.update_position(DECELERATION, True, self.check_collision(.4))

    def rotate_left(self):
        self.drone.heading -= ROTATION_SPEED
        self.drone.update_position(DECELERATION, False, self.check_collision(.4))

    def rotate_right(self):
        self.drone.heading += ROTATION_SPEED
        self.drone.update_position(DECELERATION, False, self.check_collision(.4))

    def step(self, action):


        reward = self.calculate_reward(action, self.latest_reward)

        if reward == 100 or reward == -10 or reward == -12 or reward == -14 or reward == -16:
            action = 0
        elif reward == -18:
            action = 1
        elif reward == -8:
            action = 2
        #Apply the action using drone's existing methods
        if action == 0:
            self.do_nothing()
        elif action == 1:
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

        self.latest_reward = reward
        # Check if episode is done
        done = False  # Define your termination conditions

        return reward, done
    def calculate_reward(self, action, prev_reward):
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
        velocitiy_vector = [self.drone.velocity[0], self.drone.velocity[2]]
        velocity_magnitude = np.linalg.norm(self.drone.velocity)
        velocitiy_vector /= velocity_magnitude
        # Reward for heading towards the goal
        reward += alignment * 3  # Scale the reward as needed
        reward += pow(np.linalg.norm(distance_to_goal), -1) * 10
        # if alignment correct increase reward
        if np.linalg.norm(distance_to_goal) - np.linalg.norm(prev_distance_to_goal) < -.1:
            reward += 2
        else:
            reward += -2
        # if velocity less then 55% correct then decrease reward
        if np.dot(velocitiy_vector, direction_to_goal) < .55:
            reward += -velocity_magnitude
            print("not correct velocity vector")
        # Reward for reaching the goal
        if np.linalg.norm(self.drone.position - self.goal_position) < SIZE:
            self.is_goal_reached = True
            return 100.0
        # if drone is too low command it to go up
        if self.drone.position[1] < 2:
            print("drone too low!")
            self.move_up()
        # if alignment and velocity vector are correct increase reward
        if np.dot(velocitiy_vector , direction_to_goal) > .75:
            reward += 3
            print("proceeding in correct direction")
        # if drone is too high make it go down
        if self.drone.position[1] > 10:
            print("drone too high!")
            self.move_down()
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
            reward += -pow(self.spatial_grid.nearby_obstacle_distance(self.drone, .4),-1) * 100
        if 0.99 < alignment < 0.9992:
            print("almost correct alignment")
            reward += 4
        if alignment > 0.9992:
            print("correct alignment")
            reward += 5
            if random.randint(0,1) == 1:
                self.move_forward()
        """
        if prev_reward > reward:
            if random.uniform(0,1) < .5:
                next_action = action
            else:
                next_action = random.randint(0,8)
        else:
            next_action = random.randint(0,8)
        """
        # Reward shaping based on opposite actions
        if action == 7 and self.previous_action == 8:
            reward += -3  # Penalize for taking opposite actions consecutively
        elif action == 8 and self.previous_action == 7:
            reward += -3  # Penalize for taking opposite actions consecutively
        if action == 5 and self.previous_action == 6:
            reward += -3  # Penalize for taking opposite actions consecutively
        elif action == 6 and self.previous_action == 5:
            reward += -3  # Penalize for taking opposite actions consecutively
        if action == 3 and self.previous_action == 4:
            reward += -3  # Penalize for taking opposite actions consecutively
        elif action == 4 and self.previous_action == 3:
            reward += -3  # Penalize for taking opposite actions consecutively
        if action == 1 and self.previous_action == 2:
            reward += -3  # Penalize for taking opposite actions consecutively
        elif action == 2 and self.previous_action == 1:
            reward += -3  # Penalize for taking opposite actions consecutively
        if action == self.previous_action:
            reward += 1

        self.previous_action = action
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