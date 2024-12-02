import pygame
import time
from pygame.locals import DOUBLEBUF, OPENGL
import random
import numpy as np
from drone import Drone
from obstacle import Obstacle
from spatial_grid import SpatialGrid
from DQL_agent import DQLAgent
from OpenGL.GL import *
from OpenGL.GLU import *

WIDTH, HEIGHT = 800, 600
SIZE = 0.5
ROTATION_SPEED = 2
ACCELERATION = 0.01
DECELERATION = 0.005
MAX_VELOCITY = 0.2
BOUNDARY_TOLERANCE = 0.05

vertices = [
    [SIZE, SIZE * 2, -SIZE],
    [SIZE, 0, -SIZE],
    [-SIZE, 0, -SIZE],
    [-SIZE, SIZE * 2, -SIZE],
    [SIZE, SIZE * 2, SIZE],
    [SIZE, 0, SIZE],
    [-SIZE, 0, SIZE],
    [-SIZE, SIZE * 2, SIZE]
]

edges = [
    (0, 1),
    (2, 3),
    (3, 0),
    (4, 5),
    (6, 7),
    (7, 4),
    (0, 4),
    (3, 7)
]

tri_vertices = [
    (0.0, 0, -SIZE),    # front vertex
    (-SIZE/2, 0, 0),  # Bottom left vertex
    (SIZE/2, 0, 0),   # Bottom right vertex
    (0.0, SIZE/2, 0)    # Bottom top vertex
]

tri_edges = [
    (0, 1),  # Top vertex to bottom left vertex
    (0, 2),  # Top vertex to bottom right vertex
    (0, 3),  # Top vertex to bottom front vertex
    (1, 2),  # Bottom left vertex to bottom right vertex
    (2, 3),  # Bottom right vertex to bottom front vertex
    (3, 1)   # Bottom front vertex to bottom left vertex
]
class Game:
    def __init__(self):
        self.drone = Drone([0, .2, 0], [0.0, 0.0, 0.0], 0)
        self.obstacles = self.generate_obstacles(45)
        self.spatial_grid = SpatialGrid(cell_size=10)
        for obstacle in self.obstacles:
            self.spatial_grid.add_obstacle(obstacle)
        self.agent = DQLAgent(state_size=6, action_size=9)  # Example state and action sizes
        self.state = self.get_state()
        self.done = False
        self.goal_position = np.array([50, 5, 50])

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

    def check_collision(self, radius):

        drone_min_x, drone_max_x = self.drone.position[0] - (SIZE / 4 + BOUNDARY_TOLERANCE), \
                                   self.drone.position[0] + (SIZE / 4 + BOUNDARY_TOLERANCE)
        drone_min_y, drone_max_y = self.drone.position[1], self.drone.position[1] + (
                SIZE / 2 + BOUNDARY_TOLERANCE)
        drone_min_z, drone_max_z = self.drone.position[2] - (SIZE / 2 + BOUNDARY_TOLERANCE), \
                                   self.drone.position[2] + (SIZE / 2 + BOUNDARY_TOLERANCE)

        obstacle_min_x, obstacle_max_x = -100, 100
        obstacle_min_y, obstacle_max_y = -10, 0
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

    def run(self):
        pygame.init()
        display = (WIDTH, HEIGHT)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)

        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            action = self.agent.act(self.state)
            self.perform_action(action)
            next_state = self.get_state()
            reward = self.calculate_reward()
            self.agent.remember(self.state, action, reward, next_state, self.done)
            self.state = next_state

            if len(self.agent.memory) > 32:
                self.agent.replay(32)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.update_view()
            self.draw_ground_plane()
            for obstacle in self.obstacles:
                obstacle.draw(vertices, edges)

            # Check if the drone has reached the goal
            if np.linalg.norm(self.drone.position - self.goal_position) < SIZE:
                print("Goal reached!")
                self.done = True

            pygame.display.flip()
            pygame.time.wait(15)

    def update_view(self):
        rotation_matrix = np.array([[np.cos(np.radians(self.drone.heading)), -np.sin(np.radians(self.drone.heading))],
                                    [np.sin(np.radians(self.drone.heading)), np.cos(np.radians(self.drone.heading))]])
        # Rotate the drone's shape based on the heading around its relative origin
        glPushMatrix()
        glTranslatef(*self.drone.position)
        glRotatef(-self.drone.heading, 0, 1, 0)  # Rotate around the y-axis based on heading
        self.drone.draw(tri_vertices, tri_edges)  # Draw the rotated drone shape
        glPopMatrix()

        camera_matrix = [0, 3]

        # Define the camera position relative to the drone
        rotated_camera = np.matmul(rotation_matrix, camera_matrix)

        camera_position = (rotated_camera[0] + self.drone.position[0],
                           self.drone.position[1],
                           rotated_camera[1] + self.drone.position[2])

        # Rotate the camera position with the drone
        glLoadIdentity()
        gluPerspective(60, (WIDTH / HEIGHT), 0.1, 100)
        gluLookAt(camera_position[0], camera_position[1] + .5, camera_position[2],  # Camera position
                  self.drone.position[0], self.drone.position[1], self.drone.position[2],  # Look at the drone
                  0, 1, 0)  # Up vector

    def get_state(self):
        # Example state representation
        return np.array(self.drone.position + self.drone.velocity)

    def perform_action(self, action):

        rotation_matrix = np.array([[np.cos(np.radians(self.drone.heading)), -np.sin(np.radians(self.drone.heading))],
                                    [np.sin(np.radians(self.drone.heading)), np.cos(np.radians(self.drone.heading))]])

        # Map action index to drone control
        if action == 0:
            delta_a = np.matmul(rotation_matrix, [0, -ACCELERATION]).tolist()
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY, self.check_collision(5))  # Move forward
            self.drone.update_position(DECELERATION, True, self.check_collision(5))
        elif action == 1:
            self.drone.heading += ROTATION_SPEED
            self.drone.update_position(DECELERATION, False, self.check_collision(5))
        elif action == 2:
            self.drone.heading -= ROTATION_SPEED
            self.drone.update_position(DECELERATION, False, self.check_collision(5))
        elif action == 3:
            delta_a = np.matmul(rotation_matrix, [0, ACCELERATION])
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY, self.check_collision(5))  # Move backward
            self.drone.update_position(DECELERATION, True, self.check_collision(5))
        elif action == 4:
            delta_v = [self.drone.velocity[0], self.drone.velocity[1] + ACCELERATION, self.drone.velocity[2]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY, self.check_collision(5)) # Move up
            self.drone.update_position(DECELERATION, True, self.check_collision(5))
        elif action == 5:
            delta_v = [self.drone.velocity[0], self.drone.velocity[1] - ACCELERATION, self.drone.velocity[2]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY)  # Move down
            self.drone.update_position(DECELERATION, True, self.check_collision(5))
        elif action == 6:
            delta_a = np.matmul(rotation_matrix, [-ACCELERATION, 0])
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY, self.check_collision(5))
            self.drone.update_position(DECELERATION, True, self.check_collision(5))
        elif action == 7:
            delta_a = np.matmul(rotation_matrix, [ACCELERATION, 0])
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY, self.check_collision(5))
            self.drone.update_position(DECELERATION, True, self.check_collision(5))
        elif action == 8:
            self.drone.update_position(DECELERATION, False, self.check_collision(5))

    def calculate_reward(self):
        reward = 0
        # Check for collision


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
        reward += alignment * 2  # Scale the reward as needed
        reward += pow(np.linalg.norm(distance_to_goal),-1)
        # if alignment correct increase reward
        print(self.drone.position)
        if np.linalg.norm(distance_to_goal) - np.linalg.norm(prev_distance_to_goal) < -.2:
            reward += .5
        else:
            reward += -.1
        # if velocity less then 55% correct then decrease reward
        if np.dot([self.drone.velocity[0], self.drone.velocity[2]], direction_to_goal) < .55:
            reward += -velocity_magnitude * 5
            print("not correct velocity vector")
        # Reward for reaching the goal
        if np.linalg.norm(self.drone.position - self.goal_position) < SIZE:
            return 100
        #if drone is too low command it to go up
        if self.drone.position[1] < 3.5:
            print("drone too low!")
            return -18
        #if alignment and velocity vector are correct increase reward
        if alignment > 0.9 and np.dot([self.drone.velocity[0], self.drone.velocity[2]], direction_to_goal) > .75:
            reward += alignment + velocity_magnitude * 20
            print("aligned and proceeding in correct direction")
        #if drone is too high make it go down
        if self.drone.position[1] > 10:
            print("drone too high!")
            return -8
            # Favor action 8
        if self.drone.position[0] > 90:
            print("drone too far!")
            return -10
            # Favor action 8
        if self.drone.position[0] < -90:
            print("drone too far!")
            return -12
            # Favor action 8
        if self.drone.position[2] > 90:
            print("drone too far!")
            return -14
            # Favor action 8
        if self.drone.position[2] < -90:
            print("drone too far!")
            return -16
            # Favor action 8

        if self.spatial_grid.nearby_obstacles(self.drone, SIZE, 2):
            print("nearby obstacle detected")
            return -2
        if alignment > 0.999:
            print("correct alignment")
            return 10

        print(reward)
        # Combine rewards
        return reward
    def draw_ground_plane(self):
        glColor4f(0.5, 0.5, 0.5, 0.5)
        glBegin(GL_QUADS)
        glVertex3f(-100, 0, -100)
        glVertex3f(100, 0, -100)
        glVertex3f(100, 0, 100)
        glVertex3f(-100, 0, 100)
        glEnd()

        # Convert lists to numpy arrays for element-wise operations
        size_array = np.array([SIZE, self.goal_position[1], SIZE])
        vertex1 = self.goal_position - size_array
        vertex2 = self.goal_position - np.array([-SIZE, self.goal_position[1], SIZE])
        vertex3 = self.goal_position + size_array
        vertex4 = self.goal_position + np.array([-SIZE, self.goal_position[1], SIZE])

        glColor4f(1, 1, 0, 1)
        glBegin(GL_QUADS)
        glVertex3f(*vertex1)
        glVertex3f(*vertex2)
        glVertex3f(*vertex3)
        glVertex3f(*vertex4)
        glEnd()