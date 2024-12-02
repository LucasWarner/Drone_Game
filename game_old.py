import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import random
import numpy as np
from drone import Drone
from obstacle import Obstacle
from spatial_grid import SpatialGrid
from OpenGL.GL import *
from OpenGL.GLU import *


WIDTH, HEIGHT = 800, 600
SIZE = 0.5
ROTATION_SPEED = 3
ACCELERATION = 0.01
DECELERATION = 0.005
MAX_VELOCITY = 0.2
BOUNDARY_TOLERANCE = 0

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
        self.key_pressed = False

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

    def run(self):
        pygame.init()
        display = (WIDTH, HEIGHT)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            keys = pygame.key.get_pressed()

            self.handle_input(keys)
            self.drone.update_position(DECELERATION, self.key_pressed)


            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.update_view()
            self.draw_ground_plane()
            for obstacle in self.obstacles:
                obstacle.draw(vertices, edges)

            if self.check_collision(5):
                for i in range(3):
                    if self.drone.velocity[i] > 0:
                        self.drone.velocity[i] = -self.drone.velocity[i]
                        self.drone.position[i] += self.drone.velocity[i] - 0.01
                    elif self.drone.velocity[i] < 0:
                        self.drone.velocity[i] = -self.drone.velocity[i]
                        self.drone.position[i] += self.drone.velocity[i] + 0.01

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


    def handle_input(self, keys):
        if any(keys):
            self.key_pressed = True
        else:
            self.key_pressed = False

        if keys[pygame.K_LEFT]:
            self.drone.heading -= ROTATION_SPEED
        if keys[pygame.K_RIGHT]:
            self.drone.heading += ROTATION_SPEED

        rotation_matrix = np.array([[np.cos(np.radians(self.drone.heading)), -np.sin(np.radians(self.drone.heading))],
                                    [np.sin(np.radians(self.drone.heading)), np.cos(np.radians(self.drone.heading))]])

        if keys[pygame.K_a]:
            delta_a = np.matmul(rotation_matrix, [-ACCELERATION, 0])
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY)
        if keys[pygame.K_d]:
            delta_a = np.matmul(rotation_matrix, [ACCELERATION, 0])
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY)
        if keys[pygame.K_UP]:
            delta_a = np.matmul(rotation_matrix, [0, -ACCELERATION]).tolist()
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY)
        if keys[pygame.K_DOWN]:
            delta_a = np.matmul(rotation_matrix, [0, ACCELERATION])
            delta_v = [self.drone.velocity[0] + delta_a[0], self.drone.velocity[1], self.drone.velocity[2] + delta_a[1]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY)
        if keys[pygame.K_w]:
            delta_v = [self.drone.velocity[0], self.drone.velocity[1] + ACCELERATION, self.drone.velocity[2]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY)
        if keys[pygame.K_s]:
            delta_v = [self.drone.velocity[0], self.drone.velocity[1] - ACCELERATION, self.drone.velocity[2]]
            self.drone.update_velocity(delta_v, MAX_VELOCITY)

    def get_state(self):
        # Example state representation
        return np.array(self.drone.position + self.drone.velocity)

    @staticmethod
    def draw_ground_plane():
        glColor4f(0.5, 0.5, 0.5, 0.5)
        glBegin(GL_QUADS)
        glVertex3f(-100, 0, -100)
        glVertex3f(100, 0, -100)
        glVertex3f(100, 0, 100)
        glVertex3f(-100, 0, 100)
        glEnd()