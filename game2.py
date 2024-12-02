import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import random
import numpy as np
from drone import Drone
from obstacle import Obstacle
from spatial_grid import SpatialGrid
from drone_env import DroneEnv
from OpenGL.GL import *
from OpenGL.GLU import *
from stable_baselines3 import PPO

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
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

# Training (if you want to train a new model)
model.learn(total_timesteps=100000)
model.save("ppo_drone")
class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.drone = Drone([0, .2, 0], [0.0, 0.0, 0.0], 0)
        self.obstacles = self.generate_obstacles(45)
        self.spatial_grid = SpatialGrid(cell_size=10)
        for obstacle in self.obstacles:
            self.spatial_grid.add_obstacle(obstacle)
        self.key_pressed = False
        self.env = DroneEnv()
        self.model = PPO("MlpPolicy", self.env, verbose=1)

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
            # Your existing game loop code (event handling, drawing, etc.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            self.update()

            self.clock.tick(60)

    def update(self):
        # 1. Get current observation from environment
        obs = self.env.get_observation()

        # 2. Get AI action from trained model
        action, _states = self.model.predict(obs)

        # 3. Apply action and get new state
        obs, reward, done, truncated, info = self.env.step(action)

        # 4. Handle manual controls (for testing/debugging)
        keys = pygame.key.get_pressed()
        manual_action = None

        if keys[pygame.K_w]:
            manual_action = 1  # move up
        elif keys[pygame.K_s]:
            manual_action = 2  # move down
        elif keys[pygame.K_a]:
            manual_action = 3  # move left
        elif keys[pygame.K_d]:
            manual_action = 4  # move right
        elif keys[pygame.K_UP]:
            manual_action = 5  # move forward
        elif keys[pygame.K_DOWN]:
            manual_action = 6  # move backward
        elif keys[pygame.K_LEFT]:
            manual_action = 7  # rotate left
        elif keys[pygame.K_RIGHT]:
            manual_action = 8  # rotate right

        # If manual control is being used, override AI action
        if manual_action is not None:
            obs, reward, done, truncated, info = self.env.step(manual_action)

        # 5. Update drone visualization
        self.drone.x = self.env.position[0]
        self.drone.y = self.env.position[1]
        self.drone.z = self.env.position[2]
        self.drone.heading = self.env.heading

        # 6. Handle episode reset if needed
        if done or truncated:
            obs, info = self.env.reset()
            # Reset visualization if needed
            self.reset_visualization()

        # 7. Update elements
        self.update_view()

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

    def reset_visualization(self):
        # Reset drone position and heading
        self.drone.x = 0
        self.drone.y = 0.2
        self.drone.z = 0
        self.drone.heading = 0

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