import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import random
import numpy as np
from drone import Drone
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
goal_vertices= [
    [55, 5, 45],
    [55, 0.1, 45],
    [45, 0.1, 45],
    [45, 5, 45],
    [55, 5, 55],
    [55, 0.1, 55],
    [45, 0.1, 55],
    [45, 5, 55]
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
        self.clock = pygame.time.Clock()
        self.key_pressed = False
        self.env = DroneEnv()
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=0.0022,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.96,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=2,
            tensorboard_log="./ppo_tensorboard/"
        )


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



    def update(self):
        print("updating...")
        # 1. Get current observation from environment
        obs = self.env.get_observation()

        # 2. Get AI action from trained model
        action, _states = self.model.predict(obs)

        # 3. Apply action and get new state
        self.env.step(action)

        # 4. Update elements
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.update_view()

        self.env.draw_ground_plane()
        self.env.draw_goal_box(goal_vertices, edges)
        for obstacle in self.env.obstacles:
            obstacle.draw(vertices, edges)

        # Update the display
        pygame.display.flip()

        self.clock.tick(60)

    def update_view(self):
        rotation_matrix = np.array([[np.cos(np.radians(self.env.drone.heading)), -np.sin(np.radians(self.env.drone.heading))],
                                    [np.sin(np.radians(self.env.drone.heading)), np.cos(np.radians(self.env.drone.heading))]])
        # Rotate the drone's shape based on the heading around its relative origin
        glPushMatrix()
        glTranslatef(*self.env.drone.position)
        glRotatef(-self.env.drone.heading, 0, 1, 0)  # Rotate around the y-axis based on heading
        self.env.drone.draw(tri_vertices, tri_edges)  # Draw the rotated drone shape
        glPopMatrix()

        camera_matrix = [0, 6]

        # Define the camera position relative to the drone
        rotated_camera = np.matmul(rotation_matrix, camera_matrix)

        camera_position = (rotated_camera[0] + self.env.drone.position[0],
                           self.env.drone.position[1],
                           rotated_camera[1] + self.env.drone.position[2])

        # Rotate the camera position with the drone
        glLoadIdentity()
        gluPerspective(60, (WIDTH / HEIGHT), 0.1, 100)
        gluLookAt(camera_position[0], camera_position[1] + .5, camera_position[2],  # Camera position
                  self.env.drone.position[0], self.env.drone.position[1], self.env.drone.position[2],  # Look at the drone
                  0, 1, 0)  # Up vector
    """
    def reset_visualization(self):
    # Reset drone position and heading
    self.env.drone.position = [0, 0.2, 0]
    self.env.drone.heading = 0
    """

    """
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
    """

    def get_state(self):
        # Example state representation
        return np.array(self.drone.position + self.drone.velocity)

