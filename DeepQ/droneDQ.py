import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

class Drone:
    def __init__(self, position, velocity, heading):
        self.position = position
        self.last_position = position
        self.velocity = velocity
        self.heading = heading

    def update_velocity(self, delta_v, max_velocity, hit_obstacle = False):
        if not hit_obstacle:
            delta_v_mag = np.sqrt((delta_v[0])**2 + (delta_v[1])**2 + (delta_v[2])**2)
            delta_v_unit = [delta_v[0] / delta_v_mag, delta_v[1] / delta_v_mag, delta_v[2] / delta_v_mag]
            if delta_v_mag > max_velocity:
                delta_v_max = [i * max_velocity for i in delta_v_unit]
                self.velocity[0] = delta_v_max[0]
                self.velocity[1] = delta_v_max[1]
                self.velocity[2] = delta_v_max[2]
            else:
                self.velocity[0] = delta_v[0]
                self.velocity[1] = delta_v[1]
                self.velocity[2] = delta_v[2]

    def update_position(self, deceleration, key_pressed, hit_obstacle):
        if hit_obstacle:
            self.position[0] = self.last_position[0]
            self.position[1] = self.last_position[1]
            self.position[2] = self.last_position[2]
            self.position[0] += -self.velocity[0]
            self.position[1] += -self.velocity[1]
            self.position[2] += -self.velocity[2]

        else:
            self.last_position[0] += self.velocity[0]
            self.last_position[1] += self.velocity[1]
            self.last_position[2] += self.velocity[2]
            delta_v_mag = np.sqrt((self.velocity[0]) ** 2 + (self.velocity[1]) ** 2 + (self.velocity[2]) ** 2)
            if delta_v_mag == 0:
                delta_a = [0,0,0]
            else:
                delta_v_unit = [self.velocity[0] / delta_v_mag, self.velocity[1] / delta_v_mag, self.velocity[2] / delta_v_mag]
                delta_a = [delta_v_unit[0] * deceleration, delta_v_unit[1] * deceleration, delta_v_unit[2] * deceleration]
            if not key_pressed:
                for i in range(3):
                    if self.velocity[i] > 0:
                        self.velocity[i] = max(0, self.velocity[i] - delta_a[i])
                    elif self.velocity[i] < 0:
                        self.velocity[i] = min(0, self.velocity[i] - delta_a[i])

            self.position[0] += self.velocity[0]
            self.position[1] += self.velocity[1]
            self.position[2] += self.velocity[2]

    @staticmethod
    def draw(tri_vertices, tri_edges):
        glPushMatrix()
        glColor4f(0, 1, 0, 1)
        glBegin(GL_LINES)
        for edge in tri_edges:
            for vertex in edge:
                glVertex3fv(tri_vertices[vertex])
        glEnd()
        glPopMatrix()