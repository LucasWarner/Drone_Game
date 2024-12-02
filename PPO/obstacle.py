from OpenGL.GL import *



class Obstacle:

    def __init__(self, position, size):
        self.position = position
        self.size = size

    def draw(self, vertices, edges):

        glPushMatrix()
        glTranslatef(self.position[0], 0, self.position[2])
        glScalef(self.size, self.position[1], self.size)
        glColor4f(1, 0, 0, 1)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
        glPopMatrix()