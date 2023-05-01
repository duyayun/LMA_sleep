import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def draw_sphere(radius, slices, stacks):
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)

def draw_cylinder(radius, height, slices):
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluCylinder(quad, radius, radius, height, slices, 1)
    gluDeleteQuadric(quad)

def draw_snowman():
    # Snowman base
    glColor3f(1, 1, 1)  # White color
    glPushMatrix()
    glTranslatef(0, 0.5, 0)
    draw_sphere(0.5, 50, 50)
    glPopMatrix()

    # Snowman middle
    glColor3f(1, 1, 1)  # White color
    glPushMatrix()
    glTranslatef(0, 1.1, 0)
    draw_sphere(0.4, 50, 50)
    glPopMatrix()

    # Snowman head
    glColor3f(1, 1, 1)  # White color
    glPushMatrix()
    glTranslatef(0, 1.7, 0)
    draw_sphere(0.3, 50, 50)
    glPopMatrix()

    # Carrot nose
    glColor3f(1, 0.5, 0)  # Orange color
    glPushMatrix()
    glTranslatef(0, 1.7, 0.3)
    
    glRotatef(90, 1, 0, 0)
    draw_cylinder(0.025, 0.25, 20)
    glPopMatrix()

    # Stick arms
    glColor3f(0.3, 0.15, 0.07)  # Brown color
    for side in [-1, 1]:
        glPushMatrix()
        glTranslatef(side * 0.35, 1.2, 0)
        glRotatef(140 * side, 0, 0, 1)
        glScalef(1, 0.1, 0.1)
        draw_cylinder(0.05, 1, 8)
        glPopMatrix()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_snowman()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
