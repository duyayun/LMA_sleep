import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_sphere(ax, x, y, z, r, color):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X = r * np.outer(np.cos(u), np.sin(v)) + x
    Y = r * np.outer(np.sin(u), np.sin(v)) + y
    Z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z

    ax.plot_surface(X, Y, Z, color=color)

def plot_cone(ax, x, y, z, r, h, color):
    v = np.linspace(0, np.pi/2, 100)
    u = np.linspace(0, 2 * np.pi, 100)
    X = r * np.outer(np.cos(u), np.sin(v)) + x
    Y = r * np.outer(np.sin(u), np.sin(v)) + y
    Z = h * np.outer(np.ones(np.size(u)), np.cos(v)) + z

    ax.plot_surface(X, Y, Z, color=color)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the snowman's body
plot_sphere(ax, 0, 0, 3, 3, 'white')  # Bottom
plot_sphere(ax, 0, 0, 7, 2, 'white')  # Middle
plot_sphere(ax, 0, 0, 10, 1, 'white')  # Top

# Draw the snowman's eyes
plot_sphere(ax, -0.3, 0.5, 10.3, 0.1, 'black')  # Left eye
plot_sphere(ax, 0.3, 0.5, 10.3, 0.1, 'black')  # Right eye

# Draw the snowman's carrot nose
plot_cone(ax, 0, 0.8, 10, 0.1, 0.5, 'orange')

# Draw the snowman's stick arms
ax.plot([-1.5, -3], [0, 0], [7, 8], color='brown')  # Left arm
ax.plot([1.5, 3], [0, 0], [7, 8], color='brown')  # Right arm

# Set the aspect ratio and limits of the plot
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 15)

# Remove the axis lines and ticks
ax.set_axis_off()

# Show the plot
plt.show()
