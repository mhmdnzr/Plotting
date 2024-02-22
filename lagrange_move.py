import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def lagrange_interpolation(x, y, t):
    result = 0
    n = len(x)

    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term = term * (t - x[j]) / (x[i] - x[j])
        result += term

    return result

# Generate sinusoidal data
t_values = np.linspace(0, 10, 100)
x_values = np.sin(t_values)
y_values = np.cos(t_values)

# Choose some data points for interpolation
interpolation_points = np.array([2, 4, 6, 8])

# Create a figure and axis for plotting
fig, ax = plt.subplots()
line_x, = ax.plot(t_values, x_values, label='Original x(t)')
line_y, = ax.plot(t_values, y_values, label='Original y(t)')

# Initialize interpolated points
points_x = lagrange_interpolation(t_values, x_values, interpolation_points)
points_y = lagrange_interpolation(t_values, y_values, interpolation_points)
sc_x = ax.scatter(interpolation_points, points_x, color='red', label='Interpolated x(t)')
sc_y = ax.scatter(interpolation_points, points_y, color='blue', label='Interpolated y(t)')

ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('Values')
ax.set_title('Lagrange Interpolation of Sinusoidal Motion')

# Update function for animation
def update(frame):
    t = frame / 10.0
    points_x = lagrange_interpolation(t_values, x_values, interpolation_points * t)
    points_y = lagrange_interpolation(t_values, y_values, interpolation_points * t)

    sc_x.set_offsets(np.column_stack((interpolation_points, points_x)))
    sc_y.set_offsets(np.column_stack((interpolation_points, points_y)))

# Create the animation
animation = FuncAnimation(fig, update, frames=range(100), interval=100)
plt.show()
