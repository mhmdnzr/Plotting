import numpy as np
import matplotlib.pyplot as plt

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
x_interpolated = lagrange_interpolation(t_values, x_values, interpolation_points)
y_interpolated = lagrange_interpolation(t_values, y_values, interpolation_points)

# Plot the original sinusoidal motion
plt.plot(t_values, x_values, label='Original x(t)')
plt.plot(t_values, y_values, label='Original y(t)')

# Plot the Lagrange interpolation at selected points
plt.scatter(interpolation_points, x_interpolated, color='red', label='Interpolated x(t)')
plt.scatter(interpolation_points, y_interpolated, color='blue', label='Interpolated y(t)')

plt.legend()
plt.xlabel('t')
plt.ylabel('Values')
plt.title('Lagrange Interpolation of Sinusoidal Motion')
plt.show()
