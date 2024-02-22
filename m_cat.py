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

# Define three categories for mx
mx_lambda = lambda x: np.exp(-x) / (1 + x**2)
mx_sin = np.sin(t_values)
mx_constant = np.ones_like(t_values)

# Perform Lagrange interpolation for each category
x_interpolated_lambda = lagrange_interpolation(t_values, x_values * mx_lambda(t_values), interpolation_points)
y_interpolated_lambda = lagrange_interpolation(t_values, y_values * mx_lambda(t_values), interpolation_points)

x_interpolated_sin = lagrange_interpolation(t_values, x_values * mx_sin, interpolation_points)
y_interpolated_sin = lagrange_interpolation(t_values, y_values * mx_sin, interpolation_points)

x_interpolated_constant = lagrange_interpolation(t_values, x_values * mx_constant, interpolation_points)
y_interpolated_constant = lagrange_interpolation(t_values, y_values * mx_constant, interpolation_points)

# Plot the original sinusoidal motion
plt.plot(t_values, x_values, label='Original x(t)')
plt.plot(t_values, y_values, label='Original y(t)')

# Plot the Lagrange interpolation for mx=lambda
plt.scatter(interpolation_points, x_interpolated_lambda, color='red', label='Interpolated x(t) - mx=lambda')
plt.scatter(interpolation_points, y_interpolated_lambda, color='blue', label='Interpolated y(t) - mx=lambda')

# Plot the Lagrange interpolation for mx=sin(x)
plt.scatter(interpolation_points, x_interpolated_sin, color='green', label='Interpolated x(t) - mx=sin(x)')
plt.scatter(interpolation_points, y_interpolated_sin, color='purple', label='Interpolated y(t) - mx=sin(x)')

# Plot the Lagrange interpolation for mx=constant
plt.scatter(interpolation_points, x_interpolated_constant, color='orange', label='Interpolated x(t) - mx=constant')
plt.scatter(interpolation_points, y_interpolated_constant, color='brown', label='Interpolated y(t) - mx=constant')

plt.legend()
plt.xlabel('t')
plt.ylabel('Values')
plt.title('Lagrange Interpolation with Different mx Categories')
plt.show()
