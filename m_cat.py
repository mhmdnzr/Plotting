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

# Define three different mx functions
def mx_lambda(x):
    return 1 / (1 + x**2)

def mx_sin(x):
    return np.sin(x)

def mx_constant(x):
    return np.full_like(x, 0.5)  # You can replace 0.5 with any constant

# Perform Lagrange interpolation for each mx function
x_interpolated_lambda = lagrange_interpolation(t_values, mx_lambda(t_values), interpolation_points)
x_interpolated_sin = lagrange_interpolation(t_values, mx_sin(t_values), interpolation_points)
x_interpolated_constant = lagrange_interpolation(t_values, mx_constant(t_values), interpolation_points)

# Plot the original sinusoidal motion
plt.figure(figsize=(15, 5))

# Plot for mx = lambda / (1 + x^2)
plt.subplot(1, 3, 1)
plt.plot(t_values, x_values, label='Original x(t)')
plt.scatter(interpolation_points, x_interpolated_lambda, color='red', label='Interpolated x(t)')
plt.title('mx = λ / (1 + x^2)')
plt.xlabel('t')
plt.ylabel('Values')
plt.legend()

# Plot for mx = sin(x)
plt.subplot(1, 3, 2)
plt.plot(t_values, x_values, label='Original x(t)')
plt.scatter(interpolation_points, x_interpolated_sin, color='blue', label='Interpolated x(t)')
plt.title('mx = sin(x)')
plt.xlabel('t')
plt.ylabel('Values')
plt.legend()

# Plot for mx = constant
plt.subplot(1, 3, 3)
plt.plot(t_values, x_values, label='Original x(t)')
plt.scatter(interpolation_points, x_interpolated_constant, color='green', label='Interpolated x(t)')
plt.title('mx = constant')
plt.xlabel('t')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()
