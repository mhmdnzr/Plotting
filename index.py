import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x_data, y_data, x):
    result = 0
    for i in range(len(y_data)):
        term = y_data[i]
        for j in range(len(x_data)):
            if j != i:
                term = term * (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

# Given data points
x_data = np.array([1, 2, 3, 4])
y_data = np.array([2, 3, 5, 7])

# Generate x values for the plot
x_values = np.linspace(min(x_data), max(x_data), 100)

# Calculate corresponding y values using Lagrange interpolation
y_values = [lagrange_interpolation(x_data, y_data, x) for x in x_values]

# Plot the original data points and the Lagrange interpolation polynomial
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_values, y_values, label='Lagrange Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lagrange Interpolation Example')
plt.legend()
plt.grid(True)
plt.show()
