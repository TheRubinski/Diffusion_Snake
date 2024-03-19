import numpy as np
import matplotlib.pyplot as plt

# Generate example data for a closed curve (replace with your own data)
num_points = 100
theta = np.linspace(0, 2*np.pi, num_points)
x = np.cos(theta)+np.random.random(num_points)*0.1
y = np.sin(theta)+np.random.random(num_points)*0.1

# Compute distances between neighboring points
dx = x -np.roll(x,1,axis=0) 
dy = y -np.roll(y,1,axis=0)
distances = np.sqrt(dx**2 + dy**2)
distances=(distances+np.roll(distances,-1,axis=0))
# Calculate local density based on distances
density =  distances  # Inverse of distances as a simple measure of density

# Normalize density values to range [0, 1]
density_norm = (density - np.min(density)) / (np.max(density) - np.min(density))

# Define point size based on density
point_size = 100 * density_norm  # Adjust scaling factor as needed
print(density_norm)
# Plot the closed curve with points colored by density
plt.plot(x, y, label='Closed Curve')
plt.scatter(x, y, c=density_norm, cmap='viridis', label='Points')
plt.colorbar(label='Density')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Density of Points along Closed Curve')
plt.legend()
plt.show()
