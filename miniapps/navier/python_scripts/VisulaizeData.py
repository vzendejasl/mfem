import numpy as np
import matplotlib.pyplot as plt

# Load data
# file_to_extract_data = '/p/lustre1/zendejas/mfem_parallel/mfem/miniapps/navier/element_centers_scalar.txt'
# file_to_extract_data = '/p/lustre1/zendejas/mfem_parallel/mfem/miniapps/navier/element_centers_0.txt'
# file_to_extract_data = '/p/lustre1/zendejas/mfem_parallel/mfem/miniapps/navier//taylor_green_3d_64_cubed/element_centers_56699.txt'
file_to_extract_data = '/g/g11/zendejas/Documents/mfem_build/mfem/miniapps/navier/element_centers_scalar_0.txt'
data = np.genfromtxt(file_to_extract_data, delimiter=' ', skip_header=6)

# Assign column data
xpos = data[:, 0]
ypos = data[:, 1]
zpos = data[:, 2]
solution = data[:, 3]

# Round the coordinates to avoid floating-point precision issues
# Choose a decimal precision that matches your grid spacing resolution.
xpos_rounded = np.round(xpos, decimals=12)
ypos_rounded = np.round(ypos, decimals=12)
zpos_rounded = np.round(zpos, decimals=12)

# Determine unique coordinates after rounding
x_unique = np.unique(xpos_rounded)
y_unique = np.unique(ypos_rounded)
z_unique = np.unique(zpos_rounded)

nx = len(x_unique)
ny = len(y_unique)
nz = len(z_unique)

print(f"Number of unique x values: {nx}")
print(f"Number of unique y values: {ny}")
print(f"Number of unique z values: {nz}")

expected_num_points = nx * ny * nz
actual_num_points = xpos.size

print(f"Expected number of points: {expected_num_points}")
print(f"Actual number of points: {actual_num_points}")

# Check if the actual number of points matches the expected number
if actual_num_points != expected_num_points:
    print("Warning: The actual number of data points does not match the expected number based on grid sizes.")
    # Proceeding with caution; there might be missing or extra data points

values_grid = np.full((nx, ny, nz), np.nan)  # Using NaN to identify unassigned points

# Step 4: Assign data to the grid array
for i in range(actual_num_points):
    xi = np.where(x_unique == xpos[i])[0][0]
    yi = np.where(y_unique == ypos[i])[0][0]
    zi = np.where(z_unique == zpos[i])[0][0]
    values_grid[xi, yi, zi] = solution[i]
    
    
# Plotting the 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(xpos,ypos,zpos, c=solution, cmap='jet', marker='o')

# Add a colorbar
fig.colorbar(scatter, ax=ax, label='Solution')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')

plt.show()