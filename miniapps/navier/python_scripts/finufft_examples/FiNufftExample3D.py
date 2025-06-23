import numpy as np
import finufft
import matplotlib.pyplot as plt
import time

# Parameters
kmax = 100  # bandlimit per dimension (reduce for memory)
k = np.arange(-kmax, kmax)  # frequency indices in each dimension
N1 = N2 = N3 = 2 * kmax

# Create 3D frequency grid
k1, k2, k3 = np.meshgrid(k, k, k, indexing='ij')

# Random complex Fourier coefficients
np.random.seed(0)
fk = np.random.randn(N1, N2, N3) + 1j * np.random.randn(N1, N2, N3)

# Matérn kernel spectral density
k0 = 30
alpha = 3.7
fk *= ((k1**2 + k2**2 + k3**2) / k0**2 + 1) ** (-alpha / 2)

# Random target points in the cube [0, 2pi)^3
M = int(2e5)  # Reduce M for memory
x = 2 * np.pi * np.random.rand(M)
y = 2 * np.pi * np.random.rand(M)
z = 2 * np.pi * np.random.rand(M)

# Type 2 NUFFT: evaluate Fourier series at (x, y, z)
tol = 1e-9
start = time.time()
c = finufft.nufft3d2(x, y, z, fk, isign=1, eps=tol)
end = time.time()
print(f"FINUFFT elapsed time: {end - start:.6f} seconds")

# Math check at the first target point
j = 0  # Python is 0-based
c1 = np.sum(fk * np.exp(1j * (k1 * x[j] + k2 * y[j] + k3 * z[j])))
rel_error = np.abs(c1 - c[j]) / np.max(np.abs(c))
print(f"Relative error at first target: {rel_error:.3e}")

# Plot a 2D slice of the points (fix z near pi)
slice_mask = np.abs(z - np.pi) < 0.1
jplot = np.where(slice_mask)[0][:int(1e4)]  # plot up to 10k points

plt.figure(figsize=(7, 6))
plt.scatter(x[jplot], y[jplot], c=np.real(c[jplot]), s=1, cmap='viridis')
plt.axis('tight')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Re f(x, y, z≈π)')
plt.title('Re f(x, y, z≈π)  [3D Fourier series slice]')
plt.show()
