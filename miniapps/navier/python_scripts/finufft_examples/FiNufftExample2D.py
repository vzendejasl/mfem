import numpy as np
import finufft
import matplotlib.pyplot as plt
import time

# Parameters
kmax = 500  # bandlimit per dimension
k = np.arange(-kmax, kmax)  # frequency indices in each dimension
N1 = 2 * kmax
N2 = N1

# Create 2D frequency grid
k1, k2 = np.meshgrid(k, k, indexing='ij')

# Random complex Fourier coefficients
np.random.seed(0)
fk = np.random.randn(N1, N2) + 1j * np.random.randn(N1, N2)

# Matérn kernel spectral density
k0 = 30
alpha = 3.7
fk *= ((k1**2 + k2**2) / k0**2 + 1) ** (-alpha / 2)

# Random target points in the square [0, 2pi)^2
M = int(1e6)
x = 2 * np.pi * np.random.rand(M)
y = 2 * np.pi * np.random.rand(M)

# Type 2 NUFFT: evaluate Fourier series at (x, y)
tol = 1e-9
start = time.time()
c = finufft.nufft2d2(x, y, fk, isign=1, eps=tol)
end = time.time()
print(f"FINUFFT elapsed time: {end - start:.6f} seconds")

# Math check at the first target point
j = 0  # Python is 0-based
c1 = np.sum(fk * np.exp(1j * (k1 * x[j] + k2 * y[j])))
rel_error = np.abs(c1 - c[j]) / np.max(np.abs(c))
print(f"Relative error at first target: {rel_error:.3e}")

# Plot a sample of the points
jplot = np.arange(int(1e5))
plt.figure(figsize=(7, 6))
plt.scatter(x[jplot], y[jplot], c=np.real(c[jplot]), s=1, cmap='viridis')
plt.axis('tight')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Re f(x, y)')
plt.title('Re f(x, y)')
plt.show()
