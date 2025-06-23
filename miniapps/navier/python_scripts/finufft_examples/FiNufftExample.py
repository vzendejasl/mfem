import numpy as np
import finufft
import matplotlib.pyplot as plt
import time

# Parameters
L = 10  # Periodic interval [0, L)
kmax = 500  # Bandlimit
k = np.arange(-kmax, kmax)  # Frequency indices from -kmax to kmax-1
N = 2 * kmax  # Number of modes

# Make some convenient Fourier coefficients
np.random.seed(0)
fk = np.random.randn(N) + 1j * np.random.randn(N)  # Random complex data
k0 = 100  # Frequency scale
fk = fk * np.exp(-(k / k0)**2)  # Scale amplitudes to kill high frequencies

# Random target points
M = 10000  # Number of points
x = L * np.random.rand(M)  # Points in [0, L)
x_scaled = x * (2 * np.pi / L)  # Scale to 2π-periodic for FINUFFT

# Evaluate using FINUFFT
tol = 1e-12  # Tolerance
start = time.time()
c = finufft.nufft1d2(x_scaled, fk, isign=1, eps=tol)
end = time.time()
print(f"FINUFFT elapsed time: {end - start:.6f} seconds")

# Naive evaluation for comparison
start = time.time()
cn = np.zeros(M, dtype=complex)
for m in k:
    cn += fk[m + N // 2] * np.exp(1j * m * x_scaled)
end = time.time()
print(f"Naive elapsed time: {end - start:.6f} seconds")
print(f"Max absolute error: {np.max(np.abs(c - cn))}")

# Plot
Mp = 10000  # Points to plot
jplot = np.arange(Mp)
plt.figure(figsize=(10, 2.5))
plt.plot(x[jplot], c[jplot].real, 'b.', label='NU pts')
plt.xlabel('x')
plt.ylabel('Re f(x)')
plt.title('1D Fourier Series Evaluation')
plt.axis('tight')

# Extra stuff: Evaluation on uniform points via FFT
fk_pad = np.pad(fk, ((Mp - N) // 2, (Mp - N) // 2), mode='constant')  # Pad with zeros
fi = Mp * np.fft.ifft(np.fft.ifftshift(fk_pad))  # Evaluate via FFT
yi = L * np.arange(Mp) / Mp  # Spatial grid for FFT

# Plot uniform points
plt.plot(yi, fi.real, 'r.', label='unif pts')
plt.legend()

# Math check: Send uniform points into NUFFT and compare
ci = finufft.nufft1d2(yi * (2 * np.pi / L), fk, isign=1, eps=tol)
error_uniform = np.max(np.abs(fi - ci))
print(f"Max error on {Mp} uniform test pts: {error_uniform:.3e}")

plt.show()
