import numpy as np
import finufft
np.random.seed(42)

M = 40
N = M*2

x = np.random.uniform(-np.pi, np.pi, M)
y = np.random.uniform(-np.pi, np.pi, M)
z = np.random.uniform(-np.pi, np.pi, M)
# f = np.exp(-0.1*(x**2 + y**2 + z**2)).astype(np.complex128)  # <-- fixed
# f = (np.cos(x*2) + np.cos(y*2) + np.cos(z*34)*2).astype(np.complex128)  # <-- fixed
# their complex strengths
f = (np.random.standard_normal(size=M)
     + 1J * np.random.standard_normal(size=M))

energy_spatial = np.sum(np.abs(f)**2)

Fhat = finufft.nufft3d1(x, y, z, f, (N, N, N))

energy_freq = np.sum(np.abs(Fhat)**2) / N**3

error_relative = 100*np.abs(energy_freq - energy_spatial)/energy_spatial

print(f"Spatial domain energy: {energy_spatial:.4e}")
print(f"Frequency domain energy: {energy_freq:.4e}")
print(f"Energy ratio (freq/spatial): {energy_freq/energy_spatial:.4e}")
print(f"Error abs(freq-spatial): {(np.abs(energy_freq - energy_spatial)):.4e}")
print(f"Error relative(%): {error_relative:.4f}")
