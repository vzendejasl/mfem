import numpy as np
import finufft
import time
np.random.seed(42)

# number of nonuniform points
M = 400

# desired number of output Fourier modes
N = M

# input nonuniform points
x = 2 * np.pi * np.random.uniform(size=M)

# j = np.arange(M)
# x = (j/M - 0.5) * 2*np.pi          # shape (M,)
# 
# x = np.linspace(0.0, 2*np.pi, M, endpoint=False)   # 0, Δx, … 2π-Δx

# their complex strengths
c = (np.random.standard_normal(size=M)
     + 1J * np.random.standard_normal(size=M))

# f1 = 5;
# f2 = 20;

# c = (np.cos(f1*x)+0.5*np.cos(f2*x)).astype(np.complex128)

energy_spatial = np.sum(np.abs(c)**2)

# calculate the transform
t0 = time.time()
f = finufft.nufft1d1(x, c, N)
print("finufft1d1 done in {0:.2g} s.".format(time.time()-t0))

n = 14   # do a math check, for a single output mode index n
assert((n>=-N/2.) & (n<N/2.))
ftest = sum(c * np.exp(1.j*n*x))
err = np.abs(f[n + N // 2] - ftest) / np.max(np.abs(f))
print("Error relative to max: {0:.2e}".format(err))

# energy_freq = np.sum(np.abs(f[N//2+1:])**2)/ N
# energy_freq = np.sum(np.abs(f)**2)/ N/ 2
energy_freq = np.sum(np.abs(f)**2)/ N 

error_relative = 100*np.abs(energy_freq - energy_spatial)/energy_spatial

print(f"Spatial domain energy: {energy_spatial:.4e}")
print(f"Frequency domain energy: {energy_freq:.4e}")
print(f"Energy ratio (freq/spatial): {energy_freq/energy_spatial:.4e}")
print(f"Error abs(freq-spatial): {(np.abs(energy_freq - energy_spatial)):.4e}")
print(f"Error relative (%): {error_relative:.4f}")
