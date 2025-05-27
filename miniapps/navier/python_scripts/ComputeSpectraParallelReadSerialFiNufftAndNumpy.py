#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re, finufft
import argparse
import pandas as pd
import sys

# -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Compute TKE and velocity magnitude from velocity data.')
parser.add_argument('data_file', type=str, help='Path to the data file')
args = parser.parse_args()
data_filename = args.data_file

eps = 1e-12           # requested accuracy for FINUFFT

plt.figure(figsize=(8, 6))

# ---- HEADER ----
print(f"[Rank 0] Reading header and data from file:\n  {data_filename}")
with open(data_filename, 'r') as f:
    header_lines = [next(f) for _ in range(6)]
step_number_extracted = None
time_extracted = None
for line in header_lines:
    if 'Step' in line:
        m = re.search(r'Step\s*=\s*(\d+)', line)
        if m:
            step_number_extracted = m.group(1)
    if 'Time' in line:
        m = re.search(r'Time\s*=\s*([0-9.eE+-]+)', line)
        if m:
            time_extracted = float(m.group(1))
if step_number_extracted is None:
    step_number_extracted = "Unknown"
if time_extracted is None:
    time_extracted = 0.0
print(f"[Rank 0] Header: Step = {step_number_extracted}, Time = {time_extracted:.3e}")

# ---- DATA LOADING (your chunked method, unchanged) ----
print("[Rank 0] Loading data in chunks (skipping header)...")
chunk_size = 5_000_000
reader = pd.read_csv(
    data_filename,
    sep=r'\s+',
    engine='python',
    skiprows=6,
    header=None,
    chunksize=chunk_size
)

xpos_list, ypos_list, zpos_list = [], [], []
velx_list, vely_list, velz_list = [], [], []

for chunk in reader:
    xp = np.round(chunk.iloc[:, 0].values, 10)
    yp = np.round(chunk.iloc[:, 1].values, 10)
    zp = np.round(chunk.iloc[:, 2].values, 10)
    vx = chunk.iloc[:, 3].values
    vy = chunk.iloc[:, 4].values
    vz = chunk.iloc[:, 5].values

    xpos_list.append(xp)
    ypos_list.append(yp)
    zpos_list.append(zp)
    velx_list.append(vx)
    vely_list.append(vy)
    velz_list.append(vz)

total_pts = sum(arr.size for arr in xpos_list)
xpos = np.empty(total_pts, dtype=xpos_list[0].dtype)
ypos = np.empty(total_pts, dtype=ypos_list[0].dtype)
zpos = np.empty(total_pts, dtype=zpos_list[0].dtype)
velx = np.empty(total_pts, dtype=velx_list[0].dtype)
vely = np.empty(total_pts, dtype=vely_list[0].dtype)
velz = np.empty(total_pts, dtype=velz_list[0].dtype)

offset = 0
for xp, yp, zp, vx, vy, vz in zip(
        xpos_list, ypos_list, zpos_list,
        velx_list, vely_list, velz_list):
    n = xp.size
    xpos[offset:offset+n] = xp
    ypos[offset:offset+n] = yp
    zpos[offset:offset+n] = zp
    velx[offset:offset+n] = vx
    vely[offset:offset+n] = vy
    velz[offset:offset+n] = vz
    offset += n

del xpos_list, ypos_list, zpos_list
del velx_list, vely_list, velz_list

# ---- ROUNDING (for floating point safety) ----
xpos_rounded = np.round(xpos, decimals=10)
ypos_rounded = np.round(ypos, decimals=10)
zpos_rounded = np.round(zpos, decimals=10)

# ---- STRUCTURED GRID SETUP ----
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
print(f"Actual number of points:   {actual_num_points}")

# ---- DEDUPLICATION: ensure one value per (x, y, z) ----
positions = np.stack([xpos_rounded, ypos_rounded, zpos_rounded], axis=1)
unique_pos, unique_indices = np.unique(positions, axis=0, return_index=True)
xpos_rounded = xpos_rounded[unique_indices]
ypos_rounded = ypos_rounded[unique_indices]
zpos_rounded = zpos_rounded[unique_indices]
velx = velx[unique_indices]
vely = vely[unique_indices]
velz = velz[unique_indices]

if xpos_rounded.size != nx * ny * nz:
    print("ERROR: After deduplication, number of unique points does not match grid shape!")
    print(f"Points after deduplication: {xpos_rounded.size}")
    print(f"Expected grid shape: {nx} × {ny} × {nz} = {nx*ny*nz}")
    sys.exit(1)

# ---- SORT LEXICOGRAPHICALLY SO FLAT ARRAYS MATCH 3D GRID ORDER ----
x_idx = np.searchsorted(x_unique, xpos_rounded)
y_idx = np.searchsorted(y_unique, ypos_rounded)
z_idx = np.searchsorted(z_unique, zpos_rounded)
sort_indices = np.lexsort((z_idx, y_idx, x_idx))
xpos_rounded = xpos_rounded[sort_indices]
ypos_rounded = ypos_rounded[sort_indices]
zpos_rounded = zpos_rounded[sort_indices]
velx = velx[sort_indices]
vely = vely[sort_indices]
velz = velz[sort_indices]

# ---- FFT: RESHAPE TO 3D GRIDS ----
velx_grid = velx.reshape((nx, ny, nz))
vely_grid = vely.reshape((nx, ny, nz))
velz_grid = velz.reshape((nx, ny, nz))

# ---- PHYSICAL TKE ----
tke_physical = 0.5 * np.sum(velx_grid**2 + vely_grid**2 + velz_grid**2)
print(f"[Rank 0] Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")

# ---- NUMPY FFT ----
Ntot = nx * ny * nz
Fvx_ft = np.fft.fftn(velx_grid) / Ntot
Fvy_ft = np.fft.fftn(vely_grid) / Ntot
Fvz_ft = np.fft.fftn(velz_grid) / Ntot

# ---- FINUFFT ----
dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
Lx = x_unique[-1] - x_unique[0] + dx
Ly = y_unique[-1] - y_unique[0] + dy
Lz = z_unique[-1] - z_unique[0] + dz

def map_to_pi(arr, amin, L):
    return (arr - amin) * (2 * np.pi / L) - np.pi

x_s = map_to_pi(xpos_rounded, x_unique[0], Lx)
y_s = map_to_pi(ypos_rounded, y_unique[0], Ly)
z_s = map_to_pi(zpos_rounded, z_unique[0], Lz)

def finufft_forward_scattered(x_, y_, z_, v_):
    F_flat = finufft.nufft3d1(
        x_, y_, z_,
        v_.astype(np.complex128),
        (nx, ny, nz), eps=eps, isign=1, modeord=1
    )
    return F_flat.reshape((nx, ny, nz)) / Ntot

Fvx_nu = finufft_forward_scattered(x_s, y_s, z_s, velx)
Fvy_nu = finufft_forward_scattered(x_s, y_s, z_s, vely)
Fvz_nu = finufft_forward_scattered(x_s, y_s, z_s, velz)

# ---- ENERGY SPECTRA ----
E3d_nu = 0.5 * (np.abs(Fvx_nu)**2 + np.abs(Fvy_nu)**2 + np.abs(Fvz_nu)**2)
E3d_ft = 0.5 * (np.abs(Fvx_ft)**2 + np.abs(Fvy_ft)**2 + np.abs(Fvz_ft)**2)
print(f"FINUFFT KE: {E3d_nu.sum()*Ntot:.6e}  |  FFT KE: {E3d_ft.sum()*Ntot:.6e}")

# ---- fftshift for standard spectral binning! ----
Fvx_ft = np.fft.fftshift(Fvx_ft)
Fvy_ft = np.fft.fftshift(Fvy_ft)
Fvz_ft = np.fft.fftshift(Fvz_ft)

# ---- FINUFFT ----
# (leave this as before, since FINUFFT does not need shifting if you output modes in the same order as FFT)
# But for apples-to-apples, if you want, you can also apply fftshift to the FINUFFT arrays:
Fvx_nu = np.fft.fftshift(Fvx_nu)
Fvy_nu = np.fft.fftshift(Fvy_nu)
Fvz_nu = np.fft.fftshift(Fvz_nu)

# ---- ENERGY SPECTRA ----
E3d_nu = 0.5 * (np.abs(Fvx_nu)**2 + np.abs(Fvy_nu)**2 + np.abs(Fvz_nu)**2)
E3d_ft = 0.5 * (np.abs(Fvx_ft)**2 + np.abs(Fvy_ft)**2 + np.abs(Fvz_ft)**2)
print(f"FINUFFT KE: {E3d_nu.sum()*Ntot:.6e}  |  FFT KE: {E3d_ft.sum()*Ntot:.6e}")

# ---- SPECTRAL BINS AND PLOT ----
kx = np.fft.fftfreq(nx, d=dx/(2*np.pi))
ky = np.fft.fftfreq(ny, d=dy/(2*np.pi))
kz = np.fft.fftfreq(nz, d=dz/(2*np.pi))

kx = np.fft.fftshift(kx)
ky = np.fft.fftshift(ky)
kz = np.fft.fftshift(kz)

KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

# Flatten arrays for binning
k_flat = k_magnitude.flatten()

num_bins = nx
k_bin_edges = np.arange(0, num_bins+1) - 0.5
k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
E_k_nu, _ = np.histogram(k_flat, bins=k_bin_edges, weights=E3d_nu.ravel())
E_k_ft, _ = np.histogram(k_flat, bins=k_bin_edges, weights=E3d_ft.ravel())

plt.loglog(k_bin_centers, E_k_nu, 'b-',  label=f'FINUFFT step {step_number_extracted}')
plt.loglog(k_bin_centers, E_k_ft, 'k--', label='FFT')
plt.loglog(k_bin_centers, 0.1*(k_bin_centers/k_bin_centers[1])**(-5/3), 'r:', label=r'$k^{-5/3}$')
plt.xlabel(r'$|k|$'); plt.ylabel(r'$E(k)$'); plt.title('3-D energy spectrum')
plt.legend(); plt.grid(True, ls=':'); plt.tight_layout(); plt.show()

