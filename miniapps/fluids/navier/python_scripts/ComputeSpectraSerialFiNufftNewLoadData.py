#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re, finufft
import argparse
import pandas as pd
import sys

# -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Visualise MFEM velocity magnitude field.")
parser.add_argument("data_file", help="Path to SamplePointsAtDoFs*.txt")
args  = parser.parse_args()
fname = args.data_file

eps = 1e-12           # requested accuracy for FINUFFT

plt.figure(figsize=(8, 6))

print(f"[Rank 0] Reading file: {fname}")

# ---- 1. header ----
header_lines = []
with open(fname, "r") as fh:
    for line in fh:
        if not line.lstrip().startswith("#"):
            break
        header_lines.append(line)
for ln in header_lines:
    m = re.search(r"Step\s+(\d+)", ln)
    if m: step_number_extracted = m.group(1)
    m = re.search(r"Time\s+([0-9.+-Ee]+)", ln)
    if m: time_extracted = float(m.group(1))
if step_number_extracted is None:
    step_number_extracted = "Unknown"
if time_extracted is None:
    time_extracted = 0.0
print(f"[Rank 0] Header: Step {step_number_extracted}, "
      f"Time {time_extracted:.3e}")

# ---- 2. numerical payload ----
# np.loadtxt skips lines that start with '#'
flat = np.loadtxt(fname, comments="#", dtype=np.float64).ravel()
if flat.size % 6 != 0:
    raise ValueError("Data length not divisible by 6 – corrupted file?")
ND = flat.size // 6

# Slice blocks: [x | y | z | u | v | w]
xpos = flat[0*ND : 1*ND]
ypos = flat[1*ND : 2*ND]
zpos = flat[2*ND : 3*ND]
velx = flat[3*ND : 4*ND]
vely = flat[4*ND : 5*ND]
velz = flat[5*ND : 6*ND]

# ---- 3. round coords, build unique grids ----
xpos_r = np.round(xpos, 10)
ypos_r = np.round(ypos, 10)
zpos_r = np.round(zpos, 10)

x_unique = np.unique(xpos_r)
y_unique = np.unique(ypos_r)
z_unique = np.unique(zpos_r)
nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

print(f"[Rank 0] Grid size = {nx} × {ny} × {nz}")
if nx * ny * nz != ND:
    print("[Rank 0] Warning: node count mismatch!")

# ---- 4. populate 3-D velocity arrays ----
velx_grid = np.full((nx, ny, nz), np.nan)
vely_grid = np.full((nx, ny, nz), np.nan)
velz_grid = np.full((nx, ny, nz), np.nan)
xi = {v:i for i,v in enumerate(x_unique)}
yi = {v:i for i,v in enumerate(y_unique)}
zi = {v:i for i,v in enumerate(z_unique)}
for k in range(ND):
    velx_grid[xi[xpos_r[k]], yi[ypos_r[k]], zi[zpos_r[k]]] = velx[k]
    vely_grid[xi[xpos_r[k]], yi[ypos_r[k]], zi[zpos_r[k]]] = vely[k]
    velz_grid[xi[xpos_r[k]], yi[ypos_r[k]], zi[zpos_r[k]]] = velz[k]
velx_grid = np.nan_to_num(velx_grid)
vely_grid = np.nan_to_num(vely_grid)
velz_grid = np.nan_to_num(velz_grid)

# ---- 5. physical-space KE and |u| ----
tke_physical = 0.5*np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
print(f"[Rank 0] ⟨KE⟩ = {tke_physical:.6f}")

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

x_s = map_to_pi(xpos, x_unique[0], Lx)
y_s = map_to_pi(ypos, y_unique[0], Ly)
z_s = map_to_pi(zpos, z_unique[0], Lz)

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
# print(f"FINUFFT KE: {E3d_nu.sum()*Ntot:.6e}  |  FFT KE: {E3d_ft.sum()*Ntot:.6e}")
print(f"FINUFFT KE: {E3d_nu.mean()*Ntot:.6e}  |  FFT KE: {E3d_ft.mean()*Ntot:.6e}")

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
# print(f"FINUFFT KE: {E3d_nu.sum()*Ntot:.6e}  |  FFT KE: {E3d_ft.sum()*Ntot:.6e}")
print(f"FINUFFT KE: {E3d_nu.mean()*Ntot:.6e}  |  FFT KE: {E3d_ft.mean()*Ntot:.6e}")

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

