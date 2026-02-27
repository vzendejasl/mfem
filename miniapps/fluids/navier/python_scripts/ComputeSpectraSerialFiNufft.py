#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re, finufft
import argparse
import pandas as pd
import sys
import os

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def wrap_to_half_open(arr, amin, L):
    """Map arr to [amin, amin+L) robustly (periodic)."""
    out = (arr - amin) % L + amin
    # pull any numerical hits exactly at top edge back inside
    tol = 1e-12 * (abs(amin) + abs(L) + 1.0)
    top = amin + L
    mask = out >= (top - tol)
    out[mask] -= L
    return out

def deduce_L_and_wrap(coord, round_dec=12):
    """
    Infer periodic box: use span = u[-1]-u[0] as the true length (endpoints present),
    wrap to half-open [amin, amin+L), and report a representative dx ~ span/(n-1).
    """
    u = np.unique(np.round(coord, round_dec))
    if len(u) == 1:
        amin = float(u[0]); L = 1.0; dx = 1.0
        return np.full_like(coord, amin), (amin, L, dx)
    amin = float(u[0])
    span = float(u[-1] - u[0])
    if span <= 0:
        span = 1.0
    L = span
    wrapped = wrap_to_half_open(coord, amin, L)
    dx = span / max(len(u) - 1, 1)
    return wrapped, (amin, L, dx)

def periodic_1d_voronoi_weights(u, L):
    """
    Periodic 1D Voronoi (midpoint) weights for sorted unique coords u in [a, a+L).
    w[i] = 0.5 * (gap to next + gap from prev), with wrap at the seam.
    """
    u = np.array(u, dtype=float)
    n = len(u)
    w = np.empty(n, dtype=float)
    for i in range(n):
        # forward gap
        if i < n - 1:
            g_f = u[i+1] - u[i]
        else:
            g_f = (u[0] + L) - u[i]
        # backward gap
        if i > 0:
            g_b = u[i] - u[i-1]
        else:
            g_b = u[0] - (u[-1] - L)
        w[i] = 0.5 * (g_f + g_b)
    return w

# --------------------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Compute 3D energy spectrum using FINUFFT from (possibly nonuniform) velocity samples.')
parser.add_argument('data_file', type=str, help='Path to the data file')
parser.add_argument('--no-rescale', action='store_true',
                    help='Do NOT force exact TKE match (only weighted NUFFT Parseval).')
args = parser.parse_args()
data_filename = args.data_file
force_rescale = (not args.no_rescale)

eps = 1e-12  # requested accuracy for FINUFFT

plt.figure(figsize=(8, 6))

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
print(f"Reading header and data from file:\n  {data_filename}")
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
print(f"Header: Step = {step_number_extracted}, Time = {time_extracted:.3e}")

# --------------------------------------------------------------------------------------
# Load data (chunked)
# --------------------------------------------------------------------------------------
print("Loading data in chunks (skipping header)...")
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
    xp = np.round(chunk.iloc[:, 0].values, 12)
    yp = np.round(chunk.iloc[:, 1].values, 12)
    zp = np.round(chunk.iloc[:, 2].values, 12)
    vx = chunk.iloc[:, 3].values
    vy = chunk.iloc[:, 4].values
    vz = chunk.iloc[:, 5].values
    xpos_list.append(xp); ypos_list.append(yp); zpos_list.append(zp)
    velx_list.append(vx); vely_list.append(vy); velz_list.append(vz)

total_pts = sum(arr.size for arr in xpos_list)
xpos = np.empty(total_pts); ypos = np.empty(total_pts); zpos = np.empty(total_pts)
velx = np.empty(total_pts); vely = np.empty(total_pts); velz = np.empty(total_pts)

offset = 0
for xp, yp, zp, vx, vy, vz in zip(xpos_list, ypos_list, zpos_list, velx_list, vely_list, velz_list):
    n = xp.size
    xpos[offset:offset+n] = xp; ypos[offset:offset+n] = yp; zpos[offset:offset+n] = zp
    velx[offset:offset+n] = vx; vely[offset:offset+n] = vy; velz[offset:offset+n] = vz
    offset += n
del xpos_list, ypos_list, zpos_list, velx_list, vely_list, velz_list

# --------------------------------------------------------------------------------------
# Periodic canonicalization to half-open box, then deduplicate
# --------------------------------------------------------------------------------------
xpos, (xmin, Lx, dx_est_x) = deduce_L_and_wrap(xpos, round_dec=12)
ypos, (ymin, Ly, dx_est_y) = deduce_L_and_wrap(ypos, round_dec=12)
zpos, (zmin, Lz, dx_est_z) = deduce_L_and_wrap(zpos, round_dec=12)

# Round after wrapping to stabilize uniqueness
xpos_r = np.round(xpos, 12); ypos_r = np.round(ypos, 12); zpos_r = np.round(zpos, 12)

positions = np.stack([xpos_r, ypos_r, zpos_r], axis=1)
_, unique_indices = np.unique(positions, axis=0, return_index=True)

xpos_r = xpos_r[unique_indices]; ypos_r = ypos_r[unique_indices]; zpos_r = zpos_r[unique_indices]
velx   = velx[unique_indices];   vely   = vely[unique_indices];   velz   = velz[unique_indices]

# Grid info
x_unique = np.unique(xpos_r); nx = len(x_unique)
y_unique = np.unique(ypos_r); ny = len(y_unique)
z_unique = np.unique(zpos_r); nz = len(z_unique)

print(f"Number of unique x values: {nx}")
print(f"Number of unique y values: {ny}")
print(f"Number of unique z values: {nz}")

expected_num_points = nx * ny * nz
actual_num_points   = xpos_r.size
print(f"Expected number of points: {expected_num_points}")
print(f"Actual number of points:   {actual_num_points}")
if actual_num_points != expected_num_points:
    print("ERROR: points do not form a full tensor grid after canonicalization!")
    sys.exit(1)

# Lexicographic sort (i,j,k) for consistent indexing
x_idx = np.searchsorted(x_unique, xpos_r)
y_idx = np.searchsorted(y_unique, ypos_r)
z_idx = np.searchsorted(z_unique, zpos_r)
sort_indices = np.lexsort((z_idx, y_idx, x_idx))
xpos_r = xpos_r[sort_indices]; ypos_r = ypos_r[sort_indices]; zpos_r = zpos_r[sort_indices]
velx   = velx[sort_indices];   vely   = vely[sort_indices];   velz   = velz[sort_indices]
x_idx  = x_idx[sort_indices];  y_idx  = y_idx[sort_indices];  z_idx  = z_idx[sort_indices]

# --------------------------------------------------------------------------------------
# Voronoi volume weights (periodic, separable)
# --------------------------------------------------------------------------------------
wx = periodic_1d_voronoi_weights(x_unique, Lx)
wy = periodic_1d_voronoi_weights(y_unique, Ly)
wz = periodic_1d_voronoi_weights(z_unique, Lz)
# 3D volume weight per gridpoint in current ordering
wvol = wx[x_idx] * wy[y_idx] * wz[z_idx]
Wtot = wx.sum() * wy.sum() * wz.sum()   # should be ~ Lx*Ly*Lz

# --------------------------------------------------------------------------------------
# Energies (unweighted vs weighted)
# --------------------------------------------------------------------------------------
tke_phys_unweighted = 0.5 * np.sum(velx**2 + vely**2 + velz**2)
tke_phys_weighted   = 0.5 * np.sum(wvol * (velx**2 + vely**2 + velz**2))
print(f"[Physical] TKE (unweighted samples) = {tke_phys_unweighted:.10e}")
print(f"[Physical] TKE (weighted, ~integral) = {tke_phys_weighted:.10e}  (Wtot≈{Wtot:.6e})")

# --------------------------------------------------------------------------------------
# Map to [-pi, pi) for FINUFFT
# --------------------------------------------------------------------------------------
x_s = (xpos_r - xmin) * (2*np.pi / Lx) - np.pi
y_s = (ypos_r - ymin) * (2*np.pi / Ly) - np.pi
z_s = (zpos_r - zmin) * (2*np.pi / Lz) - np.pi

# --------------------------------------------------------------------------------------
# NUFFT (type-1) of sqrt(w)*u  → improves Parseval for nonuniform nodes
# --------------------------------------------------------------------------------------
Ntot = nx * ny * nz
sqrtw = np.sqrt(wvol)

def nufft_type1(x_, y_, z_, c_):
    F_flat = finufft.nufft3d1(
        x_, y_, z_,
        c_.astype(np.complex128),
        (nx, ny, nz), eps=eps, isign=1, modeord=1
    )
    return F_flat.reshape((nx, ny, nz))

# transform of sqrt(w)*u
Fvx = nufft_type1(x_s, y_s, z_s, sqrtw * velx)
Fvy = nufft_type1(x_s, y_s, z_s, sqrtw * vely)
Fvz = nufft_type1(x_s, y_s, z_s, sqrtw * velz)

# center zero mode
Fvx = np.fft.fftshift(Fvx); Fvy = np.fft.fftshift(Fvy); Fvz = np.fft.fftshift(Fvz)

# Spectral energy from weighted transform.
# With uniform grids this reduces to: ∑ w|u|^2  ≈ (1/Ntot) ∑ |F|^2.
E3d = 0.5 * (np.abs(Fvx)**2 + np.abs(Fvy)**2 + np.abs(Fvz)**2)
tke_spec_weighted = E3d.sum() / Ntot
print(f"[Spectral] TKE (weighted NUFFT)   = {tke_spec_weighted:.10e}")
rel_err = abs(tke_spec_weighted - tke_phys_weighted) / max(tke_phys_weighted, 1e-30)
print(f"[Parseval] weighted rel. error    = {rel_err:.3e}")

# Optional: force exact match (scalar rescale that preserves spectral shape)
if force_rescale and tke_spec_weighted > 0:
    gamma = tke_phys_weighted / tke_spec_weighted
    scale = np.sqrt(gamma)
    Fvx *= scale; Fvy *= scale; Fvz *= scale
    E3d = 0.5 * (np.abs(Fvx)**2 + np.abs(Fvy)**2 + np.abs(Fvz)**2)
    tke_spec_weighted = E3d.sum() / Ntot
    print(f"[Rescale ] Applied global scale √γ={scale:.6e} so TKE matches exactly.")

print(f"[Check   ] TKE_phys(weighted)={tke_phys_weighted:.10e},  TKE_spec={tke_spec_weighted:.10e}")

# --------------------------------------------------------------------------------------
# Radial spectrum (mode-index shells); plot in cycles/length if cubic
# --------------------------------------------------------------------------------------
kx_idx = np.fft.fftshift(np.fft.fftfreq(nx) * nx)
ky_idx = np.fft.fftshift(np.fft.fftfreq(ny) * ny)
kz_idx = np.fft.fftshift(np.fft.fftfreq(nz) * nz)
KX_i, KY_i, KZ_i = np.meshgrid(kx_idx, ky_idx, kz_idx, indexing='ij')
K_idx = np.sqrt(KX_i**2 + KY_i**2 + KZ_i**2)

Kmax_idx = int(np.floor(K_idx.max()))
edges_idx = np.arange(0, Kmax_idx + 1) - 0.5
centers_idx = 0.5 * (edges_idx[:-1] + edges_idx[1:])
E_k, _ = np.histogram(K_idx.ravel(), bins=edges_idx, weights=E3d.ravel())

equal_box = np.isclose(Lx, Ly, rtol=1e-6, atol=0) and np.isclose(Lx, Lz, rtol=1e-6, atol=0)
if equal_box:
    k_centers = centers_idx / Lx  # cycles per length
    xlabel = r'$|k|$ (cycles/length)'
else:
    k_centers = centers_idx        # index units
    xlabel = r'Shell index $|m|$'

# --------------------------------------------------------------------------------------
# Save spectrum
# --------------------------------------------------------------------------------------
input_basename = os.path.basename(data_filename)
step_suffix_with_underscore = f'_{step_number_extracted}.txt'
step_suffix_without_underscore = f'{step_number_extracted}.txt'

if input_basename.endswith(step_suffix_with_underscore):
    type_part = input_basename[:-len(step_suffix_with_underscore)]
elif input_basename.endswith(step_suffix_without_underscore):
    type_part = input_basename[:-len(step_suffix_without_underscore)]
else:
    type_part = 'finufft'

if type_part.startswith('sampled_data_'):
    type_part = type_part.replace('sampled_data_', '')
elif type_part.startswith('SampledData'):
    type_part = type_part.replace('SampledData', '')

output_filename = os.path.join(
    os.path.dirname(data_filename),
    f'energy_spectrum_finufft_{type_part}_step_{step_number_extracted}.txt'
)
print(f"Saving energy spectrum to {output_filename}")
np.savetxt(
    output_filename,
    np.column_stack((k_centers, E_k)),
    header=f'k_center  E(k)  (FINUFFT Step {step_number_extracted}, Time {time_extracted:.3e})',
    fmt='%.6e %.6e',
    comments='# '
)

# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------
# Reference slope (avoid divide-by-zero at k=0)
mask = k_centers > 0
ref = np.zeros_like(k_centers)
ref[mask] = (k_centers[mask] / k_centers[mask][0])**(-5/3)

plt.loglog(k_centers[mask], E_k[mask], label=f'FINUFFT (weighted) step {step_number_extracted}')
plt.loglog(k_centers[mask], 0.1 * ref[mask], 'r:', label=r'$k^{-5/3}$')
plt.xlabel(xlabel); plt.ylabel(r'$E(k)$')
title = '3-D energy spectrum (FINUFFT, weighted)'
plt.title(title + ('' if equal_box else ' — non-cubic box: index shells'))
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()
