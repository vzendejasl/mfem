Before that -- I want to normalize my energies. I have this script

#!/usr/bin/env python
"""
This script reads a 3D velocity field from an MFEM discreteley sample
data file, reconstructs a grid, uses mpi4py-fft to compute the energy spectrum,
and saves the wavenumber and energy data to a text file in a memory-efficient manner.

Usage:
    srun -n 4 python ComputeSpectraParallel.py /path/to/data/file.txt
"""

from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import subprocess
import pandas as pd
import argparse

### Memory Logging Functions
def log_memory_rank0(message):
    """Log memory usage (Resident Set Size, RSS) in MB for rank 0."""
    if rank == 0:
        pid = os.getpid()
        cmd = f"ps -p {pid} -o rss"
        try:
            output = subprocess.check_output(cmd, shell=True).decode().splitlines()[1].strip()
            rss_mb = int(output) / 1024
            print(f"[Rank 0] {message} - RSS: {rss_mb:.2f} MB")
        except Exception as e:
            print(f"[Rank 0] Failed to get memory usage: {e}")

def log_memory_global(message):
    """Log total memory usage across all MPI ranks."""
    pid = os.getpid()
    cmd = f"ps -p {pid} -o rss"
    try:
        output = subprocess.check_output(cmd, shell=True).decode().splitlines()[1].strip()
        rss_mb = int(output) / 1024
    except Exception as e:
        rss_mb = 0.0
        print(f"[Rank {rank}] Failed to get memory usage: {e}")
    total_rss = comm.reduce(rss_mb, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"{message} - Total RSS across all ranks: {total_rss:.2f} MB")

### Serial Spectrum Computation (Commented Out for Future Validation)
# def compute_serial_spectrum(velx, vely, velz, dx, dy, dz, step_number, time_value, output_dir):
#     """Compute the energy spectrum serially for validation purposes."""
#     print("[Rank 0] Running serial FFT diagnostic...")
#     fft_velx = np.fft.fftn(velx)
#     fft_vely = np.fft.fftn(vely)
#     fft_velz = np.fft.fftn(velz)
#     log_memory_rank0("After serial FFT computation")
#     fft_velx = np.fft.fftshift(fft_velx)
#     fft_vely = np.fft.fftshift(fft_vely)
#     fft_velz = np.fft.fftshift(fft_velz)
#     norm_factor = (nx * ny * nz)
#     fft_velx /= norm_factor
#     fft_vely /= norm_factor
#     fft_velz /= norm_factor
#     energy_density = 0.5 * (np.abs(fft_velx)**2 + np.abs(fft_vely)**2 + np.abs(fft_velz)**2)
#     log_memory_rank0("After computing serial energy density")
#     kx = np.fft.fftfreq(nx, d=dx/(2*np.pi))
#     ky = np.fft.fftfreq(ny, d=dy/(2*np.pi))
#     kz = np.fft.fftfreq(nz, d=dz/(2*np.pi))
#     kx = np.fft.fftshift(kx)
#     ky = np.fft.fftshift(ky)
#     kz = np.fft.fftshift(kz)
#     KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
#     k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
#     log_memory_rank0("After computing serial wavenumbers")
#     k_flat = k_magnitude.flatten()
#     energy_flat = energy_density.flatten()
#     num_bins = nx
#     k_bin_edges = np.arange(0, num_bins+1) - 0.5
#     k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
#     E_k_serial, _ = np.histogram(k_flat, bins=k_bin_edges, weights=energy_flat)
#     out_file = os.path.join(output_dir, f'serial_energy_spectrum_step_{step_number}.txt')
#     np.savetxt(out_file, np.column_stack((k_bin_centers, E_k_serial)),
#                header=f'Serial Spectrum (Step {step_number}, Time {time_value:.3e})',
#                fmt='%.6e %.6e', comments='# ')
#     log_memory_rank0("After saving serial spectrum")
#     print(f"[Rank 0] Serial energy spectrum saved to: {out_file}")
#     return k_bin_centers, E_k_serial

### MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### Parse Command-Line Arguments
parser = argparse.ArgumentParser(description='Compute energy spectrum from velocity data.')
parser.add_argument('data_file', type=str, help='Path to the data file')
args = parser.parse_args()
data_filename = args.data_file

### File Reading and Grid Reconstruction (Rank 0)
if rank == 0:
    print(f"[Rank 0] Reading header and data from file:\n  {data_filename}")
    # Read header to extract step number and time
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

    # Load data in chunks to handle large files
    print("[Rank 0] Loading data in chunks (skipping header)...")

    chunk_size = 10_000_000
    chunks = pd.read_csv(data_filename, delimiter=' ', skiprows=6, chunksize=chunk_size, header=None)
    xpos, ypos, zpos, velx, vely, velz = [], [], [], [], [], []

    for chunk in chunks:
        xpos.append(chunk.iloc[:, 0].values)
        ypos.append(chunk.iloc[:, 1].values)
        zpos.append(chunk.iloc[:, 2].values)
        velx.append(chunk.iloc[:, 3].values)
        vely.append(chunk.iloc[:, 4].values)
        velz.append(chunk.iloc[:, 5].values)
    xpos = np.concatenate(xpos)
    ypos = np.concatenate(ypos)
    zpos = np.concatenate(zpos)
    velx = np.concatenate(velx)
    vely = np.concatenate(vely)
    velz = np.concatenate(velz)
    log_memory_rank0("After reading data file in chunks")

    # Round positions to handle floating-point precision
    xpos_rounded = np.round(xpos, decimals=10)
    ypos_rounded = np.round(ypos, decimals=10)
    zpos_rounded = np.round(zpos, decimals=10)

    # Determine grid size
    x_unique = np.unique(xpos_rounded)
    y_unique = np.unique(ypos_rounded)
    z_unique = np.unique(zpos_rounded)
    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)
    print(f"Number of unique x values: {nx}")
    print(f"Number of unique y values: {ny}")
    print(f"Number of unique z values: {nz}")

    # Validate data size
    expected_num_points = nx * ny * nz
    actual_num_points = xpos.size

    print(f"Expected number of points: {expected_num_points}")
    print(f"Actual number of points: {actual_num_points}")

    if actual_num_points != expected_num_points:
        print("Warning: The actual number of data points does not match the expected number based on grid sizes.")

    # Reconstruct velocity grids
    velx_grid = np.full((nx, ny, nz), np.nan)
    vely_grid = np.full((nx, ny, nz), np.nan)
    velz_grid = np.full((nx, ny, nz), np.nan)
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}
    for i in range(actual_num_points):
        xi = x_idx[xpos_rounded[i]]
        yi = y_idx[ypos_rounded[i]]
        zi = z_idx[zpos_rounded[i]]
        velx_grid[xi, yi, zi] = velx[i]
        vely_grid[xi, yi, zi] = vely[i]
        velz_grid[xi, yi, zi] = velz[i]
    velx_grid = np.nan_to_num(velx_grid)
    vely_grid = np.nan_to_num(vely_grid)
    velz_grid = np.nan_to_num(velz_grid)
    log_memory_rank0("After reconstructing grids")

    # Compute total kinetic energy in physical space
    tke_physical = 0.5 * np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    print(f"[Rank 0] Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")
else:
    nx = None
    ny = None
    nz = None
    step_number_extracted = None
    time_extracted = None
    tke_physical = None

### Broadcast Grid Dimensions and Header Info
nx = comm.bcast(nx, root=0)
ny = comm.bcast(ny, root=0)
nz = comm.bcast(nz, root=0)
step_number_extracted = comm.bcast(step_number_extracted, root=0)
time_extracted = comm.bcast(time_extracted, root=0)
tke_physical = comm.bcast(tke_physical, root=0)

comm.Barrier()
log_memory_global("After broadcast")

### Setup MPI FFT
subcomm = [MPI.COMM_NULL, MPI.COMM_WORLD, MPI.COMM_NULL]
fft = PFFT(MPI.COMM_WORLD, (nx, ny, nz), axes=(0, 1, 2), dtype=np.complex128, subcomm=subcomm, threads=1)

comm.Barrier()
log_memory_global("After setting up MPI FFT")

# Create distributed arrays
u_dist_x_in = newDistArray(fft, forward_output=False)
u_dist_y_in = newDistArray(fft, forward_output=False)
u_dist_z_in = newDistArray(fft, forward_output=False)
u_dist_x_out = newDistArray(fft, forward_output=True)
u_dist_y_out = newDistArray(fft, forward_output=True)
u_dist_z_out = newDistArray(fft, forward_output=True)

### Distribute Local Data
if rank == 0:
    local_slice = fft.local_slice(forward_output=False)
    local_velx = np.ascontiguousarray(velx_grid[local_slice])
    local_vely = np.ascontiguousarray(vely_grid[local_slice])
    local_velz = np.ascontiguousarray(velz_grid[local_slice])

    for r in range(1, size):
        local_starts = np.zeros(3, dtype=int)
        local_stops = np.zeros(3, dtype=int)
        comm.Recv(local_starts, source=r, tag=10)
        comm.Recv(local_stops, source=r, tag=11)

        sl = tuple(slice(start, stop) for start, stop in zip(local_starts, local_stops))

        send_velx = np.ascontiguousarray(velx_grid[sl])
        send_vely = np.ascontiguousarray(vely_grid[sl])
        send_velz = np.ascontiguousarray(velz_grid[sl])
        comm.Send(send_velx, dest=r, tag=0)
        comm.Send(send_vely, dest=r, tag=1)
        comm.Send(send_velz, dest=r, tag=2)

    log_memory_rank0("After distributing data (before deleting grids)")

    # Optional serial spectrum computation
    # # Compute grid spacings
    # dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    # dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    # dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0

    # k_centers_serial, E_k_serial = compute_serial_spectrum(
    #     velx_grid, vely_grid, velz_grid,
    #     dx, dy, dz,
    #     step_number_extracted,
    #     time_extracted,
    #     os.path.dirname(data_filename)
    # )

    # Free memory
    del velx_grid, vely_grid, velz_grid
    log_memory_rank0("After deleting full grids")

else:

    local_slice = fft.local_slice(forward_output=False)
    local_starts = np.array([sl.start if isinstance(sl, slice) else 0 for sl in local_slice], dtype=int)
    local_stops = np.array([sl.stop if isinstance(sl, slice) else dim for sl, dim in zip(local_slice, (nx, ny, nz))], dtype=int)

    comm.Send(local_starts, dest=0, tag=10)
    comm.Send(local_stops, dest=0, tag=11)

    local_shape = tuple(stop - start for start, stop in zip(local_starts, local_stops))
    local_velx = np.empty(local_shape, dtype=np.float64)
    local_vely = np.empty(local_shape, dtype=np.float64)
    local_velz = np.empty(local_shape, dtype=np.float64)

    comm.Recv(local_velx, source=0, tag=0)
    comm.Recv(local_vely, source=0, tag=1)
    comm.Recv(local_velz, source=0, tag=2)

comm.Barrier()
log_memory_global("After distributing and receiving data")

### Assign Data to FFT Input Arrays
temp_x = local_velx.astype(np.complex128)
temp_y = local_vely.astype(np.complex128)
temp_z = local_velz.astype(np.complex128)

u_dist_x_in[:] = temp_x
u_dist_y_in[:] = temp_y
u_dist_z_in[:] = temp_z

comm.Barrier()
log_memory_global("After assigning to FFT input arrays")

# Clean up temporary arrays
del local_velx, local_vely, local_velz, temp_x, temp_y, temp_z

comm.Barrier()
log_memory_global("After deleting local velocity arrays")

### Perform Forward FFT
fft.forward(u_dist_x_in, u_dist_x_out)
fft.forward(u_dist_y_in, u_dist_y_out)
fft.forward(u_dist_z_in, u_dist_z_out)

comm.Barrier()
log_memory_global("After performing FFT")

# Compute local energy density
local_energy = 0.5 * (np.abs(u_dist_x_out)**2 + np.abs(u_dist_y_out)**2 + np.abs(u_dist_z_out)**2)

### Compute Wavenumbers
if rank == 0:
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
else:
    dx = dy = dz = None

dx = comm.bcast(dx, root=0)
dy = comm.bcast(dy, root=0)
dz = comm.bcast(dz, root=0)

kx = np.fft.fftfreq(nx, d=dx/(2*np.pi))
ky = np.fft.fftfreq(ny, d=dy/(2*np.pi))
kz = np.fft.fftfreq(nz, d=dz/(2*np.pi))

output_slice = fft.local_slice(forward_output=True)
local_kx = kx[output_slice[0]]
local_ky = ky[output_slice[1]]
local_kz = kz[output_slice[2]]

KX_local, KY_local, KZ_local = np.meshgrid(local_kx, local_ky, local_kz, indexing='ij')
k_magnitude_local = np.sqrt(KX_local**2 + KY_local**2 + KZ_local**2)
comm.Barrier()
log_memory_global("After computing energy and wavenumbers")

### Compute Energy Spectrum
k_flat_local = k_magnitude_local.flatten()
energy_flat_local = local_energy.flatten()

k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)
num_bins = nx

k_bin_edges = np.arange(0, num_bins+1) - 0.5
k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

local_E_k, _ = np.histogram(k_flat_local, bins=k_bin_edges, weights=energy_flat_local)

if rank == 0:
    E_k = np.zeros_like(local_E_k)
else:
    E_k = None

comm.Reduce(local_E_k, E_k, op=MPI.SUM, root=0)

# Compute total kinetic energy in Fourier space
local_tke_fourier = np.sum(local_energy)
tke_fourier = comm.allreduce(local_tke_fourier, op=MPI.SUM)

### Output Handling (Rank 0)
if rank == 0:

    # Determine output filename with flexible type extraction
    input_basename = os.path.basename(data_filename)
    step_suffix_with_underscore = f'_{step_number_extracted}.txt'
    step_suffix_without_underscore = f'{step_number_extracted}.txt'

    if input_basename.endswith(step_suffix_with_underscore):
        type_part = input_basename[:-len(step_suffix_with_underscore)]  # Remove '_9002.txt'
    elif input_basename.endswith(step_suffix_without_underscore):
        type_part = input_basename[:-len(step_suffix_without_underscore)]  # Remove '9002.txt'
    else:
        type_part = 'unknown'
    if type_part != 'unknown':
        if type_part.startswith('sampled_data_'):
            type_part = type_part.replace('sampled_data_', '')
        else:
            type_part = 'unknown'  # Fallback if prefix doesn't match expected pattern

    output_filename = os.path.join(os.path.dirname(data_filename), f'energy_spectrum_{type_part}_step_{step_number_extracted}.txt')

    print(f"[Rank 0] Saving parallel energy spectrum to {output_filename}")

    np.savetxt(output_filename, np.column_stack((k_bin_centers, E_k)),
               header=f'Wavenumber_k Energy_E(k) (Step {step_number_extracted}, Time {time_extracted:.3e})',
               fmt='%.6e %.6e', comments='# ')

    log_memory_rank0("After saving spectrum")

    # Energy conservation check
    print("\n--- Total Kinetic Energy Comparison ---")
    print(f"Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")
    print(f"Total Kinetic Energy in Fourier Space (TKE_fourier):  {tke_fourier:.6f}")

    if tke_physical != 0:
        rel_error = np.abs(tke_physical - tke_fourier) / tke_physical * 100
    else:
        rel_error = np.nan
    print(f"Relative Energy Error: {rel_error:.6f}%")

    # Plotting
    print("[Rank 0] Plotting energy spectra (parallel vs serial)...")

    # Uncomment to include plot
    plt.figure(figsize=(10, 8))
    label_str = f"Step {step_number_extracted}, Time = {time_extracted:.3e}"
    plt.loglog(k_bin_centers, E_k, 'b-', label='Parallel Spectrum')

    # # Uncomment to include serial spectrum in the plot (if computed)
    # # if 'k_centers_serial' in locals() and 'E_k_serial' in locals():
    # #     plt.loglog(k_centers_serial, E_k_serial, 'g--', label='Serial Spectrum')

    k_ref = 1
    E_ref = 0.1e1
    E_line = E_ref * (k_bin_centers / k_ref)**(-5.0/3.0)
    plt.loglog(k_bin_centers, E_line, 'r--', label='$k^{-5/3}$ slope')
    plt.xlabel('Wavenumber k')
    plt.ylabel('E(k)')
    plt.title('Energy Spectrum: Parallel FFT')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

### Finalize MPI
comm.Barrier()
log_memory_global("Before finalizing MPI")
MPI.Finalize()

And this one which I have been working on

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re, os, finufft
import argparse

# ────────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Compute TKE and velocity magnitude from velocity data.')
parser.add_argument('data_file', type=str, help='Path to the data file')
args = parser.parse_args()
data_filename = args.data_file

eps  = 1e-12           # requested accuracy for FINUFFT (≈ machine precision)
# ────────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 6))

for fname in [data_filename]:
    # 1) header -----------------------------------------------------------------
    with open(fname) as f:
        hdr = [next(f) for _ in range(6)]
    step = next((re.search(r'Step\s*=\s*(\d+)', L).group(1)
                 for L in hdr if 'Step' in L), '???')
    tval = next((float(re.search(r'Time\s*=\s*([\deE+.-]+)', L).group(1))
                 for L in hdr if 'Time' in L), 0.0)

    # 2) load samples -----------------------------------------------------------
    data = np.genfromtxt(fname, skip_header=6)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]

    # 3) rebuild uniform cube ---------------------------------------------------
    xr, yr, zr = np.round(x, 10), np.round(y, 10), np.round(z, 10)
    ux, uy, uz = np.unique(xr), np.unique(yr), np.unique(zr)
    nx, ny, nz = ux.size, uy.size, uz.size
    ix = {v: i for i, v in enumerate(ux)}
    iy = {v: i for i, v in enumerate(uy)}
    iz = {v: i for i, v in enumerate(uz)}
    Gx = np.zeros((nx, ny, nz)); Gy = np.zeros_like(Gx); Gz = np.zeros_like(Gx)
    for p in range(x.size):
        i, j, k = ix[xr[p]], iy[yr[p]], iz[zr[p]]
        Gx[i, j, k] = vx[p]; Gy[i, j, k] = vy[p]; Gz[i, j, k] = vz[p]

    print(f"Number of unique x values: {nx}")
    print(f"Number of unique y values: {ny}")
    print(f"Number of unique z values: {nz}")

    expected_num_points = nx * ny * nz
    actual_num_points = x.size

    print(f"Expected number of points: {expected_num_points}")
    print(f"Actual number of points: {actual_num_points}")

    if actual_num_points != expected_num_points:
        print("Warning: The actual number of data points does not match the expected number based on grid sizes.")


    spatial_ke = 0.5*np.mean(Gx**2 + Gy**2 + Gz**2)
    print(f"Spatial KE : {spatial_ke:.6e}")

    # 4) coordinates remapped to [-π, π) ---------------------------------------
    # FINUFFT expects coords in [-π, π); scale each axis independently
    Lx, Ly, Lz = ux[-1] - ux[0] + (ux[1] - ux[0]), \
                 uy[-1] - uy[0] + (uy[1] - uy[0]), \
                 uz[-1] - uz[0] + (uz[1] - uz[0])

    x_scaled = (x - ux[0]) * (2*np.pi / Lx) - np.pi
    y_scaled = (y - uy[0]) * (2*np.pi / Ly) - np.pi
    z_scaled = (z - uz[0]) * (2*np.pi / Lz) - np.pi

    # 5) FINUFFT type-1 (non-uniform → uniform modes) ---------------------------
    Nd = (nx, ny, nz)
    Ntot = np.prod(Nd)

    def finufft_forward(values):
        """Helper: one NUFFT, reshape, 1/N scaling to match NumPy FFT."""
        f = finufft.nufft3d1(
                x_scaled, y_scaled, z_scaled,
                values.astype(np.complex128),       
                Nd, eps=eps, isign=1, modeord=1 
            ).reshape(Nd)
        return f / x.size

    Fvx_nu = finufft_forward(vx)
    Fvy_nu = finufft_forward(vy)
    Fvz_nu = finufft_forward(vz)

    # 6) NumPy FFT (no fftshift) -------------------------------------------------
    Fvx_ft = np.fft.fftn(Gx) / Ntot
    Fvy_ft = np.fft.fftn(Gy) / Ntot
    Fvz_ft = np.fft.fftn(Gz) / Ntot

    # 7) energies ---------------------------------------------------------------
    E3d_nu = 0.5*(np.abs(Fvx_nu)**2 + np.abs(Fvy_nu)**2 + np.abs(Fvz_nu)**2)
    E3d_ft = 0.5*(np.abs(Fvx_ft)**2 + np.abs(Fvy_ft)**2 + np.abs(Fvz_ft)**2)
    print(f"FINUFFT KE: {E3d_nu.sum()*Ntot:.6e}  |  FFT KE: {E3d_ft.sum()*Ntot:.6e}")

    # 8) k-grid and radial bins (unchanged) -------------------------------------
    kx = np.fft.fftfreq(nx)*2*np.pi
    ky = np.fft.fftfreq(ny)*2*np.pi
    kz = np.fft.fftfreq(nz)*2*np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()

    bins = np.linspace(0, k_mag.max(), nx+1)
    k_centers = 0.5*(bins[:-1] + bins[1:])
    E_k_nu, _ = np.histogram(k_mag, bins=bins, weights=E3d_nu.ravel())
    E_k_ft, _ = np.histogram(k_mag, bins=bins, weights=E3d_ft.ravel())

    # 9) plot -------------------------------------------------------------------
    plt.loglog(k_centers, E_k_nu, 'b-',  label=f'FINUFFT step {step}')
    plt.loglog(k_centers, E_k_ft, 'k--', label='FFT')

# reference −5/3
plt.loglog(k_centers, 0.1*(k_centers/k_centers[1])**(-5/3), 'r:', label=r'$k^{-5/3}$')
plt.xlabel(r'$|k|$'); plt.ylabel(r'$E(k)$'); plt.title('3-D energy spectrum')
plt.legend(); plt.grid(True, ls=':'); plt.tight_layout(); plt.show()
