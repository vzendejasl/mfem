#!/usr/bin/env python
"""
This script reads a 3D velocity field from an MFEM “element centers”
data file, reconstructs a grid, uses mpi4py-fft to compute the energy spectrum,
and saves the wavenumber and energy data to a text file.

Usage:
    mpiexec -n 4 python ComputeSpectraParallel.py
"""

from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# -------------------- MPI SETUP --------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------- FILE READING (Rank 0) --------------------
file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_64_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P4/'
data_filename = os.path.join(file_directory, 'cycle_10001', 'element_centers_10001.txt')

# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/'
# data_filename = os.path.join(file_directory, 'cycle_9001', 'element_centers_9001.txt')

# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/'
# data_filename = os.path.join(file_directory, 'cycle_9001', 'element_centers_9001.txt')

# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/'
# data_filename = os.path.join(file_directory, 'cycle_9001','element_centers_9001.txt')

# file_directory = '/g/g11/zendejas/Documents/mfem_build/mfem/miniapps/navier/ElementCentersVelocity_Re1600NumPtsPerDir4RefLv0P4'
# data_filename = os.path.join(file_directory, 'cycle_0', 'element_centers_0.txt')

# file_directory = '/p/lustre2/zendejas/TestCases/mfem/TGV/Order2_Re1600/tgv_512/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv4P2/'
# data_filename = os.path.join(file_directory, 'cycle_9000', 'element_centers_9000.txt')

# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P2/'
# data_filename = os.path.join(file_directory, 'cycle_9000','element_centers_9000.txt')

if rank == 0:
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
    
    print("[Rank 0] Loading data (skipping header)...")
    data = np.genfromtxt(data_filename, delimiter=' ', skip_header=6)
    xpos = data[:, 0]
    ypos = data[:, 1]
    zpos = data[:, 2]
    velx = data[:, 3]
    vely = data[:, 4]
    velz = data[:, 5]
    
    xpos_r = np.round(xpos, decimals=12)
    ypos_r = np.round(ypos, decimals=12)
    zpos_r = np.round(zpos, decimals=12)
    
    x_unique = np.unique(xpos_r)
    y_unique = np.unique(ypos_r)
    z_unique = np.unique(zpos_r)
    
    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)
    print(f"[Rank 0] Unique grid points: nx = {nx}, ny = {ny}, nz = {nz}")
    
    expected_points = nx * ny * nz
    actual_points = xpos.size
    print(f"[Rank 0] Expected number of grid points: {expected_points}")
    print(f"[Rank 0] Actual number of data points: {actual_points}")
    if actual_points != expected_points:
        print("WARNING: Data point count does not match grid dimensions. Will average duplicates.")
    
    velx_sum = np.zeros((nx, ny, nz))
    vely_sum = np.zeros((nx, ny, nz))
    velz_sum = np.zeros((nx, ny, nz))
    counts = np.zeros((nx, ny, nz))
    
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}
    
    for i in range(actual_points):
        ix = x_idx[xpos_r[i]]
        iy = y_idx[ypos_r[i]]
        iz = z_idx[zpos_r[i]]
        velx_sum[ix, iy, iz] += velx[i]
        vely_sum[ix, iy, iz] += vely[i]
        velz_sum[ix, iy, iz] += velz[i]
        counts[ix, iy, iz] += 1
    
    velx_grid = np.where(counts > 0, velx_sum / counts, 0)
    vely_grid = np.where(counts > 0, vely_sum / counts, 0)
    velz_grid = np.where(counts > 0, velz_sum / counts, 0)
    
    tke_physical = 0.5 * np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    print(f"[Rank 0] Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")
else:
    nx = ny = nz = None
    velx_grid = vely_grid = velz_grid = None
    step_number_extracted = None
    time_extracted = None

# Broadcast grid dimensions and header info
nx = comm.bcast(nx, root=0)
ny = comm.bcast(ny, root=0)
nz = comm.bcast(nz, root=0)
step_number_extracted = comm.bcast(step_number_extracted, root=0)
time_extracted = comm.bcast(time_extracted, root=0)

if rank != 0:
    velx_grid = np.empty((nx, ny, nz), dtype=np.float64)
    vely_grid = np.empty((nx, ny, nz), dtype=np.float64)
    velz_grid = np.empty((nx, ny, nz), dtype=np.float64)

comm.Bcast(velx_grid, root=0)
comm.Bcast(vely_grid, root=0)
comm.Bcast(velz_grid, root=0)
if rank == 0:
    print(f"[Rank 0] Velocity grids constructed and broadcast.")

# ------------------- Setup MPI FFT using mpi4py-fft ---------------------
global_shape = (nx, ny, nz)
if rank == 0:
    print("[Rank 0] Setting up MPI FFT object...")

subcomm = [MPI.COMM_NULL, MPI.COMM_WORLD, MPI.COMM_NULL]
fft = PFFT(MPI.COMM_WORLD, (nx, ny, nz), axes=(0, 1, 2), dtype=np.complex128, subcomm=subcomm)

if rank == 0:
    print("[Rank 0] MPI FFT object created.")

if rank == 0:
    print("[Rank 0] Creating distributed arrays...")
u_dist_x_in = newDistArray(fft, forward_output=False)
u_dist_y_in = newDistArray(fft, forward_output=False)
u_dist_z_in = newDistArray(fft, forward_output=False)
u_dist_x_out = newDistArray(fft, forward_output=True)
u_dist_y_out = newDistArray(fft, forward_output=True)
u_dist_z_out = newDistArray(fft, forward_output=True)
if rank == 0:
    print("[Rank 0] Distributed arrays created.")

local_slice = fft.local_slice()
if rank == 0:
    print(f"[Rank 0] Local input slice: {local_slice}")

temp_x = velx_grid[local_slice].astype(np.complex128)
temp_y = vely_grid[local_slice].astype(np.complex128)
temp_z = velz_grid[local_slice].astype(np.complex128)
u_dist_x_in[:] = temp_x.transpose((1, 2, 0))  # Adjusted for 2D decomposition
u_dist_y_in[:] = temp_y.transpose((1, 2, 0))
u_dist_z_in[:] = temp_z.transpose((1, 2, 0))

if rank == 0:
    print("[Rank 0] Data assigned and transposed for FFT.")
    print(f"[Rank 0] u_dist_x_in.shape: {u_dist_x_in.shape}")

# ------------------- Perform forward FFT ---------------------
if rank == 0:
    print("Performing forward FFT on velocity components...")
fft.forward(u_dist_x_in, u_dist_x_out)
fft.forward(u_dist_y_in, u_dist_y_out)
fft.forward(u_dist_z_in, u_dist_z_out)
if rank == 0:
    print("Forward FFT completed.")

# Compute local spectral energy density
local_energy = 0.5 * (np.abs(u_dist_x_out)**2 + np.abs(u_dist_y_out)**2 + np.abs(u_dist_z_out)**2)
if rank == 0:
    print(f"[Rank 0] local_energy.shape: {local_energy.shape}")

# Create a full-sized array for the global energy
global_energy_full = np.zeros((nx, ny, nz), dtype=np.float64)

# Place local energy into the correct portion of the full grid
global_energy_full[local_slice] = local_energy

# Allreduce to sum contributions into the full grid
comm.Allreduce(MPI.IN_PLACE, global_energy_full, op=MPI.SUM)
if rank == 0:
    print("Global spectral energy density computed via Allreduce.")
    print(f"[Rank 0] global_energy_full.shape: {global_energy_full.shape}")
    
    global_energy_full = np.fft.fftshift(global_energy_full)
    
    print("[Rank 0] Building global wavenumber grid for binning...")
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
    kx = np.fft.fftfreq(nx, d=dx/(2*np.pi)); kx = np.fft.fftshift(kx)
    ky = np.fft.fftfreq(ny, d=dy/(2*np.pi)); ky = np.fft.fftshift(ky)
    kz = np.fft.fftfreq(nz, d=dz/(2*np.pi)); kz = np.fft.fftshift(kz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    k_max = np.max(k_magnitude)

    # k_max = np.max(kx)
    num_bins = nx // 2
    k_bins = np.linspace(0, k_max, num=num_bins)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    k_flat = k_magnitude.flatten()
    energy_flat = global_energy_full.flatten()
    E_k, _ = np.histogram(k_flat, bins=k_bins, weights=energy_flat)

    # Save wavenumbers and energy to a text file
    output_filename = os.path.join(file_directory, f'energy_spectrum_step_{step_number_extracted}.txt')
    print(f"[Rank 0] Saving energy spectrum to {output_filename}")
    np.savetxt(output_filename, np.column_stack((k_bin_centers, E_k)), 
               header=f'Wavenumber_k Energy_E(k) (Step {step_number_extracted}, Time {time_extracted:.3e})', 
               fmt='%.6e %.6e', comments='# ')
    
    tke_fourier = np.sum(global_energy_full)
    print("\n--- Total Kinetic Energy Comparison ---")
    print(f"Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")
    print(f"Total Kinetic Energy in Fourier Space (TKE_fourier):  {tke_fourier:.6f}")
    if tke_physical != 0:
        rel_error = np.abs(tke_physical - tke_fourier) / tke_physical * 100
    else:
        rel_error = np.nan
    print(f"Relative Energy Error: {rel_error:.6f}%")
    
    print("[Rank 0] Plotting energy spectrum...")
    plt.figure(figsize=(10, 8))
    label_str = f"Step {step_number_extracted}, Time = {time_extracted:.3e}"
    plt.loglog(k_bin_centers, E_k, 'b-', label=label_str)
    k_ref = k_bin_centers[1] if k_bin_centers[0] == 0 else k_bin_centers[0]
    E_ref = 0.1e1
    E_line = E_ref * (k_bin_centers / k_ref)**(-5.0/3.0)
    plt.loglog(k_bin_centers, E_line, 'r--', label='$k^{-5/3}$ slope')
    plt.xlabel('Wavenumber k')
    plt.ylabel('E(k)')
    plt.title('Energy Spectrum from Distributed FFT')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    # xmax = np.max(kx)
    # xmin = 1
    # ymax = 1e1
    # ymin = 1e-6
    # plt.xlim(xmin,xmax)
    # plt.ylim(ymin,ymax)
    plt.show()
    

MPI.Finalize()
