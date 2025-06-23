#!/usr/bin/env python
"""
This script reads a 3D velocity field from an MFEM discretely sampled
data file, reconstructs the grid, computes the total kinetic energy (TKE)
in physical space, computes the velocity magnitude field, and plots the 3D
distribution of velocity magnitude. MPI is used for parallel distribution of data.

Usage:
    srun -n 4 python ModifiedScript.py /path/to/data/file.txt
"""

from mpi4py import MPI
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

### Helper: Define a simple 1D slicing function along the x-direction
def get_local_slice(n, rank, size):
    counts = [n // size + (1 if r < n % size else 0) for r in range(size)]
    start = sum(counts[:rank])
    stop = start + counts[rank]
    return start, stop

### MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### Parse Command-Line Arguments
parser = argparse.ArgumentParser(description='Compute TKE and velocity magnitude from velocity data.')
parser.add_argument('data_file', type=str, help='Path to the data file')
args = parser.parse_args()
data_filename = args.data_file

# Initialize global variables for non-root processes
if rank != 0:
    nx = ny = nz = None
    step_number_extracted = None
    time_extracted = None
    tke_physical = None

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

    # Load data in chunks to handle large files (skipping header)
    print("[Rank 0] Loading data in chunks (skipping header)...")
    chunk_size = 10_000_000
    chunks = pd.read_csv(data_filename, delimiter=' ', skiprows=6, chunksize=chunk_size, header=None)
    xpos, ypos, zpos, velx, vely, velz = [], [], [], [], [], []
    for chunk in chunks:
        print(chunk)
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

    # Determine grid size from unique positions
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

    # Reconstruct velocity grids using the same sorting as before
    velx_grid = np.full((nx, ny, nz), np.nan)
    vely_grid = np.full((nx, ny, nz), np.nan)
    velz_grid = np.full((nx, ny, nz), np.nan)
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}
    for i in range(actual_num_points):
        xi = x_idx[xpos_rounded[i]]
        # print(ypos_rounded[i])
        yi = y_idx[ypos_rounded[i]]
        zi = z_idx[zpos_rounded[i]]
        velx_grid[xi, yi, zi] = velx[i]
        vely_grid[xi, yi, zi] = vely[i]
        velz_grid[xi, yi, zi] = velz[i]
    velx_grid = np.nan_to_num(velx_grid)
    vely_grid = np.nan_to_num(vely_grid)
    velz_grid = np.nan_to_num(velz_grid)
    log_memory_rank0("After reconstructing grids")

    # Compute total kinetic energy in physical space (TKE)
    tke_physical = 0.5 * np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    print(f"[Rank 0] Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")
    tke_before_process = 0.5 * np.mean(velx**2 + vely**2 + velz**2)
    print(f"[Rank 0] Total Kinetic Energy in Physical Space (TKE before process): {tke_before_process:.6f}")

    # Compute the velocity magnitude field using NumPy
    velocity_magnitude = np.sqrt(velx_grid**2 + vely_grid**2 + velz_grid**2)

    ### Distribute Data Along x-direction Using MPI
    local_start, local_stop = get_local_slice(nx, 0, size)
    local_velx = velx_grid[local_start:local_stop, :, :]
    local_vely = vely_grid[local_start:local_stop, :, :]
    local_velz = velz_grid[local_start:local_stop, :, :]
    local_velocity_magnitude = velocity_magnitude[local_start:local_stop, :, :]

    # Send the corresponding slices to other ranks
    for r in range(1, size):
        s, e = get_local_slice(nx, r, size)
        comm.Send(velx_grid[s:e, :, :], dest=r, tag=100)
        comm.Send(vely_grid[s:e, :, :], dest=r, tag=101)
        comm.Send(velz_grid[s:e, :, :], dest=r, tag=102)
    log_memory_rank0("After distributing data (before deleting grids)")

    # (Optional) Free full-grid arrays if memory is a concern.
    # For plotting, velocity_magnitude and the unique coordinate arrays are still needed.
else:
    # Non-root ranks will receive their respective slices after global parameters are broadcast.
    pass

### Broadcast Global Parameters to All Ranks
nx = comm.bcast(nx, root=0)
ny = comm.bcast(ny, root=0)
nz = comm.bcast(nz, root=0)
step_number_extracted = comm.bcast(step_number_extracted, root=0)
time_extracted = comm.bcast(time_extracted, root=0)
tke_physical = comm.bcast(tke_physical, root=0)

### Distribute Data to Non-Root Ranks
if rank != 0:
    local_start, local_stop = get_local_slice(nx, rank, size)
    local_shape = (local_stop - local_start, ny, nz)
    local_velx = np.empty(local_shape, dtype=np.float64)
    local_vely = np.empty(local_shape, dtype=np.float64)
    local_velz = np.empty(local_shape, dtype=np.float64)
    comm.Recv(local_velx, source=0, tag=100)
    comm.Recv(local_vely, source=0, tag=101)
    comm.Recv(local_velz, source=0, tag=102)
    # Compute the local velocity magnitude on non-root ranks.
    local_velocity_magnitude = np.sqrt(local_velx**2 + local_vely**2 + local_velz**2)

comm.Barrier()
log_memory_global("After distributing and computing local velocity magnitude")

### Plotting the 3D Velocity Magnitude Field (Rank 0)
if rank == 0:
    print("[Rank 0] Plotting the 3D velocity magnitude field...")
    # Create a meshgrid for spatial coordinates using the unique x, y, and z values.
    X, Y, Z = np.meshgrid(x_unique, y_unique, z_unique, indexing='ij')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Use a scatter plot to visualize the velocity magnitude. (For large grids, consider slicing or subsampling.)
    sc = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(),
                    c=velocity_magnitude.flatten(), cmap='viridis', marker='.')
    plt.colorbar(sc, ax=ax, label='Velocity Magnitude')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Velocity Magnitude Field\nTKE (Physical Space) = {tke_physical:.6f}\nStep: {step_number_extracted}, Time: {time_extracted:.3e}')
    plt.tight_layout()
    plt.show()

comm.Barrier()
log_memory_global("Before finalizing MPI")
MPI.Finalize()

