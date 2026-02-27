"""
Parallel FFT computation with correct velocity-position association.
Ensures consistency between serial and parallel runs.
"""

#!!!!!!!! Not working

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
NUM_HEADER_LINES = 6
file_directory = "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P2/"
filename = os.path.join(file_directory, "cycle_9000", "element_centers_9000.txt")

# Step 1: Parse header and count lines (only on rank 0)
total_data_lines = None
step_str = "Unknown"
time_val = 0.0
if rank == 0:
    with open(filename, "r") as f:
        header = [next(f) for _ in range(NUM_HEADER_LINES)]
        for hl in header:
            if "Step" in hl:
                m = re.search(r"Step\s*=\s*(\d+)", hl)
                if m: step_str = m.group(1)
            if "Time" in hl:
                m = re.search(r"Time\s*=\s*([\d.eE+\-]+)", hl)
                if m: time_val = float(m.group(1))
        total_data_lines = sum(1 for _ in f)

# Broadcast metadata to all ranks
total_data_lines = comm.bcast(total_data_lines, root=0)
step_str = comm.bcast(step_str, root=0)
time_val = comm.bcast(time_val, root=0)

# Step 2: Read all data on rank 0
all_data = None
if rank == 0:
    all_data = []
    with open(filename, "r") as f:
        for _ in range(NUM_HEADER_LINES): f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, z = map(float, parts[0:3])
                u, v, w = map(float, parts[3:6])
                all_data.append((round(x, 12), round(y, 12), round(z, 12), u, v, w))

# Step 3: Build global grid on rank 0 and broadcast
if rank == 0:
    global_x = np.unique([d[0] for d in all_data])
    global_y = np.unique([d[1] for d in all_data])
    global_z = np.unique([d[2] for d in all_data])
    Nx, Ny, Nz = len(global_x), len(global_y), len(global_z)
    print(f"[Rank 0] Grid: Nx={Nx}, Ny={Ny}, Nz={Nz}")
else:
    Nx, Ny, Nz = None, None, None
    global_x, global_y, global_z = None, None, None

# Broadcast grid sizes and coordinates
Nx, Ny, Nz = comm.bcast((Nx, Ny, Nz), root=0)
global_x = comm.bcast(global_x, root=0)
global_y = comm.bcast(global_y, root=0)
global_z = comm.bcast(global_z, root=0)

dict_x = {val: i for i, val in enumerate(global_x)}
dict_y = {val: j for j, val in enumerate(global_y)}
dict_z = {val: k for k, val in enumerate(global_z)}

# Step 4: Build full velocity arrays on rank 0
u_full = v_full = w_full = None
if rank == 0:
    u_full = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    v_full = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    w_full = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    counts = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    for x, y, z, u, v, w in all_data:
        i, j, k = dict_x[x], dict_y[y], dict_z[z]
        u_full[i, j, k] += u
        v_full[i, j, k] += v
        w_full[i, j, k] += w
        counts[i, j, k] += 1
    u_full = np.where(counts > 0, u_full / counts, 0)
    v_full = np.where(counts > 0, v_full / counts, 0)
    w_full = np.where(counts > 0, w_full / counts, 0)

# Step 5: Setup FFT and distributed arrays
fft = PFFT(comm, (Nx, Ny, Nz), axes=(0, 1, 2), dtype=np.complex128)
u_dist_x_in = newDistArray(fft, forward_output=False)
u_dist_y_in = newDistArray(fft, forward_output=False)
u_dist_z_in = newDistArray(fft, forward_output=False)

# Get local slab info
local_slice = u_dist_x_in.local_slice()
local_nx = local_slice[0].stop - local_slice[0].start

# Step 6: Prepare and scatter velocity data
if rank == 0:
    # Flatten arrays for Scatterv
    sendbuf_u = np.zeros(local_nx * size * Ny * Nz, dtype=np.float64)
    sendbuf_v = np.zeros(local_nx * size * Ny * Nz, dtype=np.float64)
    sendbuf_w = np.zeros(local_nx * size * Ny * Nz, dtype=np.float64)
    for r in range(size):
        iA = r * local_nx
        iB = (r + 1) * local_nx
        sendbuf_u[iA * Ny * Nz:iB * Ny * Nz] = u_full[iA:iB].ravel()
        sendbuf_v[iA * Ny * Nz:iB * Ny * Nz] = v_full[iA:iB].ravel()
        sendbuf_w[iA * Ny * Nz:iB * Ny * Nz] = w_full[iA:iB].ravel()
else:
    sendbuf_u = sendbuf_v = sendbuf_w = None

# Scatter each component
counts = [local_nx * Ny * Nz] * size
displs = [i * local_nx * Ny * Nz for i in range(size)]
comm.Scatterv([sendbuf_u, counts, displs, MPI.DOUBLE], u_dist_x_in.ravel(), root=0)
comm.Scatterv([sendbuf_v, counts, displs, MPI.DOUBLE], u_dist_y_in.ravel(), root=0)
comm.Scatterv([sendbuf_w, counts, displs, MPI.DOUBLE], u_dist_z_in.ravel(), root=0)

# Reshape received data
u_dist_x_in = u_dist_x_in.reshape((local_nx, Ny, Nz))
u_dist_y_in = u_dist_y_in.reshape((local_nx, Ny, Nz))
u_dist_z_in = u_dist_z_in.reshape((local_nx, Ny, Nz))

# Step 7: Perform FFT
u_dist_x_out = newDistArray(fft, forward_output=True)
u_dist_y_out = newDistArray(fft, forward_output=True)
u_dist_z_out = newDistArray(fft, forward_output=True)
fft.forward(u_dist_x_in, u_dist_x_out)
fft.forward(u_dist_y_in, u_dist_y_out)
fft.forward(u_dist_z_in, u_dist_z_out)

# Compute local energy
local_energy = 0.5 * (np.abs(u_dist_x_out)**2 + np.abs(u_dist_y_out)**2 + np.abs(u_dist_z_out)**2)

# Step 8: Gather energy for spectrum
counts = comm.allgather(local_energy.size)
displs = [sum(counts[:r]) for r in range(size)] if rank == 0 else None
global_energy_1d = np.zeros(Nx * Ny * Nz, dtype=np.float64) if rank == 0 else None
comm.Gatherv(local_energy.ravel(), (global_energy_1d, counts, displs, MPI.DOUBLE), root=0)

# Step 9: Compute and plot spectrum on rank 0
if rank == 0:
    global_energy = global_energy_1d.reshape(Nx, Ny, Nz)
    global_energy = np.fft.fftshift(global_energy)

    # Compute wavenumbers
    dx = global_x[1] - global_x[0]
    kx = np.fft.fftfreq(Nx, dx / (2 * np.pi))
    ky = np.fft.fftfreq(Ny, dx / (2 * np.pi))
    kz = np.fft.fftfreq(Nz, dx / (2 * np.pi))
    kx, ky, kz = [np.fft.fftshift(k) for k in (kx, ky, kz)]
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Compute spectrum
    k_max = np.max(k_magnitude)
    num_bins = Nx // 2
    k_bins = np.linspace(0, k_max, num_bins)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    E_k, _ = np.histogram(k_magnitude.flatten(), bins=k_bins, weights=global_energy.flatten())

    # Save spectrum
    output_filename = os.path.join(file_directory, f"energy_spectrum_step_{step_str}.txt")
    np.savetxt(output_filename, np.column_stack((k_bin_centers, E_k)),
               header=f"Wavenumber_k Energy_E(k) (Step {step_str}, Time {time_val:.3e})",
               fmt="%.6e %.6e", comments="# ")

    # TKE verification
    tke_fourier = np.sum(global_energy)
    tke_physical = 0.5 * np.sum(u_full**2 + v_full**2 + w_full**2)
    print(f"--- Total Kinetic Energy Comparison ---")
    print(f"TKE Physical: {tke_physical:.6f}")
    print(f"TKE Fourier: {tke_fourier:.6f}")
    print(f"Relative Error: {abs(tke_physical - tke_fourier) / tke_physical * 100:.6f}%")

    # Plot
    plt.loglog(k_bin_centers, E_k, "b-", label=f"Step {step_str}, Time = {time_val:.3e}")
    k_ref = k_bin_centers[1] if k_bin_centers[0] == 0 else k_bin_centers[0]
    E_line = 0.1e1 * (k_bin_centers / k_ref)**(-5.0/3.0)
    plt.loglog(k_bin_centers, E_line, "r--", label="$k^{-5/3}$ slope")
    plt.xlabel("Wavenumber k")
    plt.ylabel("E(k)")
    plt.title("Energy Spectrum")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

MPI.Finalize()
