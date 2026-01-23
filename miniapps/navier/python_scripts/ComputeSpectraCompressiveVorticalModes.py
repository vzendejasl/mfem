#!/usr/bin/env python3
"""
Simple Helmholtz-Hodge decomposition and energy spectrum analysis
MINIMAL FIX: Only correcting the dx/(1.0) bug - everything else stays the same

Usage:
    python simple_script.py data_file.txt --header-lines 5                    # Basic analysis with plots
    python simple_script.py data_file.txt --header-lines 6 --no-plot         # No plots (for SLURM)
    python simple_script.py file1.txt file2.txt --header-lines 5             # Multiple files with plots
    python simple_script.py *.txt --header-lines 6 --no-plot                 # Multiple files, no plots
    python simple_script.py data_file.txt --header-lines 5 --visualize       # With velocity field visualization
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.fft as fft
import re
import os
import argparse
import pandas as pd
import h5py


# ------------------------------------------------------------------ #
#  Step 1: Read and parse data file
# ------------------------------------------------------------------ #

def detect_header_lines(filename):
    """Automatically detect the number of header lines in a text file"""
    print(f"  Auto-detecting header length for {filename}...")
    header_count = 0
    try:
        with open(filename, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                # Skip empty lines but count them as header/preamble
                if not line_stripped:
                    header_count += 1
                    continue
                
                try:
                    parts = line_stripped.split()
                    # Try to parse all parts as floats. 
                    # If it works, it's the first data line.
                    [float(x) for x in parts]
                    break 
                except ValueError:
                    # Not a data line, so it's part of the header
                    header_count += 1
    except Exception as e:
        print(f"  Error detecting header: {e}")
        return 0
    
    print(f"  Detected header lines: {header_count}")
    return header_count


def read_data_file_header(filename, header_lines):
    """Read only header information from file"""
    print(f"Reading header from: {filename}")

    hdr = []
    if filename.endswith('.h5'):
        try:
            with h5py.File(filename, 'r') as f:
                if 'header' in f:
                    header_ds = f['header'][:]
                    for line in header_ds:
                        if isinstance(line, bytes):
                            hdr.append(line.decode('utf-8'))
                        else:
                            hdr.append(str(line))
                else:
                    print("  Warning: No 'header' dataset found in HDF5 file.")
        except Exception as e:
            print(f"  Error reading HDF5 header: {e}")
    else:
        with open(filename, 'r') as f:
            hdr = [next(f) for _ in range(header_lines)]

    # Extract step number and time
    step_number = "unknown"
    time_value = 0.0

    for line in hdr:
        if 'Cycle' in line or 'Step' in line:
            match = re.search(r'(?:Cycle|Step)\s*[:=]\s*(\d+)', line)
            if match:
                step_number = match.group(1)
        if 'Time' in line:
            match = re.search(r'Time\s*[:=]\s*([0-9.eE+-]+)', line)
            if match:
                time_value = float(match.group(1))

    print(f"  Step: {step_number}, Time: {time_value:.3e}")
    return step_number, time_value


def read_data_file_chunked(filename, chunk_size=5_000_000, skiprows=5):
    """
    Read velocity data file in chunks using pandas to minimize memory usage

    Returns velocity grid directly without loading entire file into memory at once
    """
    print(f"Reading data from: {filename} (chunked, size={chunk_size})")

    if filename.endswith('.h5'):
        print(f"  Loading data from HDF5: {filename} (chunked read)")
        with h5py.File(filename, 'r') as f:
            dset = f['data']
            total_pts = dset.shape[0]
            print(f"  Total data points: {total_pts}")

            # Preallocate final arrays
            xpos = np.empty(total_pts, dtype=np.float64)
            ypos = np.empty(total_pts, dtype=np.float64)
            zpos = np.empty(total_pts, dtype=np.float64)
            velx = np.empty(total_pts, dtype=np.float64)
            vely = np.empty(total_pts, dtype=np.float64)
            velz = np.empty(total_pts, dtype=np.float64)

            # Read in chunks
            for i in range(0, total_pts, chunk_size):
                end_idx = min(i + chunk_size, total_pts)
                
                # Read raw chunk from HDF5
                chunk_data = dset[i:end_idx]
                
                # Distribute to columns
                # Round coordinates matching text loader logic
                xpos[i:end_idx] = np.round(chunk_data[:, 0], 10)
                ypos[i:end_idx] = np.round(chunk_data[:, 1], 10)
                zpos[i:end_idx] = np.round(chunk_data[:, 2], 10)
                
                velx[i:end_idx] = chunk_data[:, 3]
                vely[i:end_idx] = chunk_data[:, 4]
                velz[i:end_idx] = chunk_data[:, 5]
    else:
        print(f"  Loading data in chunks (skipping {skiprows} header lines)...")
        reader = pd.read_csv(
            filename,
            delimiter=' ',
            skiprows=skiprows,
            header=None,
            chunksize=chunk_size
        )

        # Read into per-chunk lists
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

        # Preallocate the final flat arrays
        total_pts = sum(arr.size for arr in xpos_list)
        print(f"  Total data points: {total_pts}")

        xpos = np.empty(total_pts, dtype=xpos_list[0].dtype)
        ypos = np.empty(total_pts, dtype=ypos_list[0].dtype)
        zpos = np.empty(total_pts, dtype=zpos_list[0].dtype)
        velx = np.empty(total_pts, dtype=velx_list[0].dtype)
        vely = np.empty(total_pts, dtype=vely_list[0].dtype)
        velz = np.empty(total_pts, dtype=velz_list[0].dtype)

        # Copy each chunk into its slice of the flat arrays
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

        # Release the chunk lists to free memory
        del xpos_list, ypos_list, zpos_list
        del velx_list, vely_list, velz_list

    # Determine grid size
    x_unique = np.unique(xpos)
    y_unique = np.unique(ypos)
    z_unique = np.unique(zpos)
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

    print(f"  Grid dimensions: {nx} × {ny} × {nz}")

    # Compute grid spacing
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
    print(f"  Grid spacing: dx={dx:.8f}, dy={dy:.8f}, dz={dz:.8f}")

    # Validate data size
    expected_num_points = nx * ny * nz
    if total_pts != expected_num_points:
        print(f"  Warning: Actual points ({total_pts}) != expected ({expected_num_points})")

    # Reconstruct velocity grids
    print("  Reconstructing velocity grids...")
    velx_grid = np.zeros((nx, ny, nz))
    vely_grid = np.zeros((nx, ny, nz))
    velz_grid = np.zeros((nx, ny, nz))

    # Create coordinate mappings
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}

    for i in range(total_pts):
        xi = x_idx[xpos[i]]
        yi = y_idx[ypos[i]]
        zi = z_idx[zpos[i]]
        velx_grid[xi, yi, zi] = velx[i]
        vely_grid[xi, yi, zi] = vely[i]
        velz_grid[xi, yi, zi] = velz[i]

    # Calculate statistics
    total_ke = 0.5 * np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    v_mag = np.sqrt(velx_grid**2 + vely_grid**2 + velz_grid**2)

    print(f"  Total kinetic energy: {total_ke:.8f}")
    print(f"  Velocity component ranges:")
    print(f"    vx: [{velx_grid.min():.8f}, {velx_grid.max():.8f}]")
    print(f"    vy: [{vely_grid.min():.8f}, {vely_grid.max():.8f}]")
    print(f"    vz: [{velz_grid.min():.8f}, {velz_grid.max():.8f}]")
    print(f"    |v|: [{v_mag.min():.8f}, {v_mag.max():.8f}]")

    # --- PERIODICITY FIX ---
    print("  Applying periodic slicing (dropping last point)...")
    return (velx_grid[:-1, :-1, :-1],
            vely_grid[:-1, :-1, :-1],
            velz_grid[:-1, :-1, :-1],
            x_unique[:-1],
            y_unique[:-1],
            z_unique[:-1],
            dx, dy, dz)


# ------------------------------------------------------------------ #
#  Step 3: Create wavenumber grids for FFT operations
# ------------------------------------------------------------------ #
def create_wavenumber_grids(nx, ny, nz, dx, dy, dz):
    """Create wavenumber grids for Fourier transforms"""
    print("Creating wavenumber grids...")

    kx_phys = 2 * np.pi * fft.fftfreq(nx, d=dx)
    ky_phys = 2 * np.pi * fft.fftfreq(ny, d=dy)
    kz_phys = 2 * np.pi * fft.fftfreq(nz, d=dz)

    KX, KY, KZ = np.meshgrid(kx_phys, ky_phys, kz_phys, indexing='ij')
    K_squared = KX**2 + KY**2 + KZ**2
    nonzero_mask = K_squared != 0

    print(f"  Wavenumber range: kx=[{kx_phys.min():.3f}, {kx_phys.max():.3f}]")
    return KX, KY, KZ, K_squared, nonzero_mask


# ------------------------------------------------------------------ #
#  Step 4: Compute divergence and curl
# ------------------------------------------------------------------ #
def compute_divergence(vx, vy, vz, KX, KY, KZ):
    """Compute divergence of velocity field using FFT"""
    vx_k = fft.fftn(vx)
    vy_k = fft.fftn(vy)
    vz_k = fft.fftn(vz)

    div_k = 1j * (KX * vx_k + KY * vy_k + KZ * vz_k)
    divergence = np.real(fft.ifftn(div_k))
    return divergence


def compute_curl(vx, vy, vz, KX, KY, KZ):
    """Compute curl of velocity field using FFT"""
    vx_k = fft.fftn(vx)
    vy_k = fft.fftn(vy)
    vz_k = fft.fftn(vz)

    curl_x_k = 1j * (KY * vz_k - KZ * vy_k)
    curl_y_k = 1j * (KZ * vx_k - KX * vz_k)
    curl_z_k = 1j * (KX * vy_k - KY * vx_k)

    curl_x = np.real(fft.ifftn(curl_x_k))
    curl_y = np.real(fft.ifftn(curl_y_k))
    curl_z = np.real(fft.ifftn(curl_z_k))
    return curl_x, curl_y, curl_z


# ------------------------------------------------------------------ #
#  Step 5: Helmholtz-Hodge decomposition
# ------------------------------------------------------------------ #
def compute_compressive_part(vx, vy, vz, KX, KY, KZ, K_squared, nonzero_mask):
    """Compute compressive (irrotational) part: v_c = -??"""
    print("Computing compressive component...")

    divergence = compute_divergence(vx, vy, vz, KX, KY, KZ)

    div_k = fft.fftn(divergence)
    phi_k = np.zeros_like(div_k, dtype=complex)
    phi_k[nonzero_mask] = div_k[nonzero_mask] / K_squared[nonzero_mask]

    vx_c_k = -1j * KX * phi_k
    vy_c_k = -1j * KY * phi_k
    vz_c_k = -1j * KZ * phi_k

    vx_c = np.real(fft.ifftn(vx_c_k))
    vy_c = np.real(fft.ifftn(vy_c_k))
    vz_c = np.real(fft.ifftn(vz_c_k))

    ke_comp = 0.5 * np.mean(vx_c**2 + vy_c**2 + vz_c**2)
    print(f"  Compressive KE: {ke_comp:.8f}")
    return vx_c, vy_c, vz_c


def compute_rotational_part(vx, vy, vz, KX, KY, KZ, K_squared, nonzero_mask):
    """Compute rotational (solenoidal) part: v_r = ?×A"""
    print("Computing rotational component...")

    curl_x, curl_y, curl_z = compute_curl(vx, vy, vz, KX, KY, KZ)

    curl_x_k = fft.fftn(curl_x)
    curl_y_k = fft.fftn(curl_y)
    curl_z_k = fft.fftn(curl_z)

    Ax_k = np.zeros_like(curl_x_k, dtype=complex)
    Ay_k = np.zeros_like(curl_y_k, dtype=complex)
    Az_k = np.zeros_like(curl_z_k, dtype=complex)

    Ax_k[nonzero_mask] = curl_x_k[nonzero_mask] / K_squared[nonzero_mask]
    Ay_k[nonzero_mask] = curl_y_k[nonzero_mask] / K_squared[nonzero_mask]
    Az_k[nonzero_mask] = curl_z_k[nonzero_mask] / K_squared[nonzero_mask]

    vx_r_k = 1j * (KY * Az_k - KZ * Ay_k)
    vy_r_k = 1j * (KZ * Ax_k - KX * Az_k)
    vz_r_k = 1j * (KX * Ay_k - KY * Ax_k)

    vx_r = np.real(fft.ifftn(vx_r_k))
    vy_r = np.real(fft.ifftn(vy_r_k))
    vz_r = np.real(fft.ifftn(vz_r_k))

    Ax_r = np.real(fft.ifftn(Ax_k))
    Ay_r = np.real(fft.ifftn(Ay_k))
    Az_r = np.real(fft.ifftn(Az_k))

    div_A_r = compute_divergence(Ax_r, Ay_r, Az_r, KX, KY, KZ)
    max_div_A_r = np.abs(div_A_r).max()
    print(f"  Max |?·A_r|:  {max_div_A_r:.2e} (should be ~0)")

    ke_rot = 0.5 * np.mean(vx_r**2 + vy_r**2 + vz_r**2)
    print(f"  Rotational KE: {ke_rot:.8f}")
    return vx_r, vy_r, vz_r


def print_decomposition_statistics(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r):
    v_c_mag = np.sqrt(vx_c**2 + vy_c**2 + vz_c**2)
    v_r_mag = np.sqrt(vx_r**2 + vy_r**2 + vz_r**2)

    print("Decomposition component ranges:")
    print("  Compressive component:")
    print(f"    vx_c: [{vx_c.min():.8f}, {vx_c.max():.8f}]")
    print(f"    vy_c: [{vy_c.min():.8f}, {vy_c.max():.8f}]")
    print(f"    vz_c: [{vz_c.min():.8f}, {vz_c.max():.8f}]")
    print(f"    |v_c|: [{v_c_mag.min():.8f}, {v_c_mag.max():.8f}]")
    print("  Rotational component:")
    print(f"    vx_r: [{vx_r.min():.8f}, {vx_r.max():.8f}]")
    print(f"    vy_r: [{vy_r.min():.8f}, {vy_r.max():.8f}]")
    print(f"    vz_r: [{vz_r.min():.8f}, {vz_r.max():.8f}]")
    print(f"    |v_r|: [{v_r_mag.min():.8f}, {v_r_mag.max():.8f}]")


def verify_decomposition(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r, KX, KY, KZ):
    print("Verifying decomposition quality...")

    curl_c_x, curl_c_y, curl_c_z = compute_curl(vx_c, vy_c, vz_c, KX, KY, KZ)
    curl_c_mag = np.sqrt(curl_c_x**2 + curl_c_y**2 + curl_c_z**2)
    max_curl_c = curl_c_mag.max()

    div_r = compute_divergence(vx_r, vy_r, vz_r, KX, KY, KZ)
    max_div_r = np.abs(div_r).max()

    print(f"  Max |?×v_compressive|: {max_curl_c:.2e} (should be ~0)")
    print(f"  Max |?·v_rotational|:  {max_div_r:.2e} (should be ~0)")


# ------------------------------------------------------------------ #
#  Step 6: Compute energy spectra - integer wavenumber binning
# ------------------------------------------------------------------ #
def compute_energy_spectrum(vx, vy, vz, nx, ny, nz, dx, dy, dz):
    """Compute energy spectrum E(k) - FIXED to match library approach"""

    kx_int = np.fft.fftfreq(nx, 1./nx).astype(int)
    ky_int = np.fft.fftfreq(ny, 1./ny).astype(int)
    kz_int = np.fft.fftfreq(nz, 1./nz).astype(int)

    vx_k = np.fft.fftn(vx)
    vy_k = np.fft.fftn(vy)
    vz_k = np.fft.fftn(vz)

    norm = nx * ny * nz
    vx_k /= norm
    vy_k /= norm
    vz_k /= norm

    energy_density = 0.5 * (np.abs(vx_k)**2 + np.abs(vy_k)**2 + np.abs(vz_k)**2)
    print(f"  Total kinetic energy (fourier): {np.sum(energy_density):.8f}")

    KX_int, KY_int, KZ_int = np.meshgrid(kx_int, ky_int, kz_int, indexing='ij')
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)

    from math import ceil
    k_max_int = ceil(nx * 0.5 * np.sqrt(3.0))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)

    if nx * 0.5 * np.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]

    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    k_flat = k_magnitude.flatten()
    energy_flat = energy_density.flatten()

    E_k, _ = np.histogram(k_flat, bins=k_bin_edges, weights=energy_flat)
    return k_bin_centers, E_k


def compute_energy_dissipation_enstophy(vx, vy, vz, nx, ny, nz, dx, dy, dz):
    """Compute enstrophy and dissipation (without the nu) - FIXED"""
    kx_int = np.fft.fftfreq(nx, 1./nx).astype(int)
    ky_int = np.fft.fftfreq(ny, 1./ny).astype(int)
    kz_int = np.fft.fftfreq(nz, 1./nz).astype(int)

    vx_k = np.fft.fftn(vx)
    vy_k = np.fft.fftn(vy)
    vz_k = np.fft.fftn(vz)

    norm = nx * ny * nz
    vx_k /= norm
    vy_k /= norm
    vz_k /= norm

    energy_density = 0.5 * (np.abs(vx_k)**2 + np.abs(vy_k)**2 + np.abs(vz_k)**2)
    print(f"  Total kinetic energy (fourier): {np.sum(energy_density):.8f}")

    KX_int, KY_int, KZ_int = np.meshgrid(kx_int, ky_int, kz_int, indexing='ij')
    k_squared = KX_int**2 + KY_int**2 + KZ_int**2

    k_phys_squared = (2*np.pi)**2 * k_squared
    total_energy_dissipation = np.sum(energy_density * k_phys_squared)
    print(f"  Total dissipative energy: {total_energy_dissipation:.8f}")

    omega_x_k = 1j * 2*np.pi * (KY_int*vz_k - KZ_int*vy_k)
    omega_y_k = 1j * 2*np.pi * (KZ_int*vx_k - KX_int*vz_k)
    omega_z_k = 1j * 2*np.pi * (KX_int*vy_k - KY_int*vx_k)

    enstrophy_fourier = 0.5*np.sum(np.abs(omega_x_k)**2 + np.abs(omega_y_k)**2 + np.abs(omega_z_k)**2)
    print("enstrophy vs total dissipation comparison (should be close)")
    print(enstrophy_fourier, total_energy_dissipation)


# ------------------------------------------------------------------ #
#  Step 7: Visualization
# ------------------------------------------------------------------ #
def plot_velocity_slice(x_coords, y_coords, z_coords, vx, vy, vz,
                        vx_c, vy_c, vz_c, vx_r, vy_r, vz_r,
                        slice_z, step_number, time_value):
    """Plot 2D slices of velocity fields on two orthogonal planes (XY and XZ)"""
    print(f"Creating visualization for XY and XZ planes...")

    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    slice_y = ny // 2
    if slice_z is None:
        slice_z = nz // 2

    print(f"  Slice indices: y={slice_y}/{ny}, z={slice_z}/{nz}")

    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    skip = max(max(nx, ny, nz) // 16, 1)

    # XY plane
    X_xy = X[:, :, slice_z]
    Y_xy = Y[:, :, slice_z]

    vx_xy = vx[:, :, slice_z]
    vy_xy = vy[:, :, slice_z]
    vz_xy = vz[:, :, slice_z]

    vx_c_xy = vx_c[:, :, slice_z]
    vy_c_xy = vy_c[:, :, slice_z]
    vz_c_xy = vz_c[:, :, slice_z]

    vx_r_xy = vx_r[:, :, slice_z]
    vy_r_xy = vy_r[:, :, slice_z]
    vz_r_xy = vz_r[:, :, slice_z]

    v_mag_xy = np.sqrt(vx_xy**2 + vy_xy**2 + vz_xy**2)
    v_c_mag_xy = np.sqrt(vx_c_xy**2 + vy_c_xy**2 + vz_c_xy**2)
    v_r_mag_xy = np.sqrt(vx_r_xy**2 + vy_r_xy**2 + vz_r_xy**2)

    ax = axes[0, 0]
    im = ax.contourf(X_xy, Y_xy, v_mag_xy, levels=20, cmap='viridis')
    ax.quiver(X_xy[::skip, ::skip], Y_xy[::skip, ::skip],
              vx_xy[::skip, ::skip], vy_xy[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='white')
    ax.set_title(f'Total Velocity (XY plane, z={z_coords[slice_z]:.3f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|v|')

    ax = axes[0, 1]
    im = ax.contourf(X_xy, Y_xy, v_c_mag_xy, levels=20, cmap='Blues')
    ax.quiver(X_xy[::skip, ::skip], Y_xy[::skip, ::skip],
              vx_c_xy[::skip, ::skip], vy_c_xy[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='darkblue')
    ax.set_title(f'Compressive (XY plane, z={z_coords[slice_z]:.3f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|v_c|')

    ax = axes[0, 2]
    im = ax.contourf(X_xy, Y_xy, v_r_mag_xy, levels=20, cmap='Reds')
    ax.quiver(X_xy[::skip, ::skip], Y_xy[::skip, ::skip],
              vx_r_xy[::skip, ::skip], vy_r_xy[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='darkred')
    ax.set_title(f'Rotational (XY plane, z={z_coords[slice_z]:.3f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|v_r|')

    # XZ plane
    X_xz = X[:, slice_y, :]
    Z_xz = Z[:, slice_y, :]

    vx_xz = vx[:, slice_y, :]
    vz_xz = vz[:, slice_y, :]

    vx_c_xz = vx_c[:, slice_y, :]
    vz_c_xz = vz_c[:, slice_y, :]

    vx_r_xz = vx_r[:, slice_y, :]
    vz_r_xz = vz_r[:, slice_y, :]

    v_mag_xz = np.sqrt(vx[:, slice_y, :]**2 + vy[:, slice_y, :]**2 + vz[:, slice_y, :]**2)
    v_c_mag_xz = np.sqrt(vx_c[:, slice_y, :]**2 + vy_c[:, slice_y, :]**2 + vz_c[:, slice_y, :]**2)
    v_r_mag_xz = np.sqrt(vx_r[:, slice_y, :]**2 + vy_r[:, slice_y, :]**2 + vz_r[:, slice_y, :]**2)

    ax = axes[1, 0]
    im = ax.contourf(X_xz, Z_xz, v_mag_xz, levels=20, cmap='viridis')
    ax.quiver(X_xz[::skip, ::skip], Z_xz[::skip, ::skip],
              vx_xz[::skip, ::skip], vz_xz[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='white')
    ax.set_title(f'Total Velocity (XZ plane, y={y_coords[slice_y]:.3f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|v|')

    ax = axes[1, 1]
    im = ax.contourf(X_xz, Z_xz, v_c_mag_xz, levels=20, cmap='Blues')
    ax.quiver(X_xz[::skip, ::skip], Z_xz[::skip, ::skip],
              vx_c_xz[::skip, ::skip], vz_c_xz[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='darkblue')
    ax.set_title(f'Compressive (XZ plane, y={y_coords[slice_y]:.3f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|v_c|')

    ax = axes[1, 2]
    im = ax.contourf(X_xz, Z_xz, v_r_mag_xz, levels=20, cmap='Reds')
    ax.quiver(X_xz[::skip, ::skip], Z_xz[::skip, ::skip],
              vx_r_xz[::skip, ::skip], vz_r_xz[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='darkred')
    ax.set_title(f'Rotational (XZ plane, y={y_coords[slice_y]:.3f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|v_r|')

    plt.suptitle(f'Velocity Field Decomposition (XY & XZ Planes) - Step {step_number}, Time {time_value:.3e}',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Step 8: Save results
# ------------------------------------------------------------------ #
def save_spectra(k_centers, E_total, E_comp, E_rot, filename, step_number, time_value,
                 nx, ny, nz, total_ke, comp_ke, rot_ke):
    """Save all spectra to a single file, including compensated spectra"""

    E_sum = E_comp + E_rot
    k_power = np.power(k_centers, 5.0/3.0)

    E_total_compensated = E_total * k_power
    E_comp_compensated = E_comp * k_power
    E_rot_compensated = E_rot * k_power
    E_sum_compensated = E_sum * k_power

    output_filename = os.path.join(os.path.dirname(filename),
                                   f'energy_spectrum_step_{step_number}.txt')

    with open(output_filename, 'w') as f:
        f.write(f"# Energy Spectra for Step {step_number}, Time {time_value:.6e}\n")
        f.write(f"# Domain: [0,1]³, Grid: {nx}x{ny}x{nz}\n")
        f.write(f"# Total KE: {total_ke:.8f}, Compressive KE: {comp_ke:.8f}, Rotational KE: {rot_ke:.8f}\n")
        f.write("# Columns: wavenumber, E_total, E_compressive, E_rotational, E_sum, "
                "E_total*k^(5/3), E_compressive*k^(5/3), E_rotational*k^(5/3), E_sum*k^(5/3)\n")
        f.write("# wavenumber,E_total,E_compressive,E_rotational,E_sum,"
                "E_total_compensated,E_compressive_compensated,E_rotational_compensated,E_sum_compensated\n")

        for k, e_tot, e_comp, e_rot, e_sum, e_tot_c, e_comp_c, e_rot_c, e_sum_c in zip(
                k_centers, E_total, E_comp, E_rot, E_sum,
                E_total_compensated, E_comp_compensated, E_rot_compensated, E_sum_compensated):
            f.write(f"{k:.6e},{e_tot:.6e},{e_comp:.6e},{e_rot:.6e},{e_sum:.6e},"
                    f"{e_tot_c:.6e},{e_comp_c:.6e},{e_rot_c:.6e},{e_sum_c:.6e}\n")

    print(f"Saved library-matched spectra to: {output_filename}")
    return output_filename


# ------------------------------------------------------------------ #
#  Main function - puts it all together
# ------------------------------------------------------------------ #
def analyze_file(filename, header_lines=None, visualize=False, slice_z=None, chunk_size=5_000_000):
    """Main analysis function - step by step with MINIMAL FIX"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filename}")
    print(f"{'='*60}")

    # Determine header lines if not provided
    if filename.endswith('.h5'):
        header_lines = 0
    elif header_lines is None:
        header_lines = detect_header_lines(filename)

    # Step 1: Read header
    step_number, time_value = read_data_file_header(filename, header_lines)

    # Step 2: Read data
    grid_vx, grid_vy, grid_vz, x_unique, y_unique, z_unique, dx, dy, dz = read_data_file_chunked(
        filename, chunk_size=chunk_size, skiprows=header_lines
    )

    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    total_ke = 0.5 * np.mean(grid_vx**2 + grid_vy**2 + grid_vz**2)

    # Step 3: Create wavenumber grids
    KX, KY, KZ, K_squared, nonzero_mask = create_wavenumber_grids(nx, ny, nz, dx, dy, dz)

    # Step 4: Helmholtz-Hodge decomposition
    print("Performing Helmholtz-Hodge decomposition...")
    vx_c, vy_c, vz_c = compute_compressive_part(grid_vx, grid_vy, grid_vz, KX, KY, KZ, K_squared, nonzero_mask)
    vx_r, vy_r, vz_r = compute_rotational_part(grid_vx, grid_vy, grid_vz, KX, KY, KZ, K_squared, nonzero_mask)

    # Energies
    comp_ke = 0.5 * np.mean(vx_c**2 + vy_c**2 + vz_c**2)
    rot_ke = 0.5 * np.mean(vx_r**2 + vy_r**2 + vz_r**2)

    print(f"Energy breakdown:")
    print(f"  Total: {total_ke:.8f}")
    print(f"  Compressive: {comp_ke:.8f} ({100*comp_ke/total_ke:.1f}%)")
    print(f"  Rotational: {rot_ke:.8f} ({100*rot_ke/total_ke:.1f}%)")
    print(f"  Sum: {comp_ke + rot_ke:.8f}")

    print_decomposition_statistics(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r)

    # Step 5: Verify decomposition
    verify_decomposition(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r, KX, KY, KZ)

    # Step 6: Spectra
    print("Computing energy spectra with library-matched integer wavenumber approach...")
    k_centers, E_total = compute_energy_spectrum(grid_vx, grid_vy, grid_vz, nx, ny, nz, dx, dy, dz)
    _, E_comp = compute_energy_spectrum(vx_c, vy_c, vz_c, nx, ny, nz, dx, dy, dz)
    _, E_rot = compute_energy_spectrum(vx_r, vy_r, vz_r, nx, ny, nz, dx, dy, dz)

    compute_energy_dissipation_enstophy(grid_vx, grid_vy, grid_vz, nx, ny, nz, dx, dy, dz)

    # Step 7: Save
    save_spectra(k_centers, E_total, E_comp, E_rot, filename, step_number, time_value,
                 nx, ny, nz, total_ke, comp_ke, rot_ke)

    # Step 8: Visualization (optional)
    if visualize:
        viz_slice_z = slice_z if slice_z is not None else nz // 2
        if viz_slice_z >= nz:
            viz_slice_z = nz - 1
        plot_velocity_slice(x_unique, y_unique, z_unique, grid_vx, grid_vy, grid_vz,
                            vx_c, vy_c, vz_c, vx_r, vy_r, vz_r,
                            viz_slice_z, step_number, time_value)

    return k_centers, E_total, E_comp, E_rot, step_number, time_value


def plot_spectra(results_list):
    """Plot energy spectra from multiple files"""
    print("\nPlotting library-matched energy spectra...")

    plt.figure(figsize=(12, 8))

    for k_centers, E_total, E_comp, E_rot, step_number, time_value in results_list:
        E_sum = E_comp + E_rot
        label_base = f"Step {step_number}, t={time_value:.3e}"

        plt.loglog(k_centers, E_total, '-', linewidth=2, label=f'{label_base} (Total)')
        plt.loglog(k_centers, E_comp, '--', linewidth=2, label=f'{label_base} (Compressive)')
        plt.loglog(k_centers, E_rot, ':', linewidth=2, label=f'{label_base} (Rotational)')
        plt.loglog(k_centers, E_sum, '-.', linewidth=1, alpha=0.7, label=f'{label_base} (Sum)')

    k_nonzero = k_centers[k_centers > 0]
    if len(k_nonzero) > 0:
        E_ref = 1e-2
        E_line = E_ref * (k_nonzero / 1.0)**(-5.0/3.0)
        plt.loglog(k_nonzero, E_line, 'k--', alpha=0.7, label='k^(-5/3) slope')

    plt.xlabel('Wavenumber k', fontsize=12)
    plt.ylabel('E(k)', fontsize=12)
    plt.title('Energy Spectra: Fixed Integer Wavenumber Binning to Match Library', fontsize=14)
    plt.ylim(1e-9, 1e-1)
    plt.legend(loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Command line interface
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simple Helmholtz-Hodge decomposition with minimal fix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_script.py data_file.txt --header-lines 5
  python simple_script.py data_file.h5 --header-lines 0 --no-plot
  python simple_script.py data_file.txt --header-lines 6 --no-plot
  python simple_script.py file1.txt file2.txt --header-lines 5
  python simple_script.py *.txt --header-lines 6 --no-plot
  python simple_script.py data_file.txt --header-lines 5 --visualize
        """
    )

    parser.add_argument('data_files', type=str, nargs='+',
                        help='One or more velocity data files to analyze')

    # OPTIONAL (Auto-detected if not provided):
    parser.add_argument('--header-lines', type=int, default=None,
                        help='Number of header lines to skip/read. If omitted, attempts auto-detection.')

    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Show velocity field visualization for each file')
    parser.add_argument('--slice_z', '-s', type=int, default=None,
                        help='Z-slice for visualization (default: middle slice)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting spectra (useful for batch/SLURM jobs)')

    args = parser.parse_args()

    if args.no_plot:
        matplotlib.use('Agg')
        print("Running in batch mode (no plots will be displayed)")

    results = []
    for filename in args.data_files:
        result = analyze_file(filename,
                              header_lines=args.header_lines,
                              visualize=args.visualize,
                              slice_z=args.slice_z)
        results.append(result)

    if not args.no_plot:
        plot_spectra(results)
    else:
        print(f"\nProcessed {len(results)} files. Spectrum files saved to disk.")
        print("Skipping plot display as requested (--no-plot flag).")

