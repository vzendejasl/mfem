#!/usr/bin/env python3
"""
Simple Helmholtz-Hodge decomposition and energy spectrum analysis
Easy to follow step-by-step implementation

Usage:
    python simple_script.py data_file.txt                    # Basic analysis
    python simple_script.py data_file.txt --visualize       # With visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import re
import os 
import argparse


# ------------------------------------------------------------------ #
#  Step 1: Read and parse data file
# ------------------------------------------------------------------ #
def read_data_file(filename):
    """Read velocity data from file and extract metadata"""
    print(f"Reading data from: {filename}")
    
    # Read header lines
    with open(filename, 'r') as f:
        header_lines = [next(f) for _ in range(6)]
    
    # Extract step number and time
    step_number = "unknown"
    time_value = 0.0
    
    for line in header_lines:
        if 'Cycle' in line:
            match = re.search(r'Cycle\s*[:=]\s*(\d+)', line)
            if match:
                step_number = match.group(1)
        if 'Time' in line:
            match = re.search(r'Time\s*[:=]\s*([0-9.eE+-]+)', line)
            if match:
                time_value = float(match.group(1))
    
    # Load velocity data (skip header)
    data = np.genfromtxt(filename, delimiter=' ', skip_header=6)
    
    # Extract coordinates and velocities
    x_coords = data[:, 0]
    y_coords = data[:, 1] 
    z_coords = data[:, 2]
    vel_x = data[:, 3]
    vel_y = data[:, 4]
    vel_z = data[:, 5]
    
    print(f"  Step: {step_number}, Time: {time_value:.3e}")
    print(f"  Data points: {len(x_coords)}")
    
    return x_coords, y_coords, z_coords, vel_x, vel_y, vel_z, step_number, time_value


# ------------------------------------------------------------------ #
#  Step 2: Convert scattered data to regular grid
# ------------------------------------------------------------------ #
def create_velocity_grid(x_coords, y_coords, z_coords, vel_x, vel_y, vel_z):
    """Convert scattered velocity data to regular 3D grid"""
    print("Creating regular velocity grid...")
    
    # Round coordinates to avoid floating point issues
    x_rounded = np.round(x_coords, decimals=10)
    y_rounded = np.round(y_coords, decimals=10)
    z_rounded = np.round(z_coords, decimals=10)
    
    # Find unique coordinates
    x_unique = np.unique(x_rounded)
    y_unique = np.unique(y_rounded)
    z_unique = np.unique(z_rounded)
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    print(f"  Grid dimensions: {nx} × {ny} × {nz}")
    
    # Compute grid spacing
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
    print(f"  Grid spacing: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
    
    # Create empty grids
    grid_vx = np.zeros((nx, ny, nz))
    grid_vy = np.zeros((nx, ny, nz))
    grid_vz = np.zeros((nx, ny, nz))
    
    # Create coordinate mappings
    x_to_i = {val: i for i, val in enumerate(x_unique)}
    y_to_j = {val: j for j, val in enumerate(y_unique)}
    z_to_k = {val: k for k, val in enumerate(z_unique)}
    
    # Fill grids
    for n in range(len(x_coords)):
        i = x_to_i[x_rounded[n]]
        j = y_to_j[y_rounded[n]]
        k = z_to_k[z_rounded[n]]
        grid_vx[i, j, k] = vel_x[n]
        grid_vy[i, j, k] = vel_y[n]
        grid_vz[i, j, k] = vel_z[n]
    
    # Calculate total kinetic energy
    total_ke = 0.5 * np.mean(grid_vx**2 + grid_vy**2 + grid_vz**2)
    print(f"  Total kinetic energy: {total_ke:.6f}")
    
    # Calculate velocity magnitude for the entire 3D field
    v_mag = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)
    
    # Print velocity component statistics
    print(f"  Original velocity component ranges:")
    print(f"    vx: [{grid_vx.min():.6f}, {grid_vx.max():.6f}]")
    print(f"    vy: [{grid_vy.min():.6f}, {grid_vy.max():.6f}]")
    print(f"    vz: [{grid_vz.min():.6f}, {grid_vz.max():.6f}]")
    print(f"    |v|: [{v_mag.min():.6f}, {v_mag.max():.6f}]")
    
    return grid_vx, grid_vy, grid_vz, x_unique, y_unique, z_unique, dx, dy, dz


# ------------------------------------------------------------------ #
#  Step 3: Create wavenumber grids for FFT operations
# ------------------------------------------------------------------ #
def create_wavenumber_grids(nx, ny, nz, dx, dy, dz):
    """Create wavenumber grids for Fourier transforms"""
    print("Creating wavenumber grids...")
    
    # Physical wavenumbers for Helmholtz-Hodge decomposition
    kx_phys = 2 * np.pi * fft.fftfreq(nx, d=dx)
    ky_phys = 2 * np.pi * fft.fftfreq(ny, d=dy)
    kz_phys = 2 * np.pi * fft.fftfreq(nz, d=dz)
    
    # Create 3D wavenumber grids
    KX, KY, KZ = np.meshgrid(kx_phys, ky_phys, kz_phys, indexing='ij')
    K_squared = KX**2 + KY**2 + KZ**2
    
    # Avoid division by zero at k=0
    nonzero_mask = K_squared != 0
    
    print(f"  Wavenumber range: kx=[{kx_phys.min():.3f}, {kx_phys.max():.3f}]")
    
    return KX, KY, KZ, K_squared, nonzero_mask


# ------------------------------------------------------------------ #
#  Step 4: Compute divergence and curl
# ------------------------------------------------------------------ #
def compute_divergence(vx, vy, vz, KX, KY, KZ):
    """Compute divergence of velocity field using FFT"""
    # Transform to Fourier space
    vx_k = fft.fftn(vx)
    vy_k = fft.fftn(vy)
    vz_k = fft.fftn(vz)
    
    # Compute divergence in Fourier space: div = i*k·v
    div_k = 1j * (KX * vx_k + KY * vy_k + KZ * vz_k)
    
    # Transform back to physical space
    divergence = np.real(fft.ifftn(div_k))
    
    return divergence


def compute_curl(vx, vy, vz, KX, KY, KZ):
    """Compute curl of velocity field using FFT"""
    # Transform to Fourier space
    vx_k = fft.fftn(vx)
    vy_k = fft.fftn(vy)
    vz_k = fft.fftn(vz)
    
    # Compute curl in Fourier space: curl = i*k×v
    curl_x_k = 1j * (KY * vz_k - KZ * vy_k)
    curl_y_k = 1j * (KZ * vx_k - KX * vz_k)
    curl_z_k = 1j * (KX * vy_k - KY * vx_k)
    
    # Transform back to physical space
    curl_x = np.real(fft.ifftn(curl_x_k))
    curl_y = np.real(fft.ifftn(curl_y_k))
    curl_z = np.real(fft.ifftn(curl_z_k))
    
    return curl_x, curl_y, curl_z


# ------------------------------------------------------------------ #
#  Step 5: Helmholtz-Hodge decomposition
# ------------------------------------------------------------------ #
def compute_compressive_part(vx, vy, vz, KX, KY, KZ, K_squared, nonzero_mask):
    """Compute compressive (irrotational) part: v_c = -∇φ"""
    print("Computing compressive component...")
    
    # Step 1: Compute divergence
    divergence = compute_divergence(vx, vy, vz, KX, KY, KZ)
    
    # Step 2: Solve for scalar potential φ: ∇²φ = -div
    div_k = fft.fftn(divergence)
    phi_k = np.zeros_like(div_k, dtype=complex)
    phi_k[nonzero_mask] = div_k[nonzero_mask] / K_squared[nonzero_mask]
    
    # Step 3: Compute compressive velocity: v_c = -∇φ
    vx_c_k = -1j * KX * phi_k
    vy_c_k = -1j * KY * phi_k
    vz_c_k = -1j * KZ * phi_k
    
    # Transform back to physical space
    vx_c = np.real(fft.ifftn(vx_c_k))
    vy_c = np.real(fft.ifftn(vy_c_k))
    vz_c = np.real(fft.ifftn(vz_c_k))
    
    # Calculate kinetic energy
    ke_comp = 0.5 * np.mean(vx_c**2 + vy_c**2 + vz_c**2)
    print(f"  Compressive KE: {ke_comp:.6f}")
    
    return vx_c, vy_c, vz_c


def compute_rotational_part(vx, vy, vz, KX, KY, KZ, K_squared, nonzero_mask):
    """Compute rotational (solenoidal) part: v_r = ∇×A"""
    print("Computing rotational component...")
    
    # Step 1: Compute curl
    curl_x, curl_y, curl_z = compute_curl(vx, vy, vz, KX, KY, KZ)
    
    # Step 2: Solve for vector potential A: ∇²A = -curl
    curl_x_k = fft.fftn(curl_x)
    curl_y_k = fft.fftn(curl_y)
    curl_z_k = fft.fftn(curl_z)
    
    Ax_k = np.zeros_like(curl_x_k, dtype=complex)
    Ay_k = np.zeros_like(curl_y_k, dtype=complex)
    Az_k = np.zeros_like(curl_z_k, dtype=complex)
    
    Ax_k[nonzero_mask] = curl_x_k[nonzero_mask] / K_squared[nonzero_mask]
    Ay_k[nonzero_mask] = curl_y_k[nonzero_mask] / K_squared[nonzero_mask]
    Az_k[nonzero_mask] = curl_z_k[nonzero_mask] / K_squared[nonzero_mask]
    
    # Step 3: Compute rotational velocity: v_r = ∇×A
    vx_r_k = 1j * (KY * Az_k - KZ * Ay_k)
    vy_r_k = 1j * (KZ * Ax_k - KX * Az_k)
    vz_r_k = 1j * (KX * Ay_k - KY * Ax_k)
    
    # Transform back to physical space
    vx_r = np.real(fft.ifftn(vx_r_k))
    vy_r = np.real(fft.ifftn(vy_r_k))
    vz_r = np.real(fft.ifftn(vz_r_k))

    Ax_r = np.real(fft.ifftn(Ax_k))
    Ay_r = np.real(fft.ifftn(Ay_k))
    Az_r = np.real(fft.ifftn(Az_k))

    # Check that rotational part is divergence-free
    div_A_r = compute_divergence(Ax_r, Ay_r, Az_r, KX, KY, KZ)
    max_div_A_r = np.abs(div_A_r).max()
    
    print(f"  Max |∇·A_r|:  {max_div_A_r:.2e} (should be ~0)")
    
    # Calculate kinetic energy
    ke_rot = 0.5 * np.mean(vx_r**2 + vy_r**2 + vz_r**2)
    print(f"  Rotational KE: {ke_rot:.6f}")
    
    return vx_r, vy_r, vz_r


def print_decomposition_statistics(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r):
    """Print min/max statistics for decomposition components"""
    # Calculate velocity magnitudes for the entire 3D fields
    v_c_mag = np.sqrt(vx_c**2 + vy_c**2 + vz_c**2)
    v_r_mag = np.sqrt(vx_r**2 + vy_r**2 + vz_r**2)
    
    print("Decomposition component ranges:")
    print("  Compressive component:")
    print(f"    vx_c: [{vx_c.min():.6f}, {vx_c.max():.6f}]")
    print(f"    vy_c: [{vy_c.min():.6f}, {vy_c.max():.6f}]")
    print(f"    vz_c: [{vz_c.min():.6f}, {vz_c.max():.6f}]")
    print(f"    |v_c|: [{v_c_mag.min():.6f}, {v_c_mag.max():.6f}]")
    print("  Rotational component:")
    print(f"    vx_r: [{vx_r.min():.6f}, {vx_r.max():.6f}]")
    print(f"    vy_r: [{vy_r.min():.6f}, {vy_r.max():.6f}]")
    print(f"    vz_r: [{vz_r.min():.6f}, {vz_r.max():.6f}]")
    print(f"    |v_r|: [{v_r_mag.min():.6f}, {v_r_mag.max():.6f}]")


def verify_decomposition(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r, KX, KY, KZ):
    """Verify that decomposition satisfies mathematical properties"""
    print("Verifying decomposition quality...")
    
    # Check that compressive part is curl-free
    curl_c_x, curl_c_y, curl_c_z = compute_curl(vx_c, vy_c, vz_c, KX, KY, KZ)
    curl_c_mag = np.sqrt(curl_c_x**2 + curl_c_y**2 + curl_c_z**2)
    max_curl_c = curl_c_mag.max()
    
    # Check that rotational part is divergence-free
    div_r = compute_divergence(vx_r, vy_r, vz_r, KX, KY, KZ)
    max_div_r = np.abs(div_r).max()
    
    print(f"  Max |∇×v_compressive|: {max_curl_c:.2e} (should be ~0)")
    print(f"  Max |∇·v_rotational|:  {max_div_r:.2e} (should be ~0)")


# ------------------------------------------------------------------ #
#  Step 6: Compute energy spectra
# ------------------------------------------------------------------ #
def compute_energy_spectrum(vx, vy, vz, nx, ny, nz, dx, dy, dz):
    """Compute energy spectrum E(k)"""
    # Create wavenumber grid for spectrum (different convention)
    kx_spec = np.fft.fftfreq(nx, d=dx/(1.0))
    ky_spec = np.fft.fftfreq(ny, d=dy/(1.0))
    kz_spec = np.fft.fftfreq(nz, d=dz/(1.0))
    
    # Transform to Fourier space
    vx_k = np.fft.fftshift(np.fft.fftn(vx))
    vy_k = np.fft.fftshift(np.fft.fftn(vy))
    vz_k = np.fft.fftshift(np.fft.fftn(vz))
    
    # Normalize
    norm = nx * ny * nz
    vx_k /= norm
    vy_k /= norm
    vz_k /= norm
    
    # Compute energy density
    energy_density = 0.5 * (np.abs(vx_k)**2 + np.abs(vy_k)**2 + np.abs(vz_k)**2)
    print(f"  Total kinetic energy (fourier): {np.sum(energy_density):.6f}")
    
    # Create wavenumber magnitude grid
    kx_shift = np.fft.fftshift(kx_spec)
    ky_shift = np.fft.fftshift(ky_spec)
    kz_shift = np.fft.fftshift(kz_spec)
    
    KX_spec, KY_spec, KZ_spec = np.meshgrid(kx_shift, ky_shift, kz_shift, indexing='ij')
    k_magnitude = np.sqrt(KX_spec**2 + KY_spec**2 + KZ_spec**2)
    k_squared = KX_spec**2 + KY_spec**2 + KZ_spec**2
    
    # Bin energy by wavenumber magnitude
    k_flat = k_magnitude.flatten()
    energy_flat = energy_density.flatten()

    # Create bins
    num_bins = nx
    k_bin_edges = np.arange(0, num_bins+1) - 0.5
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
    
    # Compute spectrum
    E_k, _ = np.histogram(k_flat, bins=k_bin_edges, weights=energy_flat)
    
    return k_bin_centers, E_k

# ------------------------------------------------------------------ #
#  Step 6 (Optional): Compute dissipation & enstrophy
# ------------------------------------------------------------------ #
def compute_energy_dissipation_enstophy(vx, vy, vz, nx, ny, nz, dx, dy, dz):
    """Compute enstrophy and dissipation (without the nu)"""
    # Create wavenumber grid for spectrum (different convention)
    kx_spec = np.fft.fftfreq(nx, d=dx/(1.0))
    ky_spec = np.fft.fftfreq(ny, d=dy/(1.0))
    kz_spec = np.fft.fftfreq(nz, d=dz/(1.0))
    
    # Transform to Fourier space
    vx_k = np.fft.fftshift(np.fft.fftn(vx))
    vy_k = np.fft.fftshift(np.fft.fftn(vy))
    vz_k = np.fft.fftshift(np.fft.fftn(vz))
    
    # Normalize
    norm = nx * ny * nz
    vx_k /= norm
    vy_k /= norm
    vz_k /= norm
    
    # Compute energy density
    energy_density = 0.5 * (np.abs(vx_k)**2 + np.abs(vy_k)**2 + np.abs(vz_k)**2)
    print(f"  Total kinetic energy (fourier): {np.sum(energy_density):.6f}")
    
    # Create wavenumber magnitude grid
    kx_shift = np.fft.fftshift(kx_spec)
    ky_shift = np.fft.fftshift(ky_spec)
    kz_shift = np.fft.fftshift(kz_spec)
    
    KX_spec, KY_spec, KZ_spec = np.meshgrid(kx_shift, ky_shift, kz_shift, indexing='ij')
    k_magnitude = np.sqrt(KX_spec**2 + KY_spec**2 + KZ_spec**2)
    k_squared = KX_spec**2 + KY_spec**2 + KZ_spec**2
    
    # Bin energy by wavenumber magnitude
    k_flat = k_magnitude.flatten()
    energy_flat = energy_density.flatten()

    # Compute \sum of k*k*E(k)
    # The factor of 2pi is there so convert from cycles per lengh to radians per lenght
    # We need this conversation to match phsyical space units
    total_energy_dissipation = np.sum((2*np.pi)**2*energy_density*k_squared)
    print(f"  Total dissipative energy: {total_energy_dissipation:.6f}")

    # compute vorticity in Fourier space with angular wavenumbers
    vx_k = np.fft.fftshift(np.fft.fftn(vx)) / (nx*ny*nz)
    vy_k = np.fft.fftshift(np.fft.fftn(vy)) / (nx*ny*nz)
    vz_k = np.fft.fftshift(np.fft.fftn(vz)) / (nx*ny*nz)
    omega_x_k = 1j*2*np.pi*(KY_spec*vz_k - KZ_spec*vy_k)
    omega_y_k = 1j*2*np.pi*(KZ_spec*vx_k - KX_spec*vz_k)
    omega_z_k = 1j*2*np.pi*(KX_spec*vy_k - KY_spec*vx_k)
    
    enstrophy_fourier = 0.5*np.sum(np.abs(omega_x_k)**2 + np.abs(omega_y_k)**2 + np.abs(omega_z_k)**2)
    print("enstrophy vs total dissipation comparision (should be close)")
    print(enstrophy_fourier, total_energy_dissipation)


# ------------------------------------------------------------------ #
#  Step 7: Visualization
# ------------------------------------------------------------------ #
def plot_velocity_slice(x_coords, y_coords, z_coords, vx, vy, vz, 
                       vx_c, vy_c, vz_c, vx_r, vy_r, vz_r,
                       slice_z, step_number, time_value):
    """Plot 2D slice of velocity fields"""
    print(f"Creating visualization for z-slice {slice_z}...")
    
    # Create coordinate grids
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Extract slice data
    X_slice = X[:, :, slice_z]
    Y_slice = Y[:, :, slice_z]
    
    # Velocity components on slice
    vx_slice = vx[:, :, slice_z]
    vy_slice = vy[:, :, slice_z]
    vx_c_slice = vx_c[:, :, slice_z]
    vy_c_slice = vy_c[:, :, slice_z]
    vx_r_slice = vx_r[:, :, slice_z]
    vy_r_slice = vy_r[:, :, slice_z]
    
    # Compute magnitudes
    v_mag = np.sqrt(vx_slice**2 + vy_slice**2 + vz[:, :, slice_z]**2)
    v_c_mag = np.sqrt(vx_c_slice**2 + vy_c_slice**2 + vz_c[:, :, slice_z]**2)
    v_r_mag = np.sqrt(vx_r_slice**2 + vy_r_slice**2 + vz_r[:, :, slice_z]**2)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Arrow subsampling
    skip = max(len(x_coords) // 16, 1)
    
    # Total field
    ax = axes[0]
    im1 = ax.contourf(X_slice, Y_slice, v_mag, levels=20, cmap='viridis')
    ax.quiver(X_slice[::skip, ::skip], Y_slice[::skip, ::skip],
              vx_slice[::skip, ::skip], vy_slice[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='white')
    ax.set_title(f'Total Velocity\n(z-slice {slice_z})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax, label='|v|')
    
    # Compressive field
    ax = axes[1]
    im2 = ax.contourf(X_slice, Y_slice, v_c_mag, levels=20, cmap='Blues')
    ax.quiver(X_slice[::skip, ::skip], Y_slice[::skip, ::skip],
              vx_c_slice[::skip, ::skip], vy_c_slice[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='darkblue')
    ax.set_title(f'Compressive\n(z-slice {slice_z})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, label='|v_c|')
    
    # Rotational field
    ax = axes[2]
    im3 = ax.contourf(X_slice, Y_slice, v_r_mag, levels=20, cmap='Reds')
    ax.quiver(X_slice[::skip, ::skip], Y_slice[::skip, ::skip],
              vx_r_slice[::skip, ::skip], vy_r_slice[::skip, ::skip],
              scale=None, scale_units='xy', angles='xy', alpha=0.7, color='darkred')
    ax.set_title(f'Rotational\n(z-slice {slice_z})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(im3, ax=ax, label='|v_r|')
    
    plt.suptitle(f'Velocity Field Decomposition - Step {step_number}, Time {time_value:.3e}', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Step 8: Save results
# ------------------------------------------------------------------ #
def save_spectra(k_centers, E_total, E_comp, E_rot, filename, step_number, time_value,
                 nx, ny, nz, total_ke, comp_ke, rot_ke):
    """Save all spectra to a single file"""
    E_sum = E_comp + E_rot
    
    output_filename = os.path.join(os.path.dirname(filename), 
                                   f'energy_spectrum_step_{step_number}.txt')
    
    with open(output_filename, 'w') as f:
        f.write(f"# Energy Spectra for Step {step_number}, Time {time_value:.6e}\n")
        f.write(f"# Domain: [0,1]³, Grid: {nx}x{ny}x{nz}\n")
        f.write(f"# Total KE: {total_ke:.6f}, Compressive KE: {comp_ke:.6f}, Rotational KE: {rot_ke:.6f}\n")
        f.write("# Columns: wavenumber, E_total, E_compressive, E_rotational, E_sum\n")
        f.write("# wavenumber,E_total,E_compressive,E_rotational,E_sum\n")
        
        for k, e_tot, e_comp, e_rot, e_sum in zip(k_centers, E_total, E_comp, E_rot, E_sum):
            f.write(f"{k:.6e},{e_tot:.6e},{e_comp:.6e},{e_rot:.6e},{e_sum:.6e}\n")
    
    print(f"Saved spectra to: {output_filename}")
    return output_filename


# ------------------------------------------------------------------ #
#  Main function - puts it all together
# ------------------------------------------------------------------ #
def analyze_file(filename, visualize=False, slice_z=None):
    """Main analysis function - step by step"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filename}")
    print(f"{'='*60}")
    
    # Step 1: Read data
    x_coords, y_coords, z_coords, vel_x, vel_y, vel_z, step_number, time_value = read_data_file(filename)
    
    # Step 2: Create regular grid
    grid_vx, grid_vy, grid_vz, x_unique, y_unique, z_unique, dx, dy, dz = create_velocity_grid(
        x_coords, y_coords, z_coords, vel_x, vel_y, vel_z)
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    total_ke = 0.5 * np.mean(grid_vx**2 + grid_vy**2 + grid_vz**2)
    
    # Step 3: Create wavenumber grids
    KX, KY, KZ, K_squared, nonzero_mask = create_wavenumber_grids(nx, ny, nz, dx, dy, dz)
    
    # Step 4: Helmholtz-Hodge decomposition
    print("Performing Helmholtz-Hodge decomposition...")
    vx_c, vy_c, vz_c = compute_compressive_part(grid_vx, grid_vy, grid_vz, KX, KY, KZ, K_squared, nonzero_mask)
    vx_r, vy_r, vz_r = compute_rotational_part(grid_vx, grid_vy, grid_vz, KX, KY, KZ, K_squared, nonzero_mask)
    
    # Calculate component energies
    comp_ke = 0.5 * np.mean(vx_c**2 + vy_c**2 + vz_c**2)
    rot_ke = 0.5 * np.mean(vx_r**2 + vy_r**2 + vz_r**2)
    
    print(f"Energy breakdown:")
    print(f"  Total: {total_ke:.6f}")
    print(f"  Compressive: {comp_ke:.6f} ({100*comp_ke/total_ke:.1f}%)")
    print(f"  Rotational: {rot_ke:.6f} ({100*rot_ke/total_ke:.1f}%)")
    print(f"  Sum: {comp_ke + rot_ke:.6f}")
    
    # Print component statistics
    print_decomposition_statistics(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r)
    
    # Step 5: Verify decomposition
    verify_decomposition(vx_c, vy_c, vz_c, vx_r, vy_r, vz_r, KX, KY, KZ)
    
    # Step 6: Compute energy spectra
    print("Computing energy spectra...")
    k_centers, E_total = compute_energy_spectrum(grid_vx, grid_vy, grid_vz, nx, ny, nz, dx, dy, dz)
    _, E_comp = compute_energy_spectrum(vx_c, vy_c, vz_c, nx, ny, nz, dx, dy, dz)
    _, E_rot = compute_energy_spectrum(vx_r, vy_r, vz_r, nx, ny, nz, dx, dy, dz)

    compute_energy_dissipation_enstophy(grid_vx, grid_vy, grid_vz, nx, ny, nz, dx, dy, dz)
    
    # Step 7: Save results
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
    print("\nPlotting energy spectra...")
    
    plt.figure(figsize=(12, 8))
    
    for k_centers, E_total, E_comp, E_rot, step_number, time_value in results_list:
        E_sum = E_comp + E_rot
        label_base = f"Step {step_number}, t={time_value:.3e}"
        
        plt.loglog(k_centers, E_total, '-', linewidth=2, label=f'{label_base} (Total)')
        plt.loglog(k_centers, E_comp, '--', linewidth=2, label=f'{label_base} (Compressive)')
        plt.loglog(k_centers, E_rot, ':', linewidth=2, label=f'{label_base} (Rotational)')
        plt.loglog(k_centers, E_sum, '-.', linewidth=1, alpha=0.7, label=f'{label_base} (Sum)')
    
    # Add reference line
    k_nonzero = k_centers[k_centers > 0]
    if len(k_nonzero) > 0:
        E_ref = 1e-2
        E_line = E_ref * (k_nonzero / 1.0)**(-5.0/3.0)
        plt.loglog(k_nonzero, E_line, 'k--', alpha=0.7, label='k^(-5/3) slope')
    
    plt.xlabel('Wavenumber k', fontsize=12)
    plt.ylabel('E(k)', fontsize=12)
    plt.title('Energy Spectra: Total, Compressive, and Rotational Components', fontsize=14)
    plt.ylim(1e-9, 1e-1)
    plt.legend(loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Command line interface
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Helmholtz-Hodge decomposition and energy spectrum analysis')
    parser.add_argument('data_files', type=str, nargs='+', help='Velocity data files')
    parser.add_argument('--visualize', '-v', action='store_true', help='Show velocity field visualization')
    parser.add_argument('--slice_z', '-s', type=int, default=None, help='Z-slice for visualization')
    args = parser.parse_args()
    
    # Analyze each file
    results = []
    for filename in args.data_files:
        result = analyze_file(filename, visualize=args.visualize, slice_z=args.slice_z)
        results.append(result)
    
    # Plot all spectra together
    plot_spectra(results)