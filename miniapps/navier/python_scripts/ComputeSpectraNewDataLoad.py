import numpy as np
import matplotlib.pyplot as plt
import re
import os 
import argparse

import scipy.stats as stats

### Parse Command-Line Arguments
parser = argparse.ArgumentParser(description='Compute energy spectrum from velocity data.')
parser.add_argument('data_file', type=str, help='Path to the data file')
args = parser.parse_args()
data_filename = [args.data_file]

plt.figure(figsize=(10, 8))

for file_to_extract_data in data_filename:
    # ---------------------------------------------------------------------------
    # 1) Read header, extract metadata (same idea as before but regex updated)
    # ---------------------------------------------------------------------------
    with open(file_to_extract_data, 'r') as f:
        header_lines = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.lstrip().startswith('#'):
                f.seek(pos)              # put the cursor back
                break
            header_lines.append(line)
    
    step_number_extracted = None
    time_extracted = None
    for line in header_lines:
        # "# Step   42" --> 42
        m = re.search(r'Step\s+(\d+)', line)
        if m: step_number_extracted = int(m.group(1))
        # "# Time   1.234e-02" --> 1.234e-02
        m = re.search(r'Time\s+([0-9.+-Ee]+)', line)
        if m: time_extracted = float(m.group(1))
    
    # ---------------------------------------------------------------------------
    # 2) Load the numerical payload as ONE flat vector
    # ---------------------------------------------------------------------------
    with open(file_to_extract_data, 'r') as f:
        payload = ' '.join(
            line for line in f if not line.lstrip().startswith('#'))
    flat = np.fromstring(payload, sep=' ')
    
    if flat.size % 6 != 0:
        raise ValueError("Vector length is not a multiple of 6 – corrupted file?")
    
    ND = flat.size // 6
    xpos = flat[0*ND : 1*ND]
    ypos = flat[1*ND : 2*ND]
    zpos = flat[2*ND : 3*ND]
    velx = flat[3*ND : 4*ND]
    vely = flat[4*ND : 5*ND]
    velz = flat[5*ND : 6*ND]
    

    # Round the coordinates to avoid floating-point precision issues
    xpos_rounded = np.round(xpos, decimals=10)
    ypos_rounded = np.round(ypos, decimals=10)
    zpos_rounded = np.round(zpos, decimals=10)

    # Determine unique coordinates after rounding
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
    print(f"Actual number of points: {actual_num_points}")

    if actual_num_points != expected_num_points:
        print("Warning: The actual number of data points does not match the expected number based on grid sizes.")

    # Create empty grids for velocities
    velx_grid = np.full((nx, ny, nz), np.nan)
    vely_grid = np.full((nx, ny, nz), np.nan)
    velz_grid = np.full((nx, ny, nz), np.nan)

    # Create mappings from coordinate to index using the rounded coordinates
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}

    # Assign data to the grids using rounded arrays for indexing
    for i in range(actual_num_points):
        xi = x_idx[xpos_rounded[i]]
        yi = y_idx[ypos_rounded[i]]
        zi = z_idx[zpos_rounded[i]]
        velx_grid[xi, yi, zi] = velx[i]
        vely_grid[xi, yi, zi] = vely[i]
        velz_grid[xi, yi, zi] = velz[i]

    # Replace NaNs with zeros
    velx_grid = np.nan_to_num(velx_grid)
    vely_grid = np.nan_to_num(vely_grid)
    velz_grid = np.nan_to_num(velz_grid)

    # Compute TKE grid
    tke_grid = 0.5 * (velx_grid**2 + vely_grid**2 + velz_grid**2)

    tke_physical = 0.5 * np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    print(f"[Rank 0] Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")
    
    # Perform 3D FFTs
    fft_velx = np.fft.fftn(velx_grid)
    fft_vely = np.fft.fftn(vely_grid)
    fft_velz = np.fft.fftn(velz_grid)

    # Shift zero frequency to center
    fft_velx = np.fft.fftshift(fft_velx)
    fft_vely = np.fft.fftshift(fft_vely)
    fft_velz = np.fft.fftshift(fft_velz)

    # Normalize FFT by total number of points to ensure correct amplitude scaling
    norm_factor = (nx * ny * nz)
    fft_velx /= norm_factor
    fft_vely /= norm_factor
    fft_velz /= norm_factor

    # Compute spectral energy density
    energy_density = 0.5 * (np.abs(fft_velx)**2 + np.abs(fft_vely)**2 + np.abs(fft_velz)**2)

    energy_density_mean = np.mean(energy_density)*norm_factor
    print(f"[Rank 0] Total Kinetic Energy in Fourier Space (TKE_Fourier):{energy_density_mean:.6f}")

    # Compute wavenumber vectors
    dx = x_unique[1] - x_unique[0]
    dy = y_unique[1] - y_unique[0]
    dz = z_unique[1] - z_unique[0]
    
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
    energy_flat = energy_density.flatten()

    num_bins = nx
    k_bin_edges = np.arange(0, num_bins+1) - 0.5
    
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
    E_k, _ = np.histogram(k_flat, bins=k_bin_edges, weights=energy_flat)
    
    # Use the step/time extracted from the header if available
    if step_number_extracted is not None:
        label_str = f"Step {step_number_extracted}"
    else:
        label_str = "Unknown Step"

    if time_extracted is not None:
        label_str += f", Time {time_extracted:.3e}"

    # Save wavenumbers and energy to a text file
    output_filename = os.path.join(os.path.dirname(file_to_extract_data), f'energy_spectrum_step_{step_number_extracted}.txt')
    print(f"[Rank 0] Saving energy spectrum to {output_filename}")
    np.savetxt(output_filename, np.column_stack((k_bin_centers, E_k)), 
               header=f'Wavenumber_k Energy_E(k) (Step {step_number_extracted}, Time {time_extracted:.3e})', 
               fmt='%.6e %.6e', comments='# ')

    plt.loglog(k_bin_centers, E_k, '-', label=label_str)

# Plot -5/3 slope line for reference
k_ref = 1
E_ref = .1e1
E_line = E_ref * (k_bin_centers / k_ref)**(-5.0/3.0)
plt.loglog(k_bin_centers, E_line, 'r--', label='k^-5/3 slope')
#plt.semilogy(k_bin_centers, E_line, 'r--', label='k^-5/3 slope')

# ymax = 1e1  
# ymin = 1e-6
# plt.ylim(ymin, ymax)
# xmax = np.max(kx)
# xmin = 1
# plt.xlim(xmin,xmax)

plt.xlabel('Wavenumber k')
plt.ylabel('k E(k)')
plt.title('Energy Spectra of the 3D Taylor-Green Vortex (Multiple Timesteps)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
