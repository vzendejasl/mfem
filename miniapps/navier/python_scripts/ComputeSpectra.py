import numpy as np
import matplotlib.pyplot as plt
import re
import os 

import scipy.stats as stats

#file_directory = '/p/lustre2/zendejas/TestCases/mfem/TGV/Order2_Re1600/tgv_512/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv4P2/'
#file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_384/ElementCentersVelocity_Re1600NumPtsPerDir48RefLv3P2/'
#file_directory = '/g/g11/zendejas/Documents/mfem_build/mfem/miniapps/navier/ElementCentersVelocity_Re1600NumPtsPerDir4RefLv0P2/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/'
#file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_64_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P4/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P2/'
#file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128_test_sampling_higher/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/'
file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re400/tgv_128_test_sampling/ElementCentersVelocity_Re400NumPtsPerDir32RefLv2P2/'
file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re400/tgv_128_test_sampling/ElementCentersVelocity_Re400NumPtsPerDir32RefLv2P4/'
file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re3200/tgv_256/SampledDataVelocityP2/'
file_directory = '/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_256/SampledDataVelocityP3/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_64/SampledDataVelocityP3/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128/SampledDataVelocityP3/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128_more_quad_pts/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P2/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_256_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv3P2/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P3/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_64_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P4/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_64_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P3/'
# file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P2/'
files_to_extract_data = [
    #file_directory + 'cycle_3000/element_centers_3000.txt',
    # file_directory + 'cycle_9001/element_centers_9001.txt',
    #file_directory + 'cycle_11500/element_centers_11500.txt',
    # file_directory + 'cycle_9000/sampled_data_9000.txt'
    file_directory + 'cycle_9002/sampled_data_9002.txt'
]

plt.figure(figsize=(10, 8))

for file_to_extract_data in files_to_extract_data:
    # Read header lines to extract step and time
    with open(file_to_extract_data, 'r') as header_file:
        header_lines = [next(header_file) for _ in range(6)]
    
    step_number_extracted = None
    time_extracted = None
    for line in header_lines:
        if 'Step' in line:
            step_match = re.search(r'Step\s*=\s*(\d+)', line)
            if step_match:
                step_number_extracted = step_match.group(1)
        if 'Time' in line:
            time_match = re.search(r'Time\s*=\s*([0-9.eE+-]+)', line)
            if time_match:
                time_extracted = float(time_match.group(1))

    if step_number_extracted is None:
        step_number_extracted = "Unknown"
    if time_extracted is None:
        time_extracted = 0.0

    # Load data skipping the 6 header lines
    data = np.genfromtxt(file_to_extract_data, delimiter=' ', skip_header=6)

    # Assign column data
    xpos = data[:, 0]
    ypos = data[:, 1]
    zpos = data[:, 2]
    velx = data[:, 3]
    vely = data[:, 4]
    velz = data[:, 5]

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
    output_filename = os.path.join(file_directory, f'energy_spectrum_step_{step_number_extracted}.txt')
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
