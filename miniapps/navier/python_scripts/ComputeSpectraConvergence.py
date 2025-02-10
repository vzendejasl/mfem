import numpy as np
import matplotlib.pyplot as plt
import re

directories = {
    # "32_pts": "/p/lustre1/zendejas/TGV/mfem/Order2_Re400/tgv_32/ElementCentersVelocity/",
    #"64_pts": "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64/ElementCentersVelocity/",
    #"128_pts": "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128/ElementCentersVelocity/",
    "128_pts": "/p/lustre1/zendejas/TGV/mfem/Order2_Re3200/tgv_128/ElementCentersVelocity_Re3200NumPtsPerDir32RefLv2P2/",
    "256_pts": "/p/lustre1/zendejas/TGV/mfem/Order2_Re3200/tgv_256/ElementCentersVelocity_Re3200NumPtsPerDir32RefLv3P2/"
    #"256_pts": "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_256/ElementCentersVelocity/",
    #"384_pts": "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_384/ElementCentersVelocity_Re1600NumPtsPerDir48P2/"
}

files_to_extract = {
    # "32_pts": [
    #     #directories["32_pts"] + 'cycle_7000/element_centers_7000.txt',
    #     directories["32_pts"] + 'cycle_8000/element_centers_8000.txt'
    # ],
    # "64_pts": [
    #     #directories["64_pts"] + 'cycle_7000/element_centers_7000.txt',
    #     directories["64_pts"] + 'cycle_9000/element_centers_9000.txt',
    # ],
    "128_pts": [
        #directories["128_pts"] + 'cycle_7000/element_centers_7000.txt',
        directories["128_pts"] + 'cycle_8800/element_centers_8800.txt',
    ],
      "256_pts": [
        #directories["128_pts"] + 'cycle_7000/element_centers_7000.txt',
        directories["256_pts"] + 'cycle_8806/element_centers_8806.txt',
    ],
    # "384_pts": [
    #     #directories["128_pts"] + 'cycle_7000/element_centers_7000.txt',
    #     directories["384_pts"] + 'cycle_9002/element_centers_9002.txt',
    # ]
}

styles = {
    "32_pts": {'marker': 'o', 'linestyle': '-', 'color': 'blue'},
    "64_pts": {'marker': 's', 'linestyle': '--', 'color': 'magenta'},
    "128_pts": {'marker': 's', 'linestyle': '--', 'color': 'blue'},
    "256_pts": {'marker': 's', 'linestyle': '--', 'color': 'red'},
    "384_pts": {'marker': 's', 'linestyle': '--', 'color': 'cyan'}
}

# directories = {
#     "64_pts_o2": "/p/lustre1/zendejas/TGV/mfem/Order2_Re400/tgv_64/ElementCentersVelocity/",
#     "64_pts_o3": "/p/lustre1/zendejas/TGV/mfem/Order3_Re400/tgv_64/ElementCentersVelocity/",
#     "128_pts_o2": "/p/lustre1/zendejas/TGV/mfem/Order2_Re400/tgv_128/ElementCentersVelocity/",
#     "128_pts_o3": "/p/lustre1/zendejas/TGV/mfem/Order3_Re400/tgv_128/ElementCentersVelocity/",
# }

# files_to_extract = {
#     "64_pts_o2":  [
#         #directories["32_pts"] + 'cycle_7000/element_centers_7000.txt',
#         directories["64_pts_o2"] + 'cycle_8000/element_centers_8000.txt'
#     ],
#     "64_pts_o3":  [
#         #directories["32_pts"] + 'cycle_7000/element_centers_7000.txt',
#         directories["64_pts_o3"] + 'cycle_8000/element_centers_8000.txt'
#     ],
#     "128_pts_o2":  [
#         #directories["32_pts"] + 'cycle_7000/element_centers_7000.txt',
#         directories["128_pts_o2"] + 'cycle_8002/element_centers_8002.txt'
#     ],
#     "128_pts_o3":  [
#         #directories["32_pts"] + 'cycle_7000/element_centers_7000.txt',
#         directories["128_pts_o3"] + 'cycle_8002/element_centers_8002.txt'
#     ],
# }

# styles = {
#      "64_pts_o2": {'marker': 'o', 'linestyle': '-', 'color': 'blue'},
#     "64_pts_o3": {'marker': 's', 'linestyle': '--', 'color': 'red'},
#     "128_pts_o2": {'marker': 's', 'linestyle': '--', 'color': 'black'},
#      "128_pts_o3": {'marker': 's', 'linestyle': '--', 'color': 'purple'}
# }

plt.figure(figsize=(10, 8))

# Dictionary to store TKE values for comparison
tke_comparison = {}

for resolution_label, file_list in files_to_extract.items():
    style = styles.get(resolution_label, {'marker': 'o', 'linestyle': '-', 'color': 'black'})
    for file_to_extract_data in file_list:
        # Read header lines
        try:
            with open(file_to_extract_data, 'r') as header_file:
                header_lines = [next(header_file) for _ in range(6)]
        except FileNotFoundError:
            print(f"Error: File {file_to_extract_data} not found.")
            continue
        except StopIteration:
            print(f"Error: File {file_to_extract_data} has fewer than 6 header lines.")
            continue

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

        # Load data, skipping the 6 header lines
        try:
            data = np.genfromtxt(file_to_extract_data, delimiter=' ', skip_header=6)
        except Exception as e:
            print(f"Error reading {file_to_extract_data}: {e}")
            continue

        if data.ndim != 2 or data.shape[1] < 6:
            print(f"Error: Data in {file_to_extract_data} does not have at least 6 columns.")
            continue

        xpos = data[:, 0]
        ypos = data[:, 1]
        zpos = data[:, 2]
        velx = data[:, 3]
        vely = data[:, 4]
        velz = data[:, 5]

        xpos_rounded = np.round(xpos, decimals=12)
        ypos_rounded = np.round(ypos, decimals=12)
        zpos_rounded = np.round(zpos, decimals=12)

        x_unique = np.unique(xpos_rounded)
        y_unique = np.unique(ypos_rounded)
        z_unique = np.unique(zpos_rounded)

        nx = len(x_unique)
        ny = len(y_unique)
        nz = len(z_unique)

        print(f"\nProcessing {file_to_extract_data}")
        print(f"Number of unique x values: {nx}")
        print(f"Number of unique y values: {ny}")
        print(f"Number of unique z values: {nz}")

        expected_num_points = nx * ny * nz
        actual_num_points = xpos.size

        print(f"Expected number of points: {expected_num_points}")
        print(f"Actual number of points: {actual_num_points}")

        if actual_num_points != expected_num_points:
            print("Warning: The actual number of data points does not match the expected number based on grid sizes.")

        # Initialize grids
        velx_grid = np.full((nx, ny, nz), np.nan)
        vely_grid = np.full((nx, ny, nz), np.nan)
        velz_grid = np.full((nx, ny, nz), np.nan)

        # Create index mappings for each unique coordinate
        x_idx = {val: i for i, val in enumerate(x_unique)}
        y_idx = {val: i for i, val in enumerate(y_unique)}
        z_idx = {val: i for i, val in enumerate(z_unique)}

        # Populate the velocity grids
        for i in range(actual_num_points):
            xi = x_idx.get(xpos_rounded[i], None)
            yi = y_idx.get(ypos_rounded[i], None)
            zi = z_idx.get(zpos_rounded[i], None)
            if xi is None or yi is None or zi is None:
                print(f"Warning: Position ({xpos_rounded[i]}, {ypos_rounded[i]}, {zpos_rounded[i]}) out of unique grid indices.")
                continue
            velx_grid[xi, yi, zi] = velx[i]
            vely_grid[xi, yi, zi] = vely[i]
            velz_grid[xi, yi, zi] = velz[i]

        # Replace NaNs with zeros (if any)
        velx_grid = np.nan_to_num(velx_grid)
        vely_grid = np.nan_to_num(vely_grid)
        velz_grid = np.nan_to_num(velz_grid)

        # Compute Total Kinetic Energy in Physical Space
        tke_physical = 0.5 * np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
        print(f"Total Kinetic Energy in Physical Space (TKE_physical): {tke_physical:.6f}")

        # FFT
        fft_velx = np.fft.fftn(velx_grid)
        fft_vely = np.fft.fftn(vely_grid)
        fft_velz = np.fft.fftn(velz_grid)

        # Shift the zero frequency component to the center
        fft_velx = np.fft.fftshift(fft_velx)
        fft_vely = np.fft.fftshift(fft_vely)
        fft_velz = np.fft.fftshift(fft_velz)

        norm_factor = (nx * ny * nz)
        fft_velx /= norm_factor
        fft_vely /= norm_factor
        fft_velz /= norm_factor

        energy_density = 0.5 * (np.abs(fft_velx)**2 + np.abs(fft_vely)**2 + np.abs(fft_velz)**2)

        dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
        dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
        dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0

        # Compute wavenumbers in cycles per length unit
        # Since L=2π, using d=dx/(2π)=1/nx simplifies the wavenumbers to integer multiples.
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dy)
        kz = np.fft.fftfreq(nz, d=dz)

        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
    
        k_flat = k_magnitude.flatten()
        energy_flat = energy_density.flatten()

        k_max = np.max(k_magnitude)
        k_bins = np.linspace(0, k_max, num=nx//2)
        k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
        E_k, _ = np.histogram(k_flat, bins=k_bins, weights=energy_flat)
        N_k, _ = np.histogram(k_flat, bins=k_bins)

        # Bin the energy density to obtain E(k) for plotting
        E_k, _ = np.histogram(k_flat, bins=k_bins, weights=energy_flat)

        label_str = f"{resolution_label}, Step {step_number_extracted}, Time {time_extracted:.3e}"

        # Plot E(k) * k^2 vs k (exactly as in your reference code)
        #plt.loglog(k_bin_centers, E_k*k_bin_centers**2,
        plt.loglog(k_bin_centers, E_k,           
                   marker=style['marker'], linestyle=style['linestyle'], color=style['color'],
                   label=label_str)

        # Compute Total Kinetic Energy in Fourier Space
        tke_fourier = np.sum(energy_density)
        print(f"Total Kinetic Energy in Fourier Space (TKE_fourier): {tke_fourier:.6f}")

        # Store TKE values for comparison
        tke_comparison[label_str] = {'Physical': tke_physical, 'Fourier': tke_fourier}

# Reference line for k^-5/3 slope (exactly as in your reference code)
if 'k_bin_centers' in locals() and k_bin_centers.size > 0:
    # To avoid division by zero, start from the first non-zero bin
    if k_bin_centers[0] == 0:
        k_ref = k_bin_centers[1] if k_bin_centers.size > 1 else k_bin_centers[0]
    else:
        k_ref = k_bin_centers[0]
    
    E_ref = .1e1  # Arbitrary reference energy scaling
    E_line = E_ref * (k_bin_centers / k_ref)**(-5.0/3.0)
    plt.loglog(k_bin_centers, E_line, 'r--', label='k$^{-5/3}$ slope')

#plt.ylim(1e-9, 1e4)
#plt.xlim(1, 384)  # Start from k=1 to avoid log(0) issues
plt.xlabel('Wavenumber $k$')
plt.ylabel('$E(k)$')
plt.title('Energy Spectra of the 3D Taylor-Green Vortex')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Summary of TKE Comparison
print("\n--- Total Kinetic Energy Comparison ---")
for label, ke_values in tke_comparison.items():
    tke_physical = ke_values['Physical']
    tke_fourier = ke_values['Fourier']
    if tke_physical != 0:
        relative_error = np.abs(tke_physical - tke_fourier) / tke_physical * 100
    else:
        relative_error = np.nan  # Avoid division by zero
    print(f"{label}:")
    print(f"  TKE_physical  = {tke_physical:.6f}")
    print(f"  TKE_fourier   = {tke_fourier:.6f}")
    print(f"  Relative Error = {relative_error:.6f}%\n")
