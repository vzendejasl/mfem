import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import os

# =============================================================================
# Method 1: Continuous Histogram Binning (Standard)
# =============================================================================
def compute_spectrum_histogram(u, v, w, dx, dy, dz):
    """
    Computes the 1D energy spectrum using a vectorized FFT (with fftshift)
    and bins energy over spherical shells in Fourier space using a continuous
    binning (angular wavenumbers) and the 'sum' statistic.
    
    Parameters:
      u, v, w: 3D arrays (velocity components)
      dx, dy, dz: grid spacing in x, y, z.
      
    Returns:
      k_bin_centers: 1D array of wavenumber bin centers.
      E_k: Energy spectrum (integrated energy in each bin).
      k_max: Maximum wavenumber.
    """
    nx, ny, nz = u.shape
    nt = nx * ny * nz

    # Compute FFTs (normalize and shift zero frequency to center)
    fft_u = np.fft.fftn(u) / nt
    fft_v = np.fft.fftn(v) / nt
    fft_w = np.fft.fftn(w) / nt

    fft_u = np.fft.fftshift(fft_u)
    fft_v = np.fft.fftshift(fft_v)
    fft_w = np.fft.fftshift(fft_w)

    # Compute spectral kinetic energy density
    energy_density = 0.5 * (np.abs(fft_u)**2 + np.abs(fft_v)**2 + np.abs(fft_w)**2)

    # Compute angular wavenumbers; here we use:
    # k = 2π * f with f from np.fft.fftfreq assuming period = n*dx.
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
    
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kz = np.fft.fftshift(kz)

    # Build a 3D grid and compute the magnitude of k at each point.
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Flatten arrays for binning.
    k_flat = k_magnitude.flatten()
    energy_flat = energy_density.flatten()

    k_max = np.max(k_magnitude)

    # This is what I had prevously
    num_bins = nx//2  # e.g., half the grid size.
    k_bin_edges = np.linspace(0, k_max, num=num_bins)
  
    # This option matchs with the spectra result.
    # num_bins = nx
    # k_bin_edges = np.arange(0, num_bins+1) - 0.5
    
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
    E_k, _ = np.histogram(k_flat, bins=k_bin_edges, weights=energy_flat)
    
    return k_bin_centers, E_k, k_max

# =============================================================================
# Method 2: Tsaad Method (Explicit Loop with Index Rounding)
# =============================================================================
def compute_spectrum_tsaad(u, v, w, lx, ly, lz, smooth=True):
    """
    Computes the 1D energy spectrum using an explicit loop over the FFT grid.
    Each FFT coefficient is assigned an effective wave number by adjusting the
    index for negative frequencies and then rounding.
    
    Parameters:
      u, v, w: 3D arrays (velocity components).
      lx, ly, lz: Domain sizes in x, y, z.
      smooth: If True, apply a moving average to smooth the result.
      
    Returns:
      wave_numbers: 1D array of wave number bins.
      tke_spectrum: Energy spectrum.
      knyquist: Nyquist wavenumber.
    """
    nx, ny, nz = u.shape
    nt = nx * ny * nz

    # Compute fundamental frequencies from domain sizes.
    k0x = 2.0 * np.pi / lx
    k0y = 2.0 * np.pi / ly
    k0z = 2.0 * np.pi / lz
    knorm = (k0x + k0y + k0z) / 3.0

    # Compute FFTs without shifting.
    fft_u = np.fft.fftn(u) / nt
    fft_v = np.fft.fftn(v) / nt
    fft_w = np.fft.fftn(w) / nt

    # Compute spectral kinetic energy.
    tkeh = 0.5 * (np.abs(fft_u)**2 + np.abs(fft_v)**2 + np.abs(fft_w)**2)

    n_bins = nx  # Assume cubic grid.
    tke_spectrum = np.zeros(n_bins)

    # Define maximum indices for positive frequencies.
    kxmax = nx // 2
    kymax = ny // 2
    kzmax = nz // 2

    for i in range(nx):
        rkx = i if i <= kxmax else i - nx
        for j in range(ny):
            rky = j if j <= kymax else j - ny
            for k in range(nz):
                rkz = k if k <= kzmax else k - nz
                r = np.sqrt(rkx**2 + rky**2 + rkz**2)
                k_index = int(np.round(r))
                if k_index < n_bins:
                    tke_spectrum[k_index] += tkeh[i, j, k]

    tke_spectrum = tke_spectrum / knorm

    if smooth:
        window_size = 1
        window = np.ones(window_size) / window_size
        tke_spectrum = np.convolve(tke_spectrum, window, mode='same')

    wave_numbers = knorm * np.arange(n_bins)
    knyquist = knorm * min(nx, ny, nz) / 2

    return wave_numbers, tke_spectrum, knyquist

# =============================================================================
# Method 3: Discrete Histogram Binning (Mimicking Tsaad)
# =============================================================================
def compute_spectrum_histogram_discrete(u, v, w, Lx, Ly, Lz):
    """
    Computes the 1D energy spectrum by computing an effective discrete wave
    number for each FFT coefficient (using index-based negative frequency
    adjustment) and then binning energy into bins whose centers are
    knorm * (0,1,2,...). This should mimic the Tsaad method.
    
    Parameters:
      u, v, w: 3D arrays (velocity components).
      Lx, Ly, Lz: Domain sizes.
      
    Returns:
      bin_centers: 1D array of wave number bin centers (knorm * integers).
      E_k: Energy spectrum computed in these discrete bins.
      knyquist: Nyquist wavenumber.
    """
    nx, ny, nz = u.shape
    nt = nx * ny * nz

    # Fundamental frequencies based on domain sizes.
    k0x = 2 * np.pi / Lx
    k0y = 2 * np.pi / Ly
    k0z = 2 * np.pi / Lz
    knorm = (k0x + k0y + k0z) / 3.0

    # Compute FFTs (no shift) and energy.
    fft_u = np.fft.fftn(u) / nt
    fft_v = np.fft.fftn(v) / nt
    fft_w = np.fft.fftn(w) / nt
    tkeh = 0.5 * (np.abs(fft_u)**2 + np.abs(fft_v)**2 + np.abs(fft_w)**2)

    # Create index grids.
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    # Adjust indices for negative frequencies.
    I_eff = np.where(I <= nx//2, I, I - nx)
    J_eff = np.where(J <= ny//2, J, J - ny)
    K_eff = np.where(K <= nz//2, K, K - nz)
    # Compute the discrete magnitude (i.e. the "integer" index) and then the effective k value.
    r = np.sqrt(I_eff**2 + J_eff**2 + K_eff**2)
    k_vals = knorm * r  # Effective physical wave number.

    # Flatten for binning.
    k_flat = k_vals.flatten()
    energy_flat = tkeh.flatten()

    # Define discrete bins: we want bin centers at knorm*(0, 1, 2, ..., n_bins-1)
    n_bins = nx  # same as Tsaad.
    # Define bin edges so that bin i is centered at knorm*i.
    bin_edges = knorm * (np.arange(0, n_bins+1) - 0.5)
    bin_centers = knorm * np.arange(n_bins)

    # Bin the energy using the 'sum' statistic.
    #E_k, _, _ = stats.binned_statistic(k_flat, energy_flat, statistic="sum", bins=bin_edges)
    E_k, _ = np.histogram(k_flat, bins=bin_edges, weights=energy_flat)

    # For knyquist we use the same definition as in Tsaad.
    knyquist = knorm * min(nx, ny, nz) / 2

    return bin_centers, E_k, knyquist

# =============================================================================
# Main Script: Read Data, Compute and Compare the Three Spectra
# =============================================================================
if __name__ == '__main__':
    # --- File patch directory and file path (update as needed) ---

    file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P2/'
    file_path = os.path.join(file_directory, 'cycle_9000', 'element_centers_9000.txt')

    
    # --- Read header to extract metadata ---
    with open(file_path, 'r') as header_file:
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

    # --- Load the velocity data (skipping header lines) ---
    data = np.genfromtxt(file_path, delimiter=' ', skip_header=6)
    # Columns: xpos, ypos, zpos, velx, vely, velz
    xpos = data[:, 0]
    ypos = data[:, 1]
    zpos = data[:, 2]
    velx = data[:, 3]
    vely = data[:, 4]
    velz = data[:, 5]

    # --- Construct the 3D grid ---
    xpos_rounded = np.round(xpos, decimals=10)
    ypos_rounded = np.round(ypos, decimals=10)
    zpos_rounded = np.round(zpos, decimals=10)
    
    x_unique = np.unique(xpos_rounded)
    y_unique = np.unique(ypos_rounded)
    z_unique = np.unique(zpos_rounded)
    
    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)
    
    print(f"Unique grid points: nx={nx}, ny={ny}, nz={nz}")
    
    expected_num_points = nx * ny * nz
    actual_num_points = xpos.size
    print(f"Expected points: {expected_num_points}, Actual points: {actual_num_points}")
    
    # --- Assemble the 3D velocity arrays ---
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
    
    # --- Compute grid spacing and domain sizes ---
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
    
    # For a periodic domain, set L = n*dx.
    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz
    
    print(f"Domain sizes: Lx={Lx}, Ly={Ly}, Lz={Lz}")
    
    # --- Compute spectra using all three methods ---
    # Method 1: Continuous Histogram
    k_hist, E_hist, kmax_hist = compute_spectrum_histogram(velx_grid, vely_grid, velz_grid, dx, dy, dz)
    total_TKE_hist = np.sum(E_hist)
    
    # Method 2: Tsaad Method
    wave_numbers_tsaad, E_tsaad, knyquist_tsaad = compute_spectrum_tsaad(velx_grid, vely_grid, velz_grid, Lx, Ly, Lz, smooth=True)
    total_TKE_tsaad = np.sum(E_tsaad)
    
    # Method 3: Discrete Histogram (mimics Tsaad)
    k_disc, E_disc, knyquist_disc = compute_spectrum_histogram_discrete(velx_grid, vely_grid, velz_grid, Lx, Ly, Lz)
    total_TKE_disc = np.sum(E_disc)
    
    # Print diagnostics.
    print("\nHistogram Method (Continuous):")
    print("  Maximum wavenumber (k_max):", kmax_hist)
    print("  Total TKE (integrated over k):", total_TKE_hist)
    print("  First bin: k =", k_hist[0])
    print("  Last bin: k =", k_hist[-1])
    
    print("\nTsaad Method:")
    print("  Nyquist wavenumber:", knyquist_tsaad)
    print("  Total TKE (integrated over k):", total_TKE_tsaad)
    print("  First wave number:", wave_numbers_tsaad[0])
    print("  Last wave number:", wave_numbers_tsaad[-1])
    
    print("\nDiscrete Histogram Method (Mimicking Tsaad):")
    print("  Nyquist wavenumber:", knyquist_disc)
    print("  Total TKE (integrated over k):", total_TKE_disc)
    print("  First bin: k =", k_disc[0])
    print("  Last bin: k =", k_disc[-1])
    
    # --- Plot all three spectra ---
    plt.figure(figsize=(10, 6))
    plt.loglog(k_hist, E_hist, 'b-', label='Continuous Histogram')
    plt.loglog(wave_numbers_tsaad, E_tsaad, 'r--', label='Tsaad Method')
    plt.loglog(k_disc, E_disc, 'g:', label='Discrete Histogram (Mimics Tsaad)')
    plt.xlabel('Wavenumber k')
    plt.ylabel('E(k)')
    plt.title(f'Energy Spectra Comparison (Step {step_number_extracted}, Time {time_extracted:.3e})')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

