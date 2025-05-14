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


    spatial_ke = 0.5*np.sum(Gx**2 + Gy**2 + Gz**2)
    print(f"Spatial KE : {spatial_ke:.6e}")

    # 4) coordinates remapped to [-π, π) ---------------------------------------
    # FINUFFT expects coords in [-π, π); scale each axis independently
    Lx, Ly, Lz = ux[-1] - ux[0] + (ux[1] - ux[0]), \
                 uy[-1] - uy[0] + (uy[1] - uy[0]), \
                 uz[-1] - uz[0] + (uz[1] - uz[0])

    # 4b) create full scattered coordinate list
    Xg, Yg, Zg = np.meshgrid(ux, uy, uz, indexing='ij')
    xg = ((Xg - ux[0]) * (2*np.pi/Lx) - np.pi).ravel()
    yg = ((Yg - uy[0]) * (2*np.pi/Ly) - np.pi).ravel()
    zg = ((Zg - uz[0]) * (2*np.pi/Lz) - np.pi).ravel()



    # 5) FINUFFT type-1 (non-uniform → uniform modes) ---------------------------
    Nd = (nx, ny, nz)
    Ntot = np.prod(Nd)
    
    # 5b) flatten the field values
    vgx = Gx.ravel()
    vgy = Gy.ravel()
    vgz = Gz.ravel()
    
    # helper for a single scattered NUFFT
    def finufft_forward_scattered(x_, y_, z_, v_):
        F_flat = finufft.nufft3d1(
            x_, y_, z_,
            v_.astype(np.complex128),
            (nx, ny, nz), eps=eps, isign=1, modeord=1
        )
        return F_flat.reshape((nx, ny, nz)) / (nx*ny*nz)
    
    # now these all match length = M
    Fvx_nu = finufft_forward_scattered(xg, yg, zg, vgx)
    Fvy_nu = finufft_forward_scattered(xg, yg, zg, vgy)
    Fvz_nu = finufft_forward_scattered(xg, yg, zg, vgz)





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
