#!/usr/bin/env python3
"""
Compute longitudinal and transverse two-point correlations f(r), g(r)
from a 3D velocity field using the FFT spectral-tensor pipeline.

The I/O and grid reconstruction flow intentionally mirrors
ComputeSpectraCompressiveVorticalModes.py:
1) header detection + parsing
2) chunked read (text or HDF5)
3) coordinate-index map reconstruction onto a structured grid
4) automatic periodic endpoint removal (last point in each direction)
"""

import argparse
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft as fft


# ------------------------------------------------------------------ #
#  Step 1: Read and parse data file (mirrors original style)
# ------------------------------------------------------------------ #
def detect_header_lines(filename):
    """Automatically detect the number of header lines in a text file"""
    print(f"  Auto-detecting header length for {filename}...")
    header_count = 0
    try:
        with open(filename, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    header_count += 1
                    continue

                try:
                    parts = line_stripped.split()
                    [float(x) for x in parts]
                    break
                except ValueError:
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


def read_data_file_chunked(filename, chunk_size=5_000_000, skiprows=5, decimals=10):
    """
    Read velocity data file in chunks, reconstruct structured grid,
    and always drop duplicated periodic endpoint.
    """
    print(f"Reading data from: {filename} (chunked, size={chunk_size})")

    if filename.endswith('.h5'):
        print(f"  Loading data from HDF5: {filename} (chunked read)")
        with h5py.File(filename, 'r') as f:
            dset = f['data']
            total_pts = dset.shape[0]
            print(f"  Total data points: {total_pts}")

            xpos = np.empty(total_pts, dtype=np.float64)
            ypos = np.empty(total_pts, dtype=np.float64)
            zpos = np.empty(total_pts, dtype=np.float64)
            velx = np.empty(total_pts, dtype=np.float64)
            vely = np.empty(total_pts, dtype=np.float64)
            velz = np.empty(total_pts, dtype=np.float64)

            for i in range(0, total_pts, chunk_size):
                end_idx = min(i + chunk_size, total_pts)
                chunk_data = dset[i:end_idx]

                xpos[i:end_idx] = np.round(chunk_data[:, 0], decimals)
                ypos[i:end_idx] = np.round(chunk_data[:, 1], decimals)
                zpos[i:end_idx] = np.round(chunk_data[:, 2], decimals)
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

        xpos_list, ypos_list, zpos_list = [], [], []
        velx_list, vely_list, velz_list = [], [], []

        for chunk in reader:
            xp = np.round(chunk.iloc[:, 0].values, decimals)
            yp = np.round(chunk.iloc[:, 1].values, decimals)
            zp = np.round(chunk.iloc[:, 2].values, decimals)
            vx = chunk.iloc[:, 3].values
            vy = chunk.iloc[:, 4].values
            vz = chunk.iloc[:, 5].values

            xpos_list.append(xp)
            ypos_list.append(yp)
            zpos_list.append(zp)
            velx_list.append(vx)
            vely_list.append(vy)
            velz_list.append(vz)

        total_pts = sum(arr.size for arr in xpos_list)
        print(f"  Total data points: {total_pts}")

        xpos = np.empty(total_pts, dtype=xpos_list[0].dtype)
        ypos = np.empty(total_pts, dtype=ypos_list[0].dtype)
        zpos = np.empty(total_pts, dtype=zpos_list[0].dtype)
        velx = np.empty(total_pts, dtype=velx_list[0].dtype)
        vely = np.empty(total_pts, dtype=vely_list[0].dtype)
        velz = np.empty(total_pts, dtype=velz_list[0].dtype)

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

        del xpos_list, ypos_list, zpos_list
        del velx_list, vely_list, velz_list

    x_unique = np.unique(xpos)
    y_unique = np.unique(ypos)
    z_unique = np.unique(zpos)
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    print(f"  Grid dimensions: {nx} × {ny} × {nz}")

    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0
    print(f"  Grid spacing: dx={dx:.8f}, dy={dy:.8f}, dz={dz:.8f}")

    expected_num_points = nx * ny * nz
    if total_pts != expected_num_points:
        raise ValueError(
            f"Actual points ({total_pts}) != expected ({expected_num_points})"
        )

    print("  Reconstructing velocity grids...")
    velx_grid = np.zeros((nx, ny, nz))
    vely_grid = np.zeros((nx, ny, nz))
    velz_grid = np.zeros((nx, ny, nz))

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

    print("  Applying periodic slicing (dropping last point)...")
    return (velx_grid[:-1, :-1, :-1],
            vely_grid[:-1, :-1, :-1],
            velz_grid[:-1, :-1, :-1],
            x_unique[:-1],
            y_unique[:-1],
            z_unique[:-1],
            dx, dy, dz)


# ------------------------------------------------------------------ #
#  Step 2-4: FFT pipeline for tensor correlations
# ------------------------------------------------------------------ #
def compute_tensor_correlations(vx, vy, vz):
    """Compute R_ij(r) from spectral tensor via Wiener-Khinchin."""
    print("Step 1: Forward FFT of fluctuating velocity")
    ux = vx - np.mean(vx)
    uy = vy - np.mean(vy)
    uz = vz - np.mean(vz)
    n_tot = ux.size

    ux_k = fft.fftn(ux)
    uy_k = fft.fftn(uy)
    uz_k = fft.fftn(uz)

    print("Step 2: Spectral tensor Phi_ij(k) = u_i(k) u_j*(k)")
    phi = {
        "xx": ux_k * np.conj(ux_k),
        "xy": ux_k * np.conj(uy_k),
        "xz": ux_k * np.conj(uz_k),
        "yx": uy_k * np.conj(ux_k),
        "yy": uy_k * np.conj(uy_k),
        "yz": uy_k * np.conj(uz_k),
        "zx": uz_k * np.conj(ux_k),
        "zy": uz_k * np.conj(uy_k),
        "zz": uz_k * np.conj(uz_k),
    }

    print("Step 4: Inverse FFT -> R_ij(r)")
    R = {key: np.real(fft.ifftn(val)) / n_tot for key, val in phi.items()}
    return R, (ux_k, uy_k, uz_k)


def make_kgrids(nx, ny, nz, dx, dy, dz):
    """
    Create physical wavenumber grids, matching the example FFT style:
      k = 2*pi*fftfreq(N, d=spacing)
    """
    kx = 2.0 * np.pi * fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * fft.fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    return KX, KY, KZ


def compute_energy_spectrum_1d(ux_k, uy_k, uz_k, dx, dy, dz):
    """Optional Step 3: shell-integrated 1D kinetic energy spectrum E(k)."""
    nx, ny, nz = ux_k.shape
    n_tot = nx * ny * nz

    KX, KY, KZ = make_kgrids(nx, ny, nz, dx, dy, dz)
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    e_mode = 0.5 * (np.abs(ux_k)**2 + np.abs(uy_k)**2 + np.abs(uz_k)**2) / (n_tot**2)

    n_bins = max(16, min(nx, ny, nz) // 2)
    bins = np.linspace(0.0, kmag.max(), n_bins + 1)
    E_k, edges = np.histogram(kmag.ravel(), bins=bins, weights=e_mode.ravel())
    k_center = 0.5 * (edges[:-1] + edges[1:])
    return k_center, E_k


def compute_ke_consistency_from_tensor(vx, vy, vz):
    """
    Compute kinetic energy in physical space and from spectral-tensor diagonal,
    then report consistency:
      KE_phys = 0.5 * <u^2+v^2+w^2>
      KE_spec = sum_k 0.5 * (Phi11 + Phi22 + Phi33) / N^2
    with Phi_ii(k) = u_hat_i(k) * conj(u_hat_i(k)).
    """
    n_tot = vx.size
    ke_phys = 0.5 * np.mean(vx*vx + vy*vy + vz*vz)

    ux_k = fft.fftn(vx)
    uy_k = fft.fftn(vy)
    uz_k = fft.fftn(vz)

    phi_trace = np.abs(ux_k)**2 + np.abs(uy_k)**2 + np.abs(uz_k)**2
    ke_spec = 0.5 * np.sum(phi_trace) / (n_tot * n_tot)

    abs_diff = abs(ke_phys - ke_spec)
    rel_diff = abs_diff / abs(ke_phys) if abs(ke_phys) > 0.0 else 0.0
    return {
        "ke_phys": ke_phys,
        "ke_spec_tensor_diag": ke_spec,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
    }


def extract_f_g(R, dx, dy, dz, max_r_points):
    """Extract normalized f(r), g(r) from principal axes of R_ij(r)."""
    nx, ny, nz = R["xx"].shape
    nmax = min(nx, ny, nz) // 2
    if max_r_points is not None:
        nmax = min(nmax, max_r_points)

    idx = np.arange(nmax + 1, dtype=int)

    f_x = R["xx"][idx, 0, 0]
    g_x = 0.5 * (R["yy"][idx, 0, 0] + R["zz"][idx, 0, 0])

    f_y = R["yy"][0, idx, 0]
    g_y = 0.5 * (R["xx"][0, idx, 0] + R["zz"][0, idx, 0])

    f_z = R["zz"][0, 0, idx]
    g_z = 0.5 * (R["xx"][0, 0, idx] + R["yy"][0, 0, idx])

    f_avg = (f_x + f_y + f_z) / 3.0
    g_avg = (g_x + g_y + g_z) / 3.0

    f_norm = f_avg / (f_avg[0] if np.abs(f_avg[0]) > 0 else 1.0)
    g_norm = g_avg / (g_avg[0] if np.abs(g_avg[0]) > 0 else 1.0)

    def safe_normalize(curve):
        c0 = curve[0]
        if np.abs(c0) > 0.0:
            return curve / c0
        return np.full_like(curve, np.nan, dtype=np.float64)

    f_x_norm = safe_normalize(f_x)
    f_y_norm = safe_normalize(f_y)
    f_z_norm = safe_normalize(f_z)

    spacing = float((dx + dy + dz) / 3.0)
    r = idx * spacing
    return {
        "r": r,
        "f": f_norm,
        "g": g_norm,
        "f_x": f_x,
        "f_y": f_y,
        "f_z": f_z,
        "f_x_norm": f_x_norm,
        "f_y_norm": f_y_norm,
        "f_z_norm": f_z_norm,
        "g_x": g_x,
        "g_y": g_y,
        "g_z": g_z,
        "f_raw_avg": f_avg,
        "g_raw_avg": g_avg,
    }


def second_derivative_at_origin(r, y):
    """
    Estimate y''(0) near the left boundary.
    Uses 5-point forward stencil (O(h^4)) when available:
      y''(0) = (35y0 - 104y1 + 114y2 - 56y3 + 11y4) / (12 h^2)
    Falls back to 4-point forward stencil (O(h^2)):
      y''(0) = (2y0 - 5y1 + 4y2 - y3) / h^2
    """
    if len(r) < 4 or len(y) < 4:
        raise ValueError("Need at least 4 points to estimate second derivative at r=0.")
    h = r[1] - r[0]
    if h <= 0.0:
        raise ValueError("Non-positive spacing in r.")
    ncheck = 5 if len(r) >= 5 else 4
    if not np.allclose(np.diff(r[:ncheck]), h, rtol=1e-10, atol=1e-14):
        raise ValueError("Non-uniform r spacing near origin.")

    if len(y) >= 5:
        return (
            35.0*y[0] - 104.0*y[1] + 114.0*y[2] - 56.0*y[3] + 11.0*y[4]
        ) / (12.0*h*h)

    return (2.0*y[0] - 5.0*y[1] + 4.0*y[2] - y[3]) / (h*h)


def compute_taylor_microscales_from_curves(r, f_curve, g_curve):
    """
    Compute Taylor microscales from normalized/unnormalized correlation curves:
      lambda_L = sqrt(-f(0)/f''(0))
      lambda_T = sqrt(-g(0)/g''(0))
    """
    d2f0 = second_derivative_at_origin(r, f_curve)
    d2g0 = second_derivative_at_origin(r, g_curve)

    f0 = f_curve[0]
    g0 = g_curve[0]

    lam_L = np.sqrt(-f0 / d2f0) if d2f0 < 0.0 else np.nan
    lam_T = np.sqrt(-g0 / d2g0) if d2g0 < 0.0 else np.nan

    return lam_L, lam_T, d2f0, d2g0


def compute_component_taylor_microscales_from_f(fg, eps=1e-14):
    """
    Compute componentwise longitudinal Taylor microscales from:
      f_x(r)=R11(r e_x), f_y(r)=R22(r e_y), f_z(r)=R33(r e_z)
    using lambda_b = sqrt(-f_b(0) / f_b''(0)).

    For zero-energy components (f_b(0) ~ 0), return lambda_b = 0
    to mirror the MFEM-style averaging behavior.
    """
    r = fg["r"]
    result = {}
    for tag in ("x", "y", "z"):
        curve = fg[f"f_{tag}"]
        c0 = curve[0]
        if np.abs(c0) <= eps:
            result[tag] = {"lambda": 0.0, "d2": 0.0, "c0": c0, "status": "zero_component"}
            continue

        d2 = second_derivative_at_origin(r, curve)
        lam = np.sqrt(-c0 / d2) if d2 < 0.0 else np.nan
        status = "ok" if np.isfinite(lam) else "invalid_curvature"
        result[tag] = {"lambda": lam, "d2": d2, "c0": c0, "status": status}

    lam_avg = (result["x"]["lambda"] + result["y"]["lambda"] + result["z"]["lambda"]) / 3.0
    return result, lam_avg


def compute_integral_length_scale_from_curve(r, curve):
    """Integral length scale from a correlation curve: L = integral f(r) dr."""
    return np.trapz(curve, r)


def compute_component_integral_length_scales_from_f(fg, eps=1e-14):
    """
    Componentwise integral scales from longitudinal normalized curves:
      Lx = integral f_x_norm(r) dr, etc.

    If a component has near-zero zero-lag energy (undefined normalization),
    set contribution to 0 to mirror the /3 averaging convention.
    """
    r = fg["r"]
    result = {}
    for tag in ("x", "y", "z"):
        c0 = fg[f"f_{tag}"][0]
        if np.abs(c0) <= eps:
            result[tag] = {"L": 0.0, "status": "zero_component"}
            continue
        L = compute_integral_length_scale_from_curve(r, fg[f"f_{tag}_norm"])
        result[tag] = {"L": L, "status": "ok"}

    L_avg = (result["x"]["L"] + result["y"]["L"] + result["z"]["L"]) / 3.0
    return result, L_avg


def compute_taylor_microscale_gradient_spectral(vx, vy, vz, dx, dy, dz):
    """
    MFEM-style component Taylor microscales in spectral space:
      lambda_b = sqrt(<u_b^2> / <(du_b/dx_b)^2>), b = x,y,z
      lambda_avg = (lambda_x + lambda_y + lambda_z)/3
    """
    nx, ny, nz = vx.shape
    n_tot = nx * ny * nz
    KX, KY, KZ = make_kgrids(nx, ny, nz, dx, dy, dz)

    ux_k = fft.fftn(vx)
    uy_k = fft.fftn(vy)
    uz_k = fft.fftn(vz)

    # Parseval factors cancel in the ratio, but keep explicit average form.
    fac = 1.0 / (n_tot * n_tot)
    u2_x = fac * np.sum(np.abs(ux_k)**2)
    u2_y = fac * np.sum(np.abs(uy_k)**2)
    u2_z = fac * np.sum(np.abs(uz_k)**2)

    du2_x = fac * np.sum((KX**2) * (np.abs(ux_k)**2))
    du2_y = fac * np.sum((KY**2) * (np.abs(uy_k)**2))
    du2_z = fac * np.sum((KZ**2) * (np.abs(uz_k)**2))

    lam_x = np.sqrt(u2_x / du2_x) if du2_x > 0.0 else 0.0
    lam_y = np.sqrt(u2_y / du2_y) if du2_y > 0.0 else 0.0
    lam_z = np.sqrt(u2_z / du2_z) if du2_z > 0.0 else 0.0
    lam_avg = (lam_x + lam_y + lam_z) / 3.0

    return {
        "x": {"lambda": lam_x, "u2": u2_x, "du2": du2_x},
        "y": {"lambda": lam_y, "u2": u2_y, "du2": du2_y},
        "z": {"lambda": lam_z, "u2": u2_z, "du2": du2_z},
        "avg": lam_avg,
    }


def print_table(headers, rows):
    """Print a simple fixed-width ASCII table."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return "| " + " | ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print(fmt([str(h) for h in headers]))
    print(sep)
    for row in str_rows:
        print(fmt(row))
    print(sep)


# ------------------------------------------------------------------ #
#  Step 5: Save / plot
# ------------------------------------------------------------------ #
def save_fg_csv(out_csv, fg, step_number, time_value):
    columns = [
        ("r", fg["r"]),
        ("f_norm", fg["f"]),
        ("g_norm", fg["g"]),
        ("f_raw_avg", fg["f_raw_avg"]),
        ("g_raw_avg", fg["g_raw_avg"]),
        ("f_x", fg["f_x"]),
        ("f_y", fg["f_y"]),
        ("f_z", fg["f_z"]),
        ("f_x_norm", fg["f_x_norm"]),
        ("f_y_norm", fg["f_y_norm"]),
        ("f_z_norm", fg["f_z_norm"]),
        ("g_x", fg["g_x"]),
        ("g_y", fg["g_y"]),
        ("g_z", fg["g_z"]),
    ]

    colw = 24
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(", ".join(name.rjust(colw) for name, _ in columns) + "\n")
        for row in zip(*(arr for _, arr in columns)):
            f.write(", ".join(f"{val:>{colw}.16e}" for val in row) + "\n")


def plot_fg(fg, step_number, time_value, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    ax.plot(fg["r"], fg["f"], lw=2.2, color="black", label="f(r) avg longitudinal")
    ax.plot(fg["r"], fg["g"], lw=2.2, color="red", label="g(r) avg transverse")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("r")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Averaged correlations, step={step_number}, t={time_value:.4e}")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="best")

    axc = axes[1]
    axc.plot(fg["r"], fg["f_x_norm"], lw=2.0, label="R11(r,0,0)/R11(0,0,0)")
    axc.plot(fg["r"], fg["f_y_norm"], lw=2.0, label="R22(0,r,0)/R22(0,0,0)")
    axc.plot(fg["r"], fg["f_z_norm"], lw=2.0, label="R33(0,0,r)/R33(0,0,0)")
    axc.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axc.set_xlabel("r")
    axc.set_ylabel("Rii(re_i)/Rii(0,0,0)")
    axc.set_title("Component-wise Longitudinal Rij")
    axc.grid(True, alpha=0.3, ls="--")
    axc.legend(loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_energy_spectrum(out_png, k, E_k, step_number, time_value, show=True):
    mask = (k > 0.0) & (E_k > 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(k[mask], E_k[mask], lw=2.0)
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"1D energy spectrum, step={step_number}, t={time_value:.4e}")
    ax.grid(True, which="both", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------------ #
#  Verification: 3D Taylor-Green vortex
# ------------------------------------------------------------------ #
def build_taylor_green_vortex(n):
    """
    Build inviscid TGV on physical domain [0, 1)^3:
      u = sin(2*pi*x) cos(2*pi*y) cos(2*pi*z)
      v = -cos(2*pi*x) sin(2*pi*y) cos(2*pi*z)
      w = 0
    """
    L = 1.0
    x = np.linspace(0.0, L, n, endpoint=False)
    y = np.linspace(0.0, L, n, endpoint=False)
    z = np.linspace(0.0, L, n, endpoint=False)
    dx = x[1] - x[0] if n > 1 else 1.0
    dy = y[1] - y[0] if n > 1 else 1.0
    dz = z[1] - z[0] if n > 1 else 1.0

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y) * np.cos(2.0 * np.pi * Z)
    v = -np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y) * np.cos(2.0 * np.pi * Z)
    w = np.zeros_like(u)
    return u, v, w, x, y, z, dx, dy, dz


def save_tgv_axis_verification_csv(out_csv, r, f_num, g_num, f_exact, g_exact):
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("# Taylor-Green axis verification along x-axis\n")
        f.write("# Columns: r,f_num,g_num,f_exact,g_exact,abs_err_f,abs_err_g\n")
        for rr, fn, gn, fe, ge in zip(r, f_num, g_num, f_exact, g_exact):
            ef = abs(fn - fe)
            eg = abs(gn - ge)
            f.write(
                f"{rr:.12e},{fn:.12e},{gn:.12e},{fe:.12e},{ge:.12e},{ef:.12e},{eg:.12e}\n"
            )


def plot_tgv_verification(out_png, r, f_num, g_num, f_exact, g_exact, n, show=True):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(r, f_num, lw=2.0, label="f_num (R11 along x)")
    ax.plot(r, g_num, lw=2.0, label="g_num (R22 along x)")
    ax.plot(r, f_exact, "--", lw=1.6, label="f_exact = cos(r)")
    ax.plot(r, g_exact, ":", lw=1.8, label="g_exact = cos(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Taylor-Green verification (N={n})")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_tgv_verification(verify_n, max_r_points, out_prefix, no_plot, plot_ek):
    print(f"\n{'='*60}")
    print(f"VERIFY MODE: Taylor-Green vortex (N={verify_n})")
    print(f"{'='*60}")

    vx, vy, vz, x_coords, y_coords, z_coords, dx, dy, dz = build_taylor_green_vortex(verify_n)
    print(f"  Generated grid: {vx.shape}, dx={dx:.6e}, dy={dy:.6e}, dz={dz:.6e}")

    R, hats = compute_tensor_correlations(vx, vy, vz)
    fg = extract_f_g(R, dx, dy, dz, max_r_points)
    ke_check = compute_ke_consistency_from_tensor(vx, vy, vz)

    nmax = min(vx.shape) // 2
    if max_r_points is not None:
        nmax = min(nmax, max_r_points)
    idx = np.arange(nmax + 1, dtype=int)
    r = idx * dx

    # Requested exact test:
    # f(r) from R11 along x-axis, g(r) from R22 along x-axis.
    f_num = R["xx"][idx, 0, 0]
    g_num = R["yy"][idx, 0, 0]
    f_num = f_num / (f_num[0] if abs(f_num[0]) > 0 else 1.0)
    g_num = g_num / (g_num[0] if abs(g_num[0]) > 0 else 1.0)

    f_exact = np.cos(2.0 * np.pi * r)
    g_exact = np.cos(2.0 * np.pi * r)

    abs_err_f = np.abs(f_num - f_exact)
    abs_err_g = np.abs(g_num - g_exact)
    max_err_f = np.max(abs_err_f)
    max_err_g = np.max(abs_err_g)
    rms_err_f = np.sqrt(np.mean(abs_err_f**2))
    rms_err_g = np.sqrt(np.mean(abs_err_g**2))

    lam_L_axis, lam_T_axis, d2f_axis, d2g_axis = compute_taylor_microscales_from_curves(r, f_num, g_num)
    lam_exact = 1.0 / (2.0 * np.pi)

    lam_L_avg, lam_T_avg, d2f_avg, d2g_avg = compute_taylor_microscales_from_curves(
        fg["r"], fg["f"], fg["g"]
    )

    f_comp, f_comp_avg = compute_component_taylor_microscales_from_f(fg)
    L_comp, L_comp_avg = compute_component_integral_length_scales_from_f(fg)
    L_avg_curve = compute_integral_length_scale_from_curve(fg["r"], fg["f"])
    grad_comp = compute_taylor_microscale_gradient_spectral(vx, vy, vz, dx, dy, dz)

    tol = 5e-12
    passed = (max_err_f < tol) and (max_err_g < tol)

    print("Verification summary ([0,1) domain, exact = cos(2*pi*r)):")
    print_table(
        ["Metric", "Value"],
        [
            ["max|f_num-cos|", f"{max_err_f:.3e}"],
            ["max|g_num-cos|", f"{max_err_g:.3e}"],
            ["rms|f_num-cos|", f"{rms_err_f:.3e}"],
            ["rms|g_num-cos|", f"{rms_err_g:.3e}"],
            ["f''(0) axis", f"{d2f_axis:.8e}"],
            ["g''(0) axis", f"{d2g_axis:.8e}"],
            ["lambda_L axis", f"{lam_L_axis:.8e}"],
            ["lambda_T axis", f"{lam_T_axis:.8e}"],
            ["lambda exact", f"{lam_exact:.8e}"],
            ["f_avg''(0)", f"{d2f_avg:.8e}"],
            ["g_avg''(0)", f"{d2g_avg:.8e}"],
            ["lambda_L_avg", f"{lam_L_avg:.8e}"],
            ["lambda_T_avg", f"{lam_T_avg:.8e}"],
            ["L_int from f_avg", f"{L_avg_curve:.8e}"],
            ["L_int components avg/3", f"{L_comp_avg:.8e}"],
            ["PASS (tol=5e-12)", str(passed)],
        ],
    )

    comp_rows = []
    for tag in ("x", "y", "z"):
        lf = f_comp[tag]["lambda"]
        lg = grad_comp[tag]["lambda"]
        diff = abs(lf - lg)
        rel = (diff / abs(lg) * 100.0) if abs(lg) > 0.0 else 0.0
        comp_rows.append(
            [tag, f"{lf:.8e}", f"{lg:.8e}", f"{diff:.3e}", f"{rel:.3e}%"]
        )
    avg_diff = abs(f_comp_avg - grad_comp["avg"])
    avg_rel = (avg_diff / abs(grad_comp["avg"]) * 100.0) if abs(grad_comp["avg"]) > 0.0 else 0.0
    comp_rows.append(
        ["avg/3", f"{f_comp_avg:.8e}", f"{grad_comp['avg']:.8e}", f"{avg_diff:.3e}", f"{avg_rel:.3e}%"]
    )
    print("Componentwise longitudinal Taylor microscale comparison:")
    print_table(["Comp", "lambda_from_f", "lambda_grad", "abs_diff", "rel_diff"], comp_rows)

    int_rows = [
        ["x", f"{L_comp['x']['L']:.8e}", L_comp["x"]["status"]],
        ["y", f"{L_comp['y']['L']:.8e}", L_comp["y"]["status"]],
        ["z", f"{L_comp['z']['L']:.8e}", L_comp["z"]["status"]],
        ["avg/3", f"{L_comp_avg:.8e}", "component_avg"],
    ]
    print("Componentwise integral length scales from f_b(r):")
    print_table(["Comp", "L_int_from_f", "status"], int_rows)

    print("Total KE consistency (physical vs spectral tensor diagonal):")
    print_table(
        ["Metric", "Value"],
        [
            ["KE_phys", f"{ke_check['ke_phys']:.12e}"],
            ["KE_spec_from_Phi_diag", f"{ke_check['ke_spec_tensor_diag']:.12e}"],
            ["abs_diff", f"{ke_check['abs_diff']:.3e}"],
            ["rel_diff", f"{100.0*ke_check['rel_diff']:.3e}%"],
        ],
    )

    # FFT consistency check in physical units for [0,1)^3:
    # fundamental wavenumber should be 2*pi.
    KX, KY, KZ = make_kgrids(verify_n, verify_n, verify_n, dx, dy, dz)
    spec_u = np.abs(hats[0])**2
    peak_idx = np.unravel_index(np.argmax(spec_u), spec_u.shape)
    kx_peak = KX[peak_idx]
    ky_peak = KY[peak_idx]
    kz_peak = KZ[peak_idx]
    print("FFT consistency:")
    print_table(
        ["Quantity", "Value"],
        [
            ["k_peak_x", f"{kx_peak:.6e}"],
            ["k_peak_y", f"{ky_peak:.6e}"],
            ["k_peak_z", f"{kz_peak:.6e}"],
            ["fundamental k0=2*pi", f"{2.0*np.pi:.6e}"],
        ],
    )

    print("Verification mode: no CSV/PNG files are written.")
    if not no_plot:
        # Axis-wise exact check plot (display only)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(r, f_num, lw=2.0, label="f_num (R11 along x)")
        ax.plot(r, g_num, lw=2.0, label="g_num (R22 along x)")
        ax.plot(r, f_exact, "--", lw=1.6, label="f_exact = cos(2*pi*r)")
        ax.plot(r, g_exact, ":", lw=1.8, label="g_exact = cos(2*pi*r)")
        ax.set_xlabel("r")
        ax.set_ylabel("Correlation")
        ax.set_title(f"Taylor-Green axis verification (N={verify_n})")
        ax.grid(True, alpha=0.3, ls="--")
        ax.legend(loc="best")
        fig.tight_layout()
        plt.show()

        # Averaged + component f(r) plot (display only)
        fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 5.2))
        ax2 = axes2[0]
        ax2.plot(fg["r"], fg["f"], lw=2.2, color="black", label="f(r) averaged longitudinal")
        ax2.plot(fg["r"], fg["g"], lw=2.2, color="red", label="g(r) averaged transverse")
        ax2.set_xlabel("r")
        ax2.set_ylabel("Correlation")
        ax2.set_title("Averaged correlations from tensor")
        ax2.grid(True, alpha=0.3, ls="--")
        ax2.legend(loc="best")

        ax2c = axes2[1]
        ax2c.plot(fg["r"], fg["f_x_norm"], lw=2.0, label="R11(r,0,0)/R11(0,0,0)")
        ax2c.plot(fg["r"], fg["f_y_norm"], lw=2.0, label="R22(0,r,0)/R22(0,0,0)")
        ax2c.plot(fg["r"], fg["f_z_norm"], lw=2.0, label="R33(0,0,r)/R33(0,0,0)")
        ax2c.set_xlabel("r")
        ax2c.set_ylabel("Rii(re_i)/Rii(0,0,0)")
        ax2c.set_title("Component-wise Longitudinal Rij")
        ax2c.grid(True, alpha=0.3, ls="--")
        ax2c.legend(loc="best")
        fig2.tight_layout()
        plt.show()

    if plot_ek:
        print("Step 3 (optional): Compute E(k) from spectral tensor trace")
        k, E_k = compute_energy_spectrum_1d(hats[0], hats[1], hats[2], dx, dy, dz)
        if not no_plot:
            mask = (k > 0.0) & (E_k > 0.0)
            fig3, ax3 = plt.subplots(figsize=(8.5, 5.5))
            ax3.loglog(k[mask], E_k[mask], lw=2.0)
            ax3.set_xlabel("k")
            ax3.set_ylabel("E(k)")
            ax3.set_title("Taylor-Green E(k) (display only)")
            ax3.grid(True, which="both", alpha=0.3, ls="--")
            fig3.tight_layout()
            plt.show()

    return passed


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Compute f(r), g(r) from 3D velocity fields via FFT spectral tensor."
    )
    parser.add_argument("data_file", type=str, nargs='?',
                        help="Input text or HDF5 velocity file")
    parser.add_argument("--verify", action="store_true",
                        help="Run Taylor-Green verification test instead of reading a file")
    parser.add_argument("--verify-n", type=int, default=32,
                        help="Grid size N for Taylor-Green verification on [0,1)^3")
    parser.add_argument("--header-lines", type=int, default=None,
                        help="Number of header lines to skip/read (auto-detect if omitted)")
    parser.add_argument("--chunk-size", type=int, default=5_000_000,
                        help="Chunk size for loading velocity files")
    parser.add_argument("--decimals", type=int, default=10,
                        help="Rounding precision for coordinate matching")
    parser.add_argument("--max-r-points", type=int, default=None,
                        help="Maximum r index (default: min(nx,ny,nz)//2)")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output prefix (default: <input>_fg)")
    parser.add_argument("--plot-ek", action="store_true",
                        help="Also compute and plot optional shell-integrated E(k)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not open interactive plots (files are still saved)")
    args = parser.parse_args()

    if args.verify:
        run_tgv_verification(
            verify_n=args.verify_n,
            max_r_points=args.max_r_points,
            out_prefix=args.out_prefix,
            no_plot=args.no_plot,
            plot_ek=args.plot_ek
        )
        return

    if args.data_file is None:
        parser.error("data_file is required unless --verify is used")

    print(f"\n{'='*60}")
    print(f"ANALYZING: {args.data_file}")
    print(f"{'='*60}")

    if args.data_file.endswith('.h5'):
        header_lines = 0
    elif args.header_lines is None:
        header_lines = detect_header_lines(args.data_file)
    else:
        header_lines = args.header_lines

    step_number, time_value = read_data_file_header(args.data_file, header_lines)

    vx, vy, vz, x_coords, y_coords, z_coords, dx, dy, dz = read_data_file_chunked(
        args.data_file,
        chunk_size=args.chunk_size,
        skiprows=header_lines,
        decimals=args.decimals
    )
    print(f"  Final FFT grid shape: {vx.shape}")

    R, hats = compute_tensor_correlations(vx, vy, vz)
    fg = extract_f_g(R, dx, dy, dz, args.max_r_points)
    ke_check = compute_ke_consistency_from_tensor(vx, vy, vz)
    lam_L, lam_T, d2f0, d2g0 = compute_taylor_microscales_from_curves(fg["r"], fg["f"], fg["g"])

    f_comp, f_comp_avg = compute_component_taylor_microscales_from_f(fg)
    L_comp, L_comp_avg = compute_component_integral_length_scales_from_f(fg)
    L_avg_curve = compute_integral_length_scale_from_curve(fg["r"], fg["f"])
    grad_comp = compute_taylor_microscale_gradient_spectral(vx, vy, vz, dx, dy, dz)
    print("Taylor microscale summary:")
    print_table(
        ["Metric", "Value"],
        [
            ["f''(0) avg", f"{d2f0:.8e}"],
            ["g''(0) avg", f"{d2g0:.8e}"],
            ["lambda_L avg", f"{lam_L:.8e}"],
            ["lambda_T avg", f"{lam_T:.8e}"],
            ["L_int from f_avg", f"{L_avg_curve:.8e}"],
            ["L_int components avg/3", f"{L_comp_avg:.8e}"],
        ],
    )
    comp_rows = []
    for tag in ("x", "y", "z"):
        lf = f_comp[tag]["lambda"]
        lg = grad_comp[tag]["lambda"]
        diff = abs(lf - lg)
        rel = (diff / abs(lg) * 100.0) if abs(lg) > 0.0 else 0.0
        comp_rows.append(
            [tag, f"{lf:.8e}", f"{lg:.8e}", f"{diff:.3e}", f"{rel:.3e}%"]
        )
    avg_diff = abs(f_comp_avg - grad_comp["avg"])
    avg_rel = (avg_diff / abs(grad_comp["avg"]) * 100.0) if abs(grad_comp["avg"]) > 0.0 else 0.0
    comp_rows.append(
        ["avg/3", f"{f_comp_avg:.8e}", f"{grad_comp['avg']:.8e}", f"{avg_diff:.3e}", f"{avg_rel:.3e}%"]
    )
    print("Componentwise longitudinal Taylor microscale comparison:")
    print_table(["Comp", "lambda_from_f", "lambda_grad", "abs_diff", "rel_diff"], comp_rows)

    int_rows = [
        ["x", f"{L_comp['x']['L']:.8e}", L_comp["x"]["status"]],
        ["y", f"{L_comp['y']['L']:.8e}", L_comp["y"]["status"]],
        ["z", f"{L_comp['z']['L']:.8e}", L_comp["z"]["status"]],
        ["avg/3", f"{L_comp_avg:.8e}", "component_avg"],
    ]
    print("Componentwise integral length scales from f_b(r):")
    print_table(["Comp", "L_int_from_f", "status"], int_rows)

    print("Total KE consistency (physical vs spectral tensor diagonal):")
    print_table(
        ["Metric", "Value"],
        [
            ["KE_phys", f"{ke_check['ke_phys']:.12e}"],
            ["KE_spec_from_Phi_diag", f"{ke_check['ke_spec_tensor_diag']:.12e}"],
            ["abs_diff", f"{ke_check['abs_diff']:.3e}"],
            ["rel_diff", f"{100.0*ke_check['rel_diff']:.3e}%"],
        ],
    )

    if args.out_prefix is None:
        stem = os.path.splitext(args.data_file)[0]
        base = f"{stem}_fg"
    else:
        base = args.out_prefix

    out_csv = f"{base}.csv"
    save_fg_csv(out_csv, fg, step_number, time_value)
    print(f"Saved correlations CSV: {out_csv}")

    plot_fg(fg, step_number, time_value, show=(not args.no_plot))

    if args.plot_ek:
        print("Step 3 (optional): Compute E(k) from spectral tensor trace")
        k, E_k = compute_energy_spectrum_1d(hats[0], hats[1], hats[2], dx, dy, dz)

        out_ek_csv = f"{base}_Ek.csv"
        out_ek_png = f"{base}_Ek.png"
        np.savetxt(
            out_ek_csv,
            np.column_stack((k, E_k)),
            delimiter=",",
            header="k,E(k)",
            comments=""
        )
        print(f"Saved E(k): {out_ek_csv}")
        plot_energy_spectrum(out_ek_png, k, E_k, step_number, time_value, show=(not args.no_plot))
        print(f"Saved E(k) plot: {out_ek_png}")


if __name__ == "__main__":
    main()
