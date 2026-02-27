#!/usr/bin/env python3
"""
PDFs & JPDFs from sampled velocity fields
Analyzes turbulence statistics: u,v,w (velocity), ω (vorticity), S (strain), σ (vortex stretching)

Default normalization:
  1D: u_{1,1}/ω′, u_{1,2}/ω′, u_{1,3}/ω′, |ω|/ω′
  2D: (|ω|/ω′, |S|/ω′), (|ω|/ω′, σ/ω′²), (|S|/ω′, σ/ω′²)

How to run:
  1) Analyze a sampled velocity file:
     python3 python_scripts/compute_pdf_velocity.py <path/to/SampledDataXXXX.txt>

  2) Run manufactured-solution verification (Taylor-Green on [0,1)^3):
     python3 python_scripts/compute_pdf_velocity.py --verify --verify-n 32

  3) If your environment has OpenMP shared-memory issues, force serial threads:
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL \
     python3 python_scripts/compute_pdf_velocity.py --verify --verify-n 32

Verification pass criteria:
  - Max relative derivative/vorticity error should be very small (typically ~1e-12 or lower).
  - Divergence RMS should be near 0.
  - 1D/2D PDF integrals should be close to 1.

PDF normalization modes:
  - conditional (default):
      Use when you want the PDF/JPDF shape within a finite plotting range.
      The density is renormalized over that range, so the integral is 1 there.
  - unconditional:
      Use when you want absolute probability mass relative to all samples.
      Integrals over finite ranges are then less than 1 if tails are outside.
"""

import numpy as np
import argparse
import os
import re
import matplotlib.pyplot as plt
from scipy import fft as spfft


# ============================================================================
# I/O Functions
# ============================================================================

def read_velocity_file(filename, chunk_size=100000):
    """
    Read velocity data from file in chunks for memory efficiency.
    
    Args:
        filename: Path to velocity data file
        chunk_size: Number of rows to read at a time
        
    Returns:
        x, y, z, u, v, w arrays, step number, time
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Chunked reading requires pandas. Install pandas or run without --chunked."
        ) from exc

    # Read header to get metadata
    with open(filename, 'r') as f:
        header = [next(f) for _ in range(6)]
    
    step, time = "unknown", 0.0
    for line in header:
        if 'Cycle' in line:
            m = re.search(r'Cycle\s*[:=]\s*(\d+)', line)
            step = m.group(1) if m else step
        if 'Time' in line:
            m = re.search(r'Time\s*[:=]\s*([0-9.eE+-]+)', line)
            time = float(m.group(1)) if m else time
    
    # Read data in chunks
    print(f"Reading velocity data from {filename}...")
    chunks = []
    row_count = 0
    
    for chunk in pd.read_csv(filename, delimiter=' ', skiprows=6, 
                             chunksize=chunk_size, header=None,
                             skipinitialspace=True):
        chunks.append(chunk.values)
        row_count += len(chunk)
        if row_count % (chunk_size * 10) == 0:
            print(f"  Read {row_count:,} rows...")
    
    data = np.vstack(chunks)
    print(f"  Total: {len(data):,} points")
    
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError("Expected at least 6 columns: x y z u v w")
    
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    u, v, w = data[:, 3], data[:, 4], data[:, 5]
    
    return x, y, z, u, v, w, step, time


def read_velocity_file_simple(filename):
    """
    Simple version: read entire file at once (for smaller datasets).
    """
    with open(filename, 'r') as f:
        header = [next(f) for _ in range(6)]
    
    step, time = "unknown", 0.0
    for line in header:
        if 'Cycle' in line:
            m = re.search(r'Cycle\s*[:=]\s*(\d+)', line)
            step = m.group(1) if m else step
        if 'Time' in line:
            m = re.search(r'Time\s*[:=]\s*([0-9.eE+-]+)', line)
            time = float(m.group(1)) if m else time
    
    print(f"Reading velocity data from {filename}...")
    data = np.genfromtxt(filename, delimiter=' ', skip_header=6)
    print(f"  Loaded {len(data):,} points")
    
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError("Expected at least 6 columns: x y z u v w")
    
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    u, v, w = data[:, 3], data[:, 4], data[:, 5]
    
    return x, y, z, u, v, w, step, time


def make_regular_grid(x, y, z, u, v, w, decimals=10):
    """
    Reconstruct structured grid from scattered points.
    
    Args:
        x, y, z: Coordinate arrays
        u, v, w: Velocity component arrays
        decimals: Rounding precision for coordinate matching
        
    Returns:
        (U, V, W): 3D velocity arrays
        (X, Y, Z): Unique coordinate values
        (dx, dy, dz): Grid spacings
    """
    print("Reconstructing structured grid...")
    
    xr = np.round(x, decimals)
    yr = np.round(y, decimals)
    zr = np.round(z, decimals)
    
    X = np.unique(xr)
    Y = np.unique(yr)
    Z = np.unique(zr)
    
    nx, ny, nz = len(X), len(Y), len(Z)
    print(f"  Grid dimensions: {nx} x {ny} x {nz} = {nx*ny*nz:,} points")
    
    dx = (X[1] - X[0]) if nx > 1 else 1.0
    dy = (Y[1] - Y[0]) if ny > 1 else 1.0
    dz = (Z[1] - Z[0]) if nz > 1 else 1.0
    
    U = np.zeros((nx, ny, nz))
    V = np.zeros_like(U)
    W = np.zeros_like(U)
    
    # Create index maps
    xi = {v: i for i, v in enumerate(X)}
    yj = {v: j for j, v in enumerate(Y)}
    zk = {v: k for k, v in enumerate(Z)}
    
    # Fill grid
    for n in range(len(xr)):
        i, j, k = xi[xr[n]], yj[yr[n]], zk[zr[n]]
        U[i, j, k] = u[n]
        V[i, j, k] = v[n]
        W[i, j, k] = w[n]
    
    return (U, V, W), (X, Y, Z), (dx, dy, dz)


def make_kgrids(nx, ny, nz, dx, dy, dz):
    """
    Create wavenumber grids for spectral derivatives.
    """
    kx = 2 * np.pi * spfft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * spfft.fftfreq(ny, d=dy)
    kz = 2 * np.pi * spfft.fftfreq(nz, d=dz)
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    
    return KX, KY, KZ, K2, (K2 != 0)


# ============================================================================
# Spectral Operations
# ============================================================================

def fftn(a):
    return spfft.fftn(a)


def ifftn(a):
    return spfft.ifftn(a)


def deriv_fft(f, Kc):
    """Compute derivative in direction Kc using FFT."""
    return np.real(ifftn(1j * Kc * fftn(f)))


def curl(U, V, W, KX, KY, KZ):
    """
    Compute vorticity: ω = ∇ × u
    """
    print("Computing vorticity (curl)...")
    Uk, Vk, Wk = fftn(U), fftn(V), fftn(W)
    
    wx = np.real(ifftn(1j * (KY * Wk - KZ * Vk)))
    wy = np.real(ifftn(1j * (KZ * Uk - KX * Wk)))
    wz = np.real(ifftn(1j * (KX * Vk - KY * Uk)))
    
    return wx, wy, wz


def velocity_gradients(U, V, W, KX, KY, KZ):
    """
    Compute all velocity gradients: ∂u_i/∂x_j
    """
    print("Computing velocity gradients...")
    
    du_dx = deriv_fft(U, KX)
    du_dy = deriv_fft(U, KY)
    du_dz = deriv_fft(U, KZ)
    
    dv_dx = deriv_fft(V, KX)
    dv_dy = deriv_fft(V, KY)
    dv_dz = deriv_fft(V, KZ)
    
    dw_dx = deriv_fft(W, KX)
    dw_dy = deriv_fft(W, KY)
    dw_dz = deriv_fft(W, KZ)
    
    return np.array([[du_dx, du_dy, du_dz],
                     [dv_dx, dv_dy, dv_dz],
                     [dw_dx, dw_dy, dw_dz]])


def strain_symmetric(grad):
    """
    Compute symmetric strain rate tensor: S_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)
    and its magnitude: |S| = √(2 S_ij S_ij)
    """
    print("Computing strain rate tensor...")
    
    S = np.empty_like(grad)
    for i in range(3):
        for j in range(3):
            S[i, j] = 0.5 * (grad[i, j] + grad[j, i])
    
    # Compute magnitude
    S2 = np.zeros_like(S[0, 0])
    for i in range(3):
        for j in range(3):
            S2 += S[i, j] * S[i, j]
    
    Smag = np.sqrt(2.0 * S2)
    
    return S, Smag


def sigma_vortex_stretching(wx, wy, wz, S):
    """
    Compute vortex stretching: σ = ω_i ω_j S_ij
    (Production term in enstrophy equation)
    """
    print("Computing vortex stretching...")
    
    wv = np.array([wx, wy, wz])
    sig = np.zeros_like(wx)
    
    for i in range(3):
        acc = np.zeros_like(wx)
        for j in range(3):
            acc += wv[j] * S[i, j]
        sig += wv[i] * acc
    
    return sig


# ============================================================================
# Statistics and Plotting
# ============================================================================

def pdf_1d(values, bins=201, x_range=(-6, 6), normalization="conditional"):
    """
    Compute 1D probability density function.

    normalization:
      - conditional: normalize by in-range samples only
      - unconditional: normalize by all samples (tails outside range reduce mass)
    """
    values = values.ravel()
    counts, edges = np.histogram(values, bins=bins, range=x_range, density=False)
    dx = np.diff(edges)

    if normalization == "conditional":
        denom = counts.sum()
    elif normalization == "unconditional":
        denom = values.size
    else:
        raise ValueError("normalization must be 'conditional' or 'unconditional'")

    if denom <= 0:
        hist = np.zeros_like(counts, dtype=float)
    else:
        hist = counts / (denom * dx)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def draw_pdf_1d(name, centers, pdf):
    """
    Plot 1D PDF on log scale.
    """
    plt.figure(figsize=(6, 4))
    plt.semilogy(centers, pdf, 'k-', linewidth=2)
    plt.xlabel(name, fontsize=11)
    plt.ylabel("PDF", fontsize=11)
    plt.title(f"PDF: {name}", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()


def joint_pdf_2d(x, y, bins=200, range_x=None, range_y=None, normalization="conditional"):
    """
    Compute 2D joint probability density function.

    normalization:
      - conditional: normalize by in-range sample pairs only
      - unconditional: normalize by all sample pairs
    """
    x = x.ravel()
    y = y.ravel()
    counts, xe, ye = np.histogram2d(x, y, bins=bins, range=[range_x, range_y], density=False)
    dx = np.diff(xe)[:, None]
    dy = np.diff(ye)[None, :]

    if normalization == "conditional":
        denom = counts.sum()
    elif normalization == "unconditional":
        denom = x.size
    else:
        raise ValueError("normalization must be 'conditional' or 'unconditional'")

    if denom <= 0:
        H = np.zeros_like(counts, dtype=float)
    else:
        H = counts / (denom * dx * dy)

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H.T, xc, yc


def draw_joint_contours(name_x, name_y, x, y, 
                       levels=(1e-6, 1e-5, 1e-4, 1e-3),
                       bins=200, rng=(-6, 6), same_scale=True,
                       pdf_normalization="conditional"):
    """
    Plot joint PDF as contours.
    
    Note: Axes are swapped - what you pass as (name_x, x) appears on y-axis
    """
    # Permanent swap: what used to be Y now on X
    name_x, name_y = name_y, name_x
    x, y = y, x
    
    H, xc, yc = joint_pdf_2d(
        x, y, bins=bins, range_x=rng,
        range_y=rng if same_scale else None,
        normalization=pdf_normalization
    )
    
    levels = np.sort(np.array(levels, float))
    Hmax = float(np.nanmax(H)) if H.size else 0.0
    
    # Only filter out levels that are above Hmax
    levels = levels[levels < Hmax]
    
    if levels.size == 0:
        # Fallback to image plot if no valid contour levels
        plt.figure(figsize=(6, 5))
        plt.imshow(H.T, origin="lower", 
                  extent=(xc[0], xc[-1], yc[0], yc[-1]), aspect="auto")
        plt.colorbar(label="PDF")
        plt.xlabel(name_x, fontsize=11)
        plt.ylabel(name_y, fontsize=11)
        plt.title(f"Joint PDF ({name_x} vs {name_y})", fontsize=12)
        plt.tight_layout()
        return
    
    plt.figure(figsize=(6, 5))
    cs = plt.contour(xc, yc, H, levels=levels, colors='black', linewidths=1.8)
    plt.clabel(cs, fmt="%.0e", inline=True, fontsize=8)
    plt.xlabel(name_x, fontsize=11)
    plt.ylabel(name_y, fontsize=11)
    plt.title(f"Joint PDF contours ({name_x} vs {name_y})", fontsize=12)
    plt.grid(True, ls="--", alpha=0.25)
    plt.tight_layout()


def stats(name, a):
    """
    Print detailed statistics for a field.
    """
    a = a.ravel()
    p = np.percentile(a, [0.1, 1, 5, 50, 95, 99, 99.9])
    rms = np.sqrt(np.mean(a * a))
    
    print(f"{name:>12s}: "
          f"min={a.min():+.3e}  max={a.max():+.3e}  rms={rms:.3e}")
    print(f"              "
          f"p[0.1, 1, 5, 50, 95, 99, 99.9] = {p}")


# ============================================================================
# Verification
# ============================================================================

def relative_l2_error(numerical, exact, eps=1e-30):
    """
    Relative L2 error ||num - exact||_2 / ||exact||_2.
    """
    diff = np.ravel(numerical - exact)
    ref = np.ravel(exact)
    return np.linalg.norm(diff) / (np.linalg.norm(ref) + eps)


def run_taylor_green_verification(args):
    """
    Manufactured-solution verification using Taylor-Green initial condition
    on [0,1)^3 with 2*pi factors in trigonometric arguments.
    """
    n = int(args.verify_n)
    if n < 4:
        raise ValueError("--verify-n must be at least 4")

    print("\n" + "=" * 70)
    print("VERIFY MODE: Taylor-Green manufactured solution on [0,1)^3")
    print("u =  sin(2πx) cos(2πy) cos(2πz)")
    print("v = -cos(2πx) sin(2πy) cos(2πz)")
    print("w =  0")
    print("=" * 70)

    # Periodic mesh without duplicated endpoint
    x1d = np.linspace(0.0, 1.0, n, endpoint=False)
    y1d = np.linspace(0.0, 1.0, n, endpoint=False)
    z1d = np.linspace(0.0, 1.0, n, endpoint=False)
    Xg, Yg, Zg = np.meshgrid(x1d, y1d, z1d, indexing='ij')

    twopi = 2.0 * np.pi
    sx = np.sin(twopi * Xg)
    cx = np.cos(twopi * Xg)
    sy = np.sin(twopi * Yg)
    cy = np.cos(twopi * Yg)
    sz = np.sin(twopi * Zg)
    cz = np.cos(twopi * Zg)

    U = sx * cy * cz
    V = -cx * sy * cz
    W = np.zeros_like(U)

    dx = dy = dz = 1.0 / n
    KX, KY, KZ, _, _ = make_kgrids(n, n, n, dx, dy, dz)

    # Numerical derivatives/curl from FFT
    grad = velocity_gradients(U, V, W, KX, KY, KZ)
    wx, wy, wz = curl(U, V, W, KX, KY, KZ)

    # Exact derivatives
    du_dx_e = twopi * cx * cy * cz
    du_dy_e = -twopi * sx * sy * cz
    du_dz_e = -twopi * sx * cy * sz

    dv_dx_e = twopi * sx * sy * cz
    dv_dy_e = -twopi * cx * cy * cz
    dv_dz_e = twopi * cx * sy * sz

    dw_dx_e = np.zeros_like(U)
    dw_dy_e = np.zeros_like(U)
    dw_dz_e = np.zeros_like(U)

    wx_e = -dv_dz_e
    wy_e = du_dz_e
    wz_e = dv_dx_e - du_dy_e

    checks = [
        ("du/dx", grad[0, 0], du_dx_e),
        ("du/dy", grad[0, 1], du_dy_e),
        ("du/dz", grad[0, 2], du_dz_e),
        ("dv/dx", grad[1, 0], dv_dx_e),
        ("dv/dy", grad[1, 1], dv_dy_e),
        ("dv/dz", grad[1, 2], dv_dz_e),
        ("dw/dx", grad[2, 0], dw_dx_e),
        ("dw/dy", grad[2, 1], dw_dy_e),
        ("dw/dz", grad[2, 2], dw_dz_e),
        ("omega_x", wx, wx_e),
        ("omega_y", wy, wy_e),
        ("omega_z", wz, wz_e),
    ]

    print(f"\nRelative L2 errors (N={n}):")
    max_rel = 0.0
    for name, num, exact in checks:
        rel = relative_l2_error(num, exact)
        max_rel = max(max_rel, rel)
        print(f"  {name:8s}: {rel:.3e}")

    div = grad[0, 0] + grad[1, 1] + grad[2, 2]
    div_rms = np.sqrt(np.mean(div * div))
    print(f"\nDivergence RMS (should be ~0): {div_rms:.3e}")
    print(f"Max relative derivative/vorticity error: {max_rel:.3e}")

    # PDF/JPDF normalization checks on computed quantities
    wmag = np.sqrt(wx**2 + wy**2 + wz**2)
    S, Smag = strain_symmetric(grad)
    sigma = sigma_vortex_stretching(wx, wy, wz, S)
    u11, u12, u13 = grad[0, 0], grad[0, 1], grad[0, 2]
    omega_prime = np.sqrt(np.mean(wmag**2) + 1e-30)

    x_rng = (-args.range_sigma, args.range_sigma)
    print(f"\n1D PDF mass checks in range [{x_rng[0]}, {x_rng[1]}]:")
    print(f"PDF normalization mode: {args.pdf_normalization}")
    one_d_vars = [
        ("u11/ω′", u11 / omega_prime),
        ("u12/ω′", u12 / omega_prime),
        ("u13/ω′", u13 / omega_prime),
        ("|ω|/ω′", wmag / omega_prime),
    ]

    for name, arr in one_d_vars:
        centers, pdf = pdf_1d(
            arr, bins=args.bins, x_range=x_rng,
            normalization=args.pdf_normalization
        )
        dx_pdf = centers[1] - centers[0] if len(centers) > 1 else 0.0
        mass = np.sum(pdf) * dx_pdf
        frac_in = np.mean((arr >= x_rng[0]) & (arr <= x_rng[1]))
        expected = 1.0 if args.pdf_normalization == "conditional" else frac_in
        print(
            f"  {name:8s}: integral={mass:.6f}, expected={expected:.6f}, "
            f"in-range={frac_in:.6f}"
        )

    if args.sigma_norm == "omega_prime":
        Xsig = sigma / omega_prime**2
        sig_label = "σ/ω′²"
    elif args.sigma_norm == "omega_prime_times_absw":
        Xsig = sigma / (omega_prime * (np.abs(wmag) + 1e-30))
        sig_label = "σ/(|ω|ω′)"
    else:
        Xsig = sigma / (omega_prime * (Smag + 1e-30))
        sig_label = "σ/(|S|ω′)"

    Xw = wmag / omega_prime
    XS = Smag / omega_prime

    print("\n2D joint PDF mass checks:")
    two_d_pairs = [
        ("|ω|/ω′ vs |S|/ω′", Xw, XS),
        (f"|ω|/ω′ vs {sig_label}", Xw, Xsig),
        (f"|S|/ω′ vs {sig_label}", XS, Xsig),
    ]

    for name, xa, ya in two_d_pairs:
        H, xc, yc = joint_pdf_2d(
            xa, ya, bins=args.bins, range_x=x_rng, range_y=x_rng,
            normalization=args.pdf_normalization
        )
        dx_pdf = xc[1] - xc[0] if len(xc) > 1 else 0.0
        dy_pdf = yc[1] - yc[0] if len(yc) > 1 else 0.0
        mass = np.sum(H) * dx_pdf * dy_pdf
        frac_in = np.mean(
            (xa >= x_rng[0]) & (xa <= x_rng[1]) &
            (ya >= x_rng[0]) & (ya <= x_rng[1])
        )
        expected = 1.0 if args.pdf_normalization == "conditional" else frac_in
        print(
            f"  {name:22s}: integral={mass:.6f}, expected={expected:.6f}, "
            f"min(H)={np.min(H):.3e}"
        )

    print("\nVerification run complete.")
    print("=" * 70 + "\n")


# ============================================================================
# Main Program
# ============================================================================

def main():
    # Parse arguments
    epilog = (
        "Examples:\n"
        "  Analyze data file:\n"
        "    python3 python_scripts/compute_pdf_velocity.py "
        "old/SamplePointsVelocity_Re400NumPtsPerDir8RefLv2P4/cycle_9599/SampledData9599.txt\n\n"
        "  Run verification:\n"
        "    python3 python_scripts/compute_pdf_velocity.py --verify --verify-n 32\n\n"
        "  Serial/thread-limited run (for OpenMP SHM issues):\n"
        "    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL "
        "python3 python_scripts/compute_pdf_velocity.py --verify --verify-n 32\n\n"
        "PDF normalization mode:\n"
        "  --pdf-normalization conditional   (default, compare shapes within plotting range)\n"
        "  --pdf-normalization unconditional (preserve absolute mass; tails outside range remain excluded)"
    )
    ap = argparse.ArgumentParser(
        description="Compute and plot PDFs/JPDFs of turbulence statistics with ω′ scaling",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("data_file", nargs="?", type=str,
                   help="Velocity data file (not required with --verify)")
    ap.add_argument("--bins", type=int, default=201,
                   help="Number of bins for histograms")
    ap.add_argument("--range-sigma", type=float, default=6.0,
                   help="Range for normalized variables: [-range, +range]")
    ap.add_argument("--pdf-normalization",
                   choices=["conditional", "unconditional"],
                   default="conditional",
                   help=("PDF/JPDF normalization over --range-sigma: "
                         "'conditional' renormalizes in-range mass to 1; "
                         "'unconditional' preserves absolute in-range mass"))
    ap.add_argument("--sigma-norm", 
                   choices=["omega_prime", "omega_prime_times_absw", 
                           "omega_prime_times_S"],
                   default="omega_prime",
                   help="Normalization for σ: /ω′² (default), /(|ω| ω′), or /(|S| ω′)")
    ap.add_argument("--save", action="store_true",
                   help="Save figures to PNG files")
    ap.add_argument("--chunked", action="store_true",
                   help="Use chunked reading for large files (requires pandas)")
    ap.add_argument("--verify", action="store_true",
                   help="Run Taylor-Green manufactured-solution verification and exit")
    ap.add_argument("--verify-n", type=int, default=32,
                   help="Grid points per dimension for --verify (default: 32)")
    
    args = ap.parse_args()

    if args.verify:
        run_taylor_green_verification(args)
        return

    if not args.data_file:
        ap.error("data_file is required unless --verify is used")
    
    # Read velocity data
    if args.chunked:
        try:
            x, y, z, u, v, w, step, time = read_velocity_file(args.data_file)
        except ImportError:
            print("Warning: pandas not available, falling back to simple reading")
            x, y, z, u, v, w, step, time = read_velocity_file_simple(args.data_file)
    else:
        x, y, z, u, v, w, step, time = read_velocity_file_simple(args.data_file)
    
    print(f"Step: {step}, Time: {time:.6f}")
    
    # Reconstruct structured grid
    (U, V, W), (Xc, Yc, Zc), (dx, dy, dz) = make_regular_grid(x, y, z, u, v, w)
    nx, ny, nz = len(Xc), len(Yc), len(Zc)
    
    # Create wavenumber grids for spectral operations
    KX, KY, KZ, K2, nzmask = make_kgrids(nx, ny, nz, dx, dy, dz)
    
    # Compute derived fields
    wx, wy, wz = curl(U, V, W, KX, KY, KZ)
    wmag = np.sqrt(wx**2 + wy**2 + wz**2)
    
    grad = velocity_gradients(U, V, W, KX, KY, KZ)
    S, Smag = strain_symmetric(grad)
    sigma = sigma_vortex_stretching(wx, wy, wz, S)
    
    # Extract velocity gradient components
    u11, u12, u13 = grad[0, 0], grad[0, 1], grad[0, 2]
    
    # Compute normalization scale: ω′ = RMS vorticity
    omega_prime = np.sqrt(np.mean(wmag**2) + 1e-30)
    
    # Print diagnostics
    print("\n" + "="*70)
    print("FIELD STATISTICS (unnormalized)")
    print("="*70)
    stats("|ω|", wmag)
    stats("|S|", Smag)
    stats("σ", sigma)
    print(f"\nomega_prime (ω′) = {omega_prime:.6e}")
    print("="*70 + "\n")
    
    # ========================================================================
    # 1D PDFs
    # ========================================================================
    
    print("Generating 1D PDFs...")
    x_rng = (-args.range_sigma, args.range_sigma)
    
    pdf_vars = [
        ("u_{1,1} / ω′", u11 / omega_prime),
        ("u_{1,2} / ω′", u12 / omega_prime),
        ("u_{1,3} / ω′", u13 / omega_prime),
        ("|ω| / ω′",     wmag / omega_prime)
    ]
    
    for name, arr in pdf_vars:
        c, p = pdf_1d(arr, args.bins, x_rng, normalization=args.pdf_normalization)
        draw_pdf_1d(name, c, p)
    
    # ========================================================================
    # 2D Joint PDFs with different σ normalizations
    # ========================================================================
    
    print("Generating 2D joint PDFs...")
    
    # Choose σ normalization
    if args.sigma_norm == "omega_prime":
        Xsig = sigma / omega_prime**2
        sig_label = "σ / ω′²"
    elif args.sigma_norm == "omega_prime_times_absw":
        Xsig = sigma / (omega_prime * (np.abs(wmag) + 1e-30))
        sig_label = "σ / (|ω| ω′)"
    else:  # omega_prime_times_S
        Xsig = sigma / (omega_prime * (Smag + 1e-30))
        sig_label = "σ / (|S| ω′)"
    
    Xw = wmag / omega_prime
    XS = Smag / omega_prime
    
    # Generate joint PDFs
    contour_levels = (1e-6, 1e-5, 1e-4, 1e-3)
    
    draw_joint_contours("|ω| / ω′", "|S| / ω′", Xw, XS,
                       levels=contour_levels, bins=200, rng=x_rng,
                       pdf_normalization=args.pdf_normalization)
    
    draw_joint_contours("|ω| / ω′", sig_label, Xw, Xsig,
                       levels=contour_levels, bins=200, rng=x_rng,
                       pdf_normalization=args.pdf_normalization)
    
    draw_joint_contours("|S| / ω′", sig_label, XS, Xsig,
                       levels=contour_levels, bins=200, rng=x_rng,
                       pdf_normalization=args.pdf_normalization)
    
    # ========================================================================
    # Save and display
    # ========================================================================
    
    if args.save:
        outdir = os.path.dirname(os.path.abspath(args.data_file))
        print(f"\nSaving figures to {outdir}...")
        
        for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
            path = os.path.join(outdir, f"pdf_fig_{i:02d}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")
    
    print("\nDisplaying figures...")
    plt.show()


if __name__ == "__main__":
    main()
