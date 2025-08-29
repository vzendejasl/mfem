#!/usr/bin/env python3
"""
Serial reader for MFEM-sampled velocity files.

- Reads header (Step, Time)
- Loads data in chunks (x, y, z, ux, uy, uz)
- Reconstructs a structured grid from unique coords
- Computes TKE in physical space
- Plots 3D scatter of |u| with optional subsampling

Usage:
    python ModifiedScript_serial.py /path/to/SampledDataXXXX.txt
    python ModifiedScript_serial.py data.txt --round 10 --subsample 5 --no-show
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys
from math import ceil

def plot_face_slices_2d(xu, yu, zu, Vmag, title="", subsample=1, show=True, save=None, cmap="viridis"):
    """
    Plot six 2D heatmaps (no 3D projection), one for each boundary face:
      - x = xmin, x = xmax  -> (y,z) plane
      - y = ymin, y = ymax  -> (x,z) plane
      - z = zmin, z = zmax  -> (x,y) plane

    xu, yu, zu: 1D arrays of unique coordinates (increasing)
    Vmag: 3D array [nx, ny, nz]
    subsample: take every k-th point along each axis for speed
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k = max(1, int(subsample))
    xu_s, yu_s, zu_s = xu[::k], yu[::k], zu[::k]
    V = Vmag[::k, ::k, ::k]

    # helpers: imshow expects array shape (Ny, Nx). We also set correct axis extents.
    def imshow_plane(ax, A, xcoords, ycoords, xlabel, ylabel, title_):
        # A is 2D with shape (len(ycoords), len(xcoords))
        # extent = [xmin, xmax, ymin, ymax]
        extent = [float(xcoords[0]), float(xcoords[-1]), float(ycoords[0]), float(ycoords[-1])]
        im = ax.imshow(A, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_, fontsize=10)
        return im

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes.ravel()

    # x = xmin, xmax -> planes in (y,z): V[0, :, :], V[-1, :, :]
    A = V[0, :, :].T     # (nz, ny) -> want y as x-axis, z as y-axis -> transpose to (len(z), len(y))
    im1 = imshow_plane(ax[0], A, yu_s, zu_s, "y", "z", "x = xmin")

    A = V[-1, :, :].T
    im2 = imshow_plane(ax[1], A, yu_s, zu_s, "y", "z", "x = xmax")

    # y = ymin, ymax -> planes in (x,z): V[:, 0, :], V[:, -1, :]
    A = V[:, 0, :].T     # (nz, nx) -> x as x-axis, z as y-axis
    im3 = imshow_plane(ax[2], A, xu_s, zu_s, "x", "z", "y = ymin")

    A = V[:, -1, :].T
    im4 = imshow_plane(ax[3], A, xu_s, zu_s, "x", "z", "y = ymax")

    # z = zmin, zmax -> planes in (x,y): V[:, :, 0], V[:, :, -1]
    A = V[:, :, 0].T     # (ny, nx) -> x as x-axis, y as y-axis
    im5 = imshow_plane(ax[4], A, xu_s, yu_s, "x", "y", "z = zmin")

    A = V[:, :, -1].T
    im6 = imshow_plane(ax[5], A, xu_s, yu_s, "x", "y", "z = zmax")

    # single colorbar aligned to the last subplot
    cbar = fig.colorbar(im6, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("|u|")

    fig.suptitle(title + "\nBoundary face slices (2D)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig(save, dpi=150)
        print(f"[info] Saved face-slices figure to: {save}")
    if show:
        plt.show()
    plt.close(fig)


def parse_header(path):
    """Read first ~11 lines, extract Step and Time."""
    with open(path, "r") as f:
        header_lines = [next(f) for _ in range(6)]
    step = "Unknown"
    time = 0.0
    for line in header_lines:
        m = re.search(r"Step\s*=\s*(\d+)", line)
        if m: step = m.group(1)
        m = re.search(r"Time\s*=\s*([0-9.eE+\-]+)", line)
        if m: time = float(m.group(1))
    return step, time, len(header_lines)

def load_columns_chunked(path, skiprows, chunksize=1_000_000):
    """Read six columns with pandas in chunks, return concatenated numpy arrays."""
    cols = [0, 1, 2, 3, 4, 5]
    it = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        usecols=cols,
        skiprows=skiprows,
        chunksize=chunksize,
        dtype=float,
        engine="c",
    )
    xs, ys, zs, ux, uy, uz = [], [], [], [], [], []
    for chunk in it:
        xs.append(chunk.iloc[:, 0].values)
        ys.append(chunk.iloc[:, 1].values)
        zs.append(chunk.iloc[:, 2].values)
        ux.append(chunk.iloc[:, 3].values)
        uy.append(chunk.iloc[:, 4].values)
        uz.append(chunk.iloc[:, 5].values)
    x = np.concatenate(xs) if xs else np.array([], dtype=float)
    y = np.concatenate(ys) if ys else np.array([], dtype=float)
    z = np.concatenate(zs) if zs else np.array([], dtype=float)
    vx = np.concatenate(ux) if ux else np.array([], dtype=float)
    vy = np.concatenate(uy) if uy else np.array([], dtype=float)
    vz = np.concatenate(uz) if uz else np.array([], dtype=float)
    return x, y, z, vx, vy, vz

def reconstruct_grid(x, y, z, vx, vy, vz, decimals=10):
    """
    Given flat (x,y,z) and velocities, reconstruct regular 3D arrays.
    Rounds coordinates to 'decimals' to eliminate tiny FP jitter.
    """
    if x.size == 0:
        raise ValueError("No data rows found in file.")
    xr = np.round(x, decimals=decimals)
    yr = np.round(y, decimals=decimals)
    zr = np.round(z, decimals=decimals)

    x_unique = np.unique(xr)
    y_unique = np.unique(yr)
    z_unique = np.unique(zr)
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

    # sanity
    expected = nx * ny * nz
    if expected != x.size:
        print(f"[warn] expected {expected} points from unique coords, found {x.size}. "
              "Data may be incomplete or irregular; proceeding with map fill.")

    # maps for fast indexing
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}

    # allocate
    Vx = np.full((nx, ny, nz), np.nan, dtype=float)
    Vy = np.full((nx, ny, nz), np.nan, dtype=float)
    Vz = np.full((nx, ny, nz), np.nan, dtype=float)

    # fill
    for i in range(x.size):
        ix = x_idx[xr[i]]
        iy = y_idx[yr[i]]
        iz = z_idx[zr[i]]
        Vx[ix, iy, iz] = vx[i]
        Vy[ix, iy, iz] = vy[i]
        Vz[ix, iy, iz] = vz[i]

    # replace any holes with 0 (or choose another policy)
    Vx = np.nan_to_num(Vx)
    Vy = np.nan_to_num(Vy)
    Vz = np.nan_to_num(Vz)

    return x_unique, y_unique, z_unique, Vx, Vy, Vz

def compute_tke(Vx, Vy, Vz):
    """TKE per unit mass: 0.5 * ⟨|u|^2⟩"""
    return 0.5 * np.mean(Vx*Vx + Vy*Vy + Vz*Vz)

def scatter_plot_mag(xu, yu, zu, Vmag, title="", subsample=1, show=True, save=None):
    """
    3D scatter of |u| on the structured grid.
    subsample > 1 will thin points for speed (take every k-th along each axis).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

    if subsample < 1:
        subsample = 1

    X, Y, Z = np.meshgrid(xu, yu, zu, indexing="ij")
    Xs = X[::subsample, ::subsample, ::subsample].ravel()
    Ys = Y[::subsample, ::subsample, ::subsample].ravel()
    Zs = Z[::subsample, ::subsample, ::subsample].ravel()
    Ms = Vmag[::subsample, ::subsample, ::subsample].ravel()

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Xs, Ys, Zs, c=Ms, marker=".", cmap="viridis")
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.06)
    cb.set_label("|u|")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
        print(f"[info] Saved figure to: {save}")
    if show:
        plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Serial TKE + |u| plot from MFEM-sampled text.")
    ap.add_argument("data_file", type=str, help="Path to SampledDataXXXX.txt")
    ap.add_argument("--round", type=int, default=10, help="Rounding decimals for coords (default: 10)")
    ap.add_argument("--chunksize", type=int, default=1_000_000, help="Pandas chunk size (rows)")
    ap.add_argument("--subsample", type=int, default=1, help="Plot every k-th point per axis (>=1)")
    ap.add_argument("--no-show", action="store_true", help="Do not display the plot")
    ap.add_argument("--save", type=str, default=None, help="Optional path to save the plot as PNG")
    args = ap.parse_args()

    path = args.data_file
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    step, time_val, skiprows = parse_header(path)
    print(f"[info] Step: {step}, Time: {time_val:.6e}")
    print(f"[info] Reading data (chunk size = {args.chunksize:,}) ...")

    x, y, z, vx, vy, vz = load_columns_chunked(path, skiprows=skiprows, chunksize=args.chunksize)
    print(f"[info] Loaded {x.size:,} points.")

    xu, yu, zu, Vx, Vy, Vz = reconstruct_grid(x, y, z, vx, vy, vz, decimals=args.round)
    print(f"[info] Grid shape: ({len(xu)}, {len(yu)}, {len(zu)}) = {len(xu)*len(yu)*len(zu):,} pts")

    # Domain bounds
    print("[bounds] x  ∈ [{:.6e}, {:.6e}]".format(x.min(), x.max()))
    print("[bounds] y  ∈ [{:.6e}, {:.6e}]".format(y.min(), y.max()))
    print("[bounds] z  ∈ [{:.6e}, {:.6e}]".format(z.min(), z.max()))
       
    print("[bounds] xu  ∈ [{:.6e}, {:.6e}]".format(xu.min(), xu.max()))
    print("[bounds] yu  ∈ [{:.6e}, {:.6e}]".format(yu.min(), yu.max()))
    print("[bounds] zu  ∈ [{:.6e}, {:.6e}]".format(zu.min(), zu.max()))

    # TKE cross-check (flat vs grid)
    tke_grid = compute_tke(Vx, Vy, Vz)
    tke_flat = 0.5 * np.mean(vx*vx + vy*vy + vz*vz)
    print(f"[info] TKE (flat) = {tke_flat:.8f}")
    print(f"[info] TKE (grid) = {tke_grid:.8f}")

    Vmag = np.sqrt(Vx*Vx + Vy*Vy + Vz*Vz)

    title = f"Velocity Magnitude |u|  —  TKE={tke_grid:.6f}  Step={step}  Time={time_val:.3e}"

    # 2D faces (no 3D projection)
    plot_face_slices_2d(
        xu, yu, zu, Vmag,
        title=title,
        subsample=args.subsample,           # same CLI knob
        show=(not args.no_show),
        save=(args.save.replace(".png", "_faces2d.png") if args.save else None)
    )
    
    # (Optional) keep the old full 3D scatter too
    # scatter_plot_mag(xu, yu, zu, Vmag, title=title, subsample=args.subsample, show=(not args.no_show), save=args.save)


    title = f"Velocity Magnitude |u|  —  TKE={tke_grid:.6f}  Step={step}  Time={time_val:.3e}"
    scatter_plot_mag(
        xu, yu, zu, Vmag,
        title=title,
        subsample=args.subsample,
        show=(not args.no_show),
        save=args.save
    )

if __name__ == "__main__":
    main()
