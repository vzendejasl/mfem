#!/usr/bin/env python3
"""
Visualize MFEM-sampled velocity from either text (.txt) or HDF5 (.h5) files.

- For .txt (your current format): reads header (step,time) + six columns.
- For .h5: reads 2D dataset (default name: 'samples'), shape either [N,6] or [6,N].
  * Auto-detects layout unless --layout is specified.
  * Tries file attrs 'step' (int) and 'time' (double); falls back to parsing filename.

Outputs:
- TKE and bounds
- 2D face-slice heatmaps of |u|
- Optional 3D scatter of |u|

Usage:
    python plot_mfem_samples.py /path/to/SampledDataXXXX.txt
    python plot_mfem_samples.py /path/to/SampledDataXXXX.h5
    python plot_mfem_samples.py data.h5 --dataset samples --layout auto --round 10 --subsample 5 --no-show
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

# -- Optional dependency: pandas (only used for text reading)
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# -- Optional dependency: h5py (only used for HDF5 reading)
try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False


def plot_face_slices_2d(xu, yu, zu, Vmag, title="", subsample=1, show=True, save=None, cmap="viridis"):
    k = max(1, int(subsample))
    xu_s, yu_s, zu_s = xu[::k], yu[::k], zu[::k]
    V = Vmag[::k, ::k, ::k]

    def imshow_plane(ax, A, xcoords, ycoords, xlabel, ylabel, title_):
        extent = [float(xcoords[0]), float(xcoords[-1]), float(ycoords[0]), float(ycoords[-1])]
        im = ax.imshow(A, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_, fontsize=10)
        return im

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes.ravel()

    # x = xmin/xmax -> (y,z)
    imshow_plane(ax[0], V[0, :, :].T,  yu_s, zu_s, "y", "z", "x = xmin")
    im = imshow_plane(ax[1], V[-1, :, :].T, yu_s, zu_s, "y", "z", "x = xmax")

    # y = ymin/ymax -> (x,z)
    imshow_plane(ax[2], V[:, 0, :].T,  xu_s, zu_s, "x", "z", "y = ymin")
    imshow_plane(ax[3], V[:, -1, :].T, xu_s, zu_s, "x", "z", "y = ymax")

    # z = zmin/zmax -> (x,y)
    imshow_plane(ax[4], V[:, :, 0].T,  xu_s, yu_s, "x", "y", "z = zmin")
    imshow_plane(ax[5], V[:, :, -1].T, xu_s, yu_s, "x", "y", "z = zmax")

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("|u|")
    fig.suptitle(title + "\nBoundary face slices (2D)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        root, ext = os.path.splitext(save)
        save_path = root + "_faces2d.png"
        plt.savefig(save_path, dpi=150)
        print(f"[info] Saved face-slices figure: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_header_text(path):
    with open(path, "r") as f:
        header_lines = [next(f) for _ in range(6)]
    step = "Unknown"
    time = np.nan
    for line in header_lines:
        m = re.search(r"Step\s*=\s*(\d+)", line)
        if m: step = m.group(1)
        m = re.search(r"Time\s*=\s*([0-9.eE+\-]+)", line)
        if m:
            try:
                time = float(m.group(1))
            except Exception:
                pass
    return step, time, len(header_lines)


def load_text_columns(path, skiprows, chunksize=1_000_000):
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for reading text; pip install pandas")
    cols = [0, 1, 2, 3, 4, 5]
    it = pd.read_csv(path, delim_whitespace=True, header=None,
                     usecols=cols, skiprows=skiprows,
                     chunksize=chunksize, dtype=float, engine="c")
    xs, ys, zs, ux, uy, uz = [], [], [], [], [], []
    for chunk in it:
        xs.append(chunk.iloc[:, 0].values)
        ys.append(chunk.iloc[:, 1].values)
        zs.append(chunk.iloc[:, 2].values)
        ux.append(chunk.iloc[:, 3].values)
        uy.append(chunk.iloc[:, 4].values)
        uz.append(chunk.iloc[:, 5].values)
    x = np.concatenate(xs) if xs else np.empty(0, float)
    y = np.concatenate(ys) if ys else np.empty(0, float)
    z = np.concatenate(zs) if zs else np.empty(0, float)
    vx = np.concatenate(ux) if ux else np.empty(0, float)
    vy = np.concatenate(uy) if uy else np.empty(0, float)
    vz = np.concatenate(uz) if uz else np.empty(0, float)
    return x, y, z, vx, vy, vz


def load_h5(path, dataset="samples", layout="auto", block_rows=1_000_000):
    """
    Load HDF5 dataset as flat columns x,y,z,ux,uy,uz.

    layout: 'auto' (infer), 'row' ([N,6] on disk), or 'col' ([6,N] on disk).
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py is required for HDF5; pip install h5py")

    with h5py.File(path, "r") as f:
        # Try user-provided dataset name; otherwise pick first 2D dataset.
        dset = None
        if dataset in f and isinstance(f[dataset], h5py.Dataset) and f[dataset].ndim == 2:
            dset = f[dataset]
        else:
            # find first 2D dataset
            def pick_first_2d(name, obj):
                nonlocal dset
                if dset is None and isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                    dset = obj
            f.visititems(pick_first_2d)
            if dset is None:
                raise ValueError("No 2D dataset found in file; specify --dataset")

        sh = dset.shape
        # Infer layout:
        if layout == "auto":
            if sh[1] == 6:
                layout_eff = "row"
            elif sh[0] == 6:
                layout_eff = "col"
            else:
                # last resort: small sample heuristic
                raise ValueError(f"Cannot infer layout from dataset shape {sh}. "
                                 "Use --layout row or --layout col.")
        else:
            layout_eff = layout

        # Read dataset -> flat columns
        if layout_eff == "row":
            # dset shape [N,6]
            N = sh[0]
            xs, ys, zs, ux, uy, uz = [], [], [], [], [], []
            for start in range(0, N, block_rows):
                end = min(N, start + block_rows)
                block = dset[start:end, :]     # (B,6)
                xs.append(block[:, 0])
                ys.append(block[:, 1])
                zs.append(block[:, 2])
                ux.append(block[:, 3])
                uy.append(block[:, 4])
                uz.append(block[:, 5])
            x = np.concatenate(xs) if xs else np.empty(0, float)
            y = np.concatenate(ys) if ys else np.empty(0, float)
            z = np.concatenate(zs) if zs else np.empty(0, float)
            vx = np.concatenate(ux) if ux else np.empty(0, float)
            vy = np.concatenate(uy) if uy else np.empty(0, float)
            vz = np.concatenate(uz) if uz else np.empty(0, float)
        else:
            # 'col' -> dset shape [6,N]
            N = sh[1]
            x = dset[0, :].astype(float)
            y = dset[1, :].astype(float)
            z = dset[2, :].astype(float)
            vx = dset[3, :].astype(float)
            vy = dset[4, :].astype(float)
            vz = dset[5, :].astype(float)

        # Step/time from attrs if present
        step = f.attrs.get("step", "Unknown")
        time_val = f.attrs.get("time", np.nan)
        # cast HDF5 scalars
        if isinstance(step, np.ndarray) and step.shape == ():
            step = int(step)
        if isinstance(time_val, np.ndarray) and time_val.shape == ():
            time_val = float(time_val)

        return step, time_val, x, y, z, vx, vy, vz, dset.name, tuple(sh), layout_eff




def reconstruct_grid(x, y, z, vx, vy, vz, decimals=10):
    if x.size == 0:
        raise ValueError("No data rows found.")
    xr = np.round(x, decimals=decimals)
    yr = np.round(y, decimals=decimals)
    zr = np.round(z, decimals=decimals)

    x_unique = np.unique(xr)
    y_unique = np.unique(yr)
    z_unique = np.unique(zr)
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

    exp_pts = nx * ny * nz
    if exp_pts != x.size:
        print(f"[warn] expected {exp_pts} points from unique coords, found {x.size}. "
              "Proceeding with map fill (holes -> 0).")

    # maps
    x_idx = {val: i for i, val in enumerate(x_unique)}
    y_idx = {val: i for i, val in enumerate(y_unique)}
    z_idx = {val: i for i, val in enumerate(z_unique)}

    Vx = np.full((nx, ny, nz), np.nan, dtype=float)
    Vy = np.full((nx, ny, nz), np.nan, dtype=float)
    Vz = np.full((nx, ny, nz), np.nan, dtype=float)

    for i in range(x.size):
        Vx[x_idx[xr[i]], y_idx[yr[i]], z_idx[zr[i]]] = vx[i]
        Vy[x_idx[xr[i]], y_idx[yr[i]], z_idx[zr[i]]] = vy[i]
        Vz[x_idx[xr[i]], y_idx[yr[i]], z_idx[zr[i]]] = vz[i]

    Vx = np.nan_to_num(Vx)
    Vy = np.nan_to_num(Vy)
    Vz = np.nan_to_num(Vz)
    return x_unique, y_unique, z_unique, Vx, Vy, Vz


def compute_tke(Vx, Vy, Vz):
    return 0.5 * np.mean(Vx*Vx + Vy*Vy + Vz*Vz)


def scatter_plot_mag(xu, yu, zu, Vmag, title="", subsample=1, show=True, save=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    subsample = max(1, int(subsample))
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
        print(f"[info] Saved figure: {save}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize MFEM-sampled velocity from .txt or .h5")
    ap.add_argument("data_file", type=str, help="Path to SampledDataXXXX.txt or .h5")
    ap.add_argument("--dataset", type=str, default="samples",
                    help="HDF5 dataset name (default: 'samples'; if not found, first 2D dataset is used)")
    ap.add_argument("--layout", type=str, choices=["auto", "row", "col"], default="auto",
                    help="HDF5 on-disk layout: auto/[row=[N,6]/col=[6,N]]")
    ap.add_argument("--round", type=int, default=10, help="Rounding decimals for coords (default: 10)")
    ap.add_argument("--chunksize", type=int, default=1_000_000,
                    help="Pandas chunk size (rows) for text input")
    ap.add_argument("--subsample", type=int, default=1, help="Plot every k-th point per axis (>=1)")
    ap.add_argument("--no-show", action="store_true", help="Do not display plots")
    ap.add_argument("--save", type=str, default=None, help="Path to save PNG(s)")
    args = ap.parse_args()

    path = args.data_file
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    step, time_val = "Unknown", np.nan

    if ext == ".h5" or ext == ".hdf5":
        if not HAS_H5PY:
            print("h5py not installed. Try: pip install h5py")
            sys.exit(2)

        step, time_val, x, y, z, vx, vy, vz, dset_name, dset_shape, layout_eff = load_h5(
            path, dataset=args.dataset, layout=args.layout
        )
        print(f"[info] HDF5 dataset: {dset_name}, shape: {dset_shape}, layout: {layout_eff}")

        # Fallback: parse step/time from filename if attrs missing
        if (isinstance(step, str) and step == "Unknown") or (isinstance(step, float) and np.isnan(step)):
            m = re.search(r"SampledData(\d+)", os.path.basename(path))
            if m: step = m.group(1)
        if isinstance(time_val, float) and np.isnan(time_val):
            time_val = 0.0
    else:
        if not HAS_PANDAS:
            print("pandas not installed. For text input, try: pip install pandas")
            sys.exit(2)
        step, time_val, skiprows = parse_header_text(path)
        print(f"[info] Step: {step}, Time: {time_val if np.isnan(time_val) else f'{time_val:.6e}'}")
        print(f"[info] Reading text (chunk size = {args.chunksize:,}) ...")
        x, y, z, vx, vy, vz = load_text_columns(path, skiprows=skiprows, chunksize=args.chunksize)

    print(f"[info] Loaded {x.size:,} points.")
    xu, yu, zu, Vx, Vy, Vz = reconstruct_grid(x, y, z, vx, vy, vz, decimals=args.round)
    print(f"[info] Grid shape: ({len(xu)}, {len(yu)}, {len(zu)}) = {len(xu)*len(yu)*len(zu):,} pts")

    # Bounds
    print("[bounds] x ∈ [{:.6e}, {:.6e}]".format(x.min(), x.max()))
    print("[bounds] y ∈ [{:.6e}, {:.6e}]".format(y.min(), y.max()))
    print("[bounds] z ∈ [{:.6e}, {:.6e}]".format(z.min(), z.max()))
    print("[bounds] xu ∈ [{:.6e}, {:.6e}]".format(xu.min(), xu.max()))
    print("[bounds] yu ∈ [{:.6e}, {:.6e}]".format(yu.min(), yu.max()))
    print("[bounds] zu ∈ [{:.6e}, {:.6e}]".format(zu.min(), zu.max()))

    # TKE
    tke_grid = compute_tke(Vx, Vy, Vz)
    tke_flat = 0.5 * np.mean(vx*vx + vy*vy + vz*vz)
    print(f"[info] TKE (flat) = {tke_flat:.8f}")
    print(f"[info] TKE (grid) = {tke_grid:.8f}")

    Vmag = np.sqrt(Vx*Vx + Vy*Vy + Vz*Vz)
    title = f"|u| — TKE={tke_grid:.6f}  Step={step}  Time={time_val if np.isnan(time_val) else f'{time_val:.3e}'}"

    # Plots
    plot_face_slices_2d(
        xu, yu, zu, Vmag,
        title=title,
        subsample=args.subsample,
        show=(not args.no_show),
        save=(args.save.replace(".png", "_faces2d.png") if args.save else None)
    )

    scatter_plot_mag(
        xu, yu, zu, Vmag,
        title=title,
        subsample=args.subsample,
        show=(not args.no_show),
        save=args.save
    )


if __name__ == "__main__":
    main()
