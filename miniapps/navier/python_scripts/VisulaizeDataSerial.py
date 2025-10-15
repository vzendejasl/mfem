#!/usr/bin/env python3
"""
Scatter-only visualization of MFEM-sampled velocity data from text (.txt) or HDF5 (.h5).

- For .txt: reads first 6 header lines to parse Step/Time, then six numeric columns: x y z ux uy uz
- For .h5: reads a 2D dataset (default 'samples') shaped [N,6] or [6,N]; auto-detects layout unless --layout is given

Outputs:
- Optional TKE (computed on flat data)
- 3D scatter plot of |u| over points (x,y,z)

Usage:
    python plot_mfem_scatter.py /path/to/SampledDataXXXX.txt
    python plot_mfem_scatter.py /path/to/SampledDataXXXX.h5
    python plot_mfem_scatter.py data.h5 --dataset samples --layout auto --subsample 5 --no-show --save scatter.png
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

# Optional text reader dependency
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# Optional HDF5 dependency
try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False


def parse_header_text(path, header_lines=6):
    """Parse Step and Time from the first few header lines."""
    step = "Unknown"
    time = np.nan
    try:
        with open(path, "r") as f:
            lines = [next(f) for _ in range(header_lines)]
    except StopIteration:
        lines = []
    for line in lines:
        m = re.search(r"Step\s*=\s*(\d+)", line)
        if m:
            step = m.group(1)
        m = re.search(r"Time\s*=\s*([0-9.eE+\-]+)", line)
        if m:
            try:
                time = float(m.group(1))
            except Exception:
                pass
    return step, time, len(lines)


def load_text_columns(path, skiprows, chunksize=1_000_000):
    """Load x,y,z,ux,uy,uz from a whitespace-delimited text file using pandas in chunks."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for reading text; pip install pandas")
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
        dset = None
        if dataset in f and isinstance(f[dataset], h5py.Dataset) and f[dataset].ndim == 2:
            dset = f[dataset]
        else:
            def pick_first_2d(name, obj):
                nonlocal dset
                if dset is None and isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                    dset = obj
            f.visititems(pick_first_2d)
            if dset is None:
                raise ValueError("No 2D dataset found in file; specify --dataset")

        sh = dset.shape
        if layout == "auto":
            if sh[1] == 6:
                layout_eff = "row"
            elif sh[0] == 6:
                layout_eff = "col"
            else:
                raise ValueError(f"Cannot infer layout from dataset shape {sh}. Use --layout row or --layout col.")
        else:
            layout_eff = layout

        if layout_eff == "row":
            N = sh[0]
            xs, ys, zs, ux, uy, uz = [], [], [], [], [], []
            for start in range(0, N, block_rows):
                end = min(N, start + block_rows)
                block = dset[start:end, :]  # (B,6)
                xs.append(block[:, 0].astype(float))
                ys.append(block[:, 1].astype(float))
                zs.append(block[:, 2].astype(float))
                ux.append(block[:, 3].astype(float))
                uy.append(block[:, 4].astype(float))
                uz.append(block[:, 5].astype(float))
            x = np.concatenate(xs) if xs else np.empty(0, float)
            y = np.concatenate(ys) if ys else np.empty(0, float)
            z = np.concatenate(zs) if zs else np.empty(0, float)
            vx = np.concatenate(ux) if ux else np.empty(0, float)
            vy = np.concatenate(uy) if uy else np.empty(0, float)
            vz = np.concatenate(uz) if uz else np.empty(0, float)
        else:
            # 'col': [6, N]
            x = dset[0, :].astype(float)
            y = dset[1, :].astype(float)
            z = dset[2, :].astype(float)
            vx = dset[3, :].astype(float)
            vy = dset[4, :].astype(float)
            vz = dset[5, :].astype(float)

        # Step/time from attrs if present
        step = f.attrs.get("step", "Unknown")
        time_val = f.attrs.get("time", np.nan)
        if isinstance(step, np.ndarray) and step.shape == ():
            step = int(step)
        if isinstance(time_val, np.ndarray) and time_val.shape == ():
            time_val = float(time_val)

        return step, time_val, x, y, z, vx, vy, vz, dset.name, tuple(sh), layout_eff


def compute_tke_flat(vx, vy, vz):
    if vx.size == 0:
        return np.nan
    return 0.5 * np.mean(vx*vx + vy*vy + vz*vz)


def scatter_plot_flat(x, y, z, mag, title="", subsample=1, alpha=1.0, s=6, cmap="viridis",
                      show=True, save=None):
    """3D scatter of points colored by magnitude, with optional subsampling."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if subsample is None or subsample < 1:
        subsample = 1
    idx = slice(None, None, subsample)
    xs, ys, zs, ms = x[idx], y[idx], z[idx], mag[idx]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=ms, s=s, marker=".", cmap=cmap, alpha=alpha)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.06)
    cb.set_label("|u|")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
        print(f"[info] Saved scatter figure: {save}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Scatter-only visualization of MFEM samples from .txt or .h5")
    ap.add_argument("data_file", type=str, help="Path to SampledDataXXXX.txt or .h5")
    ap.add_argument("--dataset", type=str, default="samples",
                    help="HDF5 dataset name (default: 'samples'; if not found, first 2D dataset is used)")
    ap.add_argument("--layout", type=str, choices=["auto", "row", "col"], default="auto",
                    help="HDF5 on-disk layout: auto/[row=[N,6]/col=[6,N]]")
    ap.add_argument("--chunksize", type=int, default=1_000_000,
                    help="Pandas chunk size (rows) for text input")
    ap.add_argument("--subsample", type=int, default=1, help="Plot every k-th point (>=1)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Point alpha for scatter")
    ap.add_argument("--size", type=float, default=6.0, help="Marker size for scatter")
    ap.add_argument("--no-show", action="store_true", help="Do not display plot")
    ap.add_argument("--save", type=str, default=None, help="Path to save PNG")
    ap.add_argument("--no-tke", action="store_true", help="Do not compute or show TKE")
    args = ap.parse_args()

    path = args.data_file
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    step, time_val = "Unknown", np.nan

    if ext in [".h5", ".hdf5"]:
        if not HAS_H5PY:
            print("h5py not installed. Try: pip install h5py")
            sys.exit(2)
        step, time_val, x, y, z, vx, vy, vz, dset_name, dset_shape, layout_eff = load_h5(
            path, dataset=args.dataset, layout=args.layout
        )
        print(f"[info] HDF5 dataset: {dset_name}, shape: {dset_shape}, layout: {layout_eff}")

        # Fallbacks from filename
        if (isinstance(step, str) and step == "Unknown") or (isinstance(step, float) and np.isnan(step)):
            m = re.search(r"SampledData(\d+)", os.path.basename(path))
            if m:
                step = m.group(1)
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

    npts = x.size
    print(f"[info] Loaded {npts:,} points.")

    # Bounds
    if npts > 0:
        print("[bounds] x ∈ [{:.6e}, {:.6e}]".format(x.min(), x.max()))
        print("[bounds] y ∈ [{:.6e}, {:.6e}]".format(y.min(), y.max()))
        print("[bounds] z ∈ [{:.6e}, {:.6e}]".format(z.min(), z.max()))

    # Magnitude and optional TKE
    mag = np.sqrt(vx*vx + vy*vy + vz*vz)
    tke_flat = compute_tke_flat(vx, vy, vz) if not args.no_tke else np.nan
    tke_str = "" if args.no_tke else f"  TKE={tke_flat:.6f}"

    # Title
    time_str = time_val if np.isnan(time_val) else f"{time_val:.3e}"
    title = f"|u| scatter{tke_str}  Step={step}  Time={time_str}"

    # Scatter
    scatter_plot_flat(
        x, y, z, mag,
        title=title,
        subsample=max(1, int(args.subsample)),
        alpha=float(args.alpha),
        s=float(args.size),
        cmap="viridis",
        show=(not args.no_show),
        save=args.save
    )


if __name__ == "__main__":
    main()

