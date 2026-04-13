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

How to run:
  1) MFEM sampled text file:
     python3 python_scripts/ComputeFGFromVelocityFFT.py \
       SamplePointsVelocity_Re400NumPtsPerDir8RefLv2P2/cycle_0/SampledData0.txt \
       --header-lines 6

  2) Dedalus stitched HDF5 file (tasks/u):
     python3 python_scripts/ComputeFGFromVelocityFFT.py \
       spectralDNS_tgv_incomp_Re400NumPtsPerDir128/tgv_out_Re400NumPtsPerDir128_fields/tgv_out_Re400NumPtsPerDir128_fields_s1.h5 \
       --snapshot-index 0

  3) Dedalus directory auto-detection:
     python3 python_scripts/ComputeFGFromVelocityFFT.py \
       spectralDNS_tgv_incomp_Re400NumPtsPerDir128 \
       --snapshot-index 0

  Snapshot index examples (Dedalus HDF5 only):
    --snapshot-index 0   = first saved snapshot
    --snapshot-index 5   = 6th saved snapshot
    --snapshot-index -1  = last saved snapshot (default)
    --snapshot-index -2  = second-to-last saved snapshot

  4) Verification (Taylor-Green on [0,1)^3):
     python3 python_scripts/ComputeFGFromVelocityFFT.py --verify --verify-n 32

  5) Batch/headless mode:
     python3 python_scripts/ComputeFGFromVelocityFFT.py <input_path> --no-plot

  6) Spectra mode (diagonal tensor shells + derived E11/E(k) from R11):
     python3 python_scripts/ComputeFGFromVelocityFFT.py <input_path> --plot-ek

  7) Verification + spectra:
     python3 python_scripts/ComputeFGFromVelocityFFT.py --verify --verify-n 32 --plot-ek

  8) Cross-correlation spectra (off-diagonal Phi_ij):
     python3 python_scripts/ComputeFGFromVelocityFFT.py <input_path> --plot-cross-spectrum
     # plots 2D kx-ky fields (default kz~0 slice) + shell-binned summary

  9) Plot S2(r) comparison:
     python3 python_scripts/ComputeFGFromVelocityFFT.py <input_path> --plot-s2
     # overlays:
     #   local S2(r) from shifted velocity differences
     #   S2_from_f(r) = (4/3) * Ek * (1 - f(r))

  10) Example S2(r) comparison on MFEM sampled text:
      python3 python_scripts/ComputeFGFromVelocityFFT.py \
        SamplePointsVelocity_Re400NumPtsPerDir8RefLv2P2/cycle_0/SampledData0.txt \
        --header-lines 6 --plot-s2

Related script:
  For FluidSF-style 3D structure functions (ASF_V, LL, LLL, LTT) on the same
  MFEM/Dedalus inputs, use:
    python3 python_scripts/ComputeStructureFunctions3D.py <input_path>
"""

import argparse
import glob
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


def resolve_dedalus_input_file(input_path):
    """Resolve Dedalus input path (file or directory) to a concrete HDF5 file."""
    if os.path.isfile(input_path):
        return input_path
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    patterns = [
        os.path.join(input_path, "**", "*_fields_s*.h5"),
        os.path.join(input_path, "**", "*.h5"),
    ]
    candidates = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            b = os.path.basename(p)
            if "_p" in b:
                continue
            if "checkpoints" in p or "restart_history" in p:
                continue
            candidates.append(p)
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(f"No Dedalus-compatible .h5 files found in {input_path}")

    def set_num(path):
        m = re.search(r"_s(\d+)\.h5$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=set_num)[-1]


def is_dedalus_velocity_h5(filename):
    if not filename.endswith(".h5"):
        return False
    if not os.path.isfile(filename):
        return False
    try:
        with h5py.File(filename, "r") as f:
            return ("tasks" in f) and ("u" in f["tasks"])
    except Exception:
        return False


def _read_time_cycle_from_dedalus(top_file, idx):
    """Read step (cycle) and time metadata from Dedalus HDF5."""
    with h5py.File(top_file, "r") as f:
        time_val = 0.0
        step_num = "unknown"
        if "scales" in f and "sim_time" in f["scales"]:
            sim_time = np.array(f["scales/sim_time"][:]).reshape(-1)
            if sim_time.size:
                time_val = float(sim_time[idx])
        elif "tasks" in f and "sim_time" in f["tasks"]:
            sim_time = np.array(f["tasks/sim_time"][:]).reshape(-1)
            if sim_time.size:
                time_val = float(sim_time[idx])

        if "scales" in f and "iteration" in f["scales"]:
            it = np.array(f["scales/iteration"][:]).reshape(-1)
            if it.size:
                step_num = str(int(it[idx]))
        elif "tasks" in f and "cycle" in f["tasks"]:
            cyc = np.array(f["tasks/cycle"][:]).reshape(-1)
            if cyc.size:
                step_num = str(int(cyc[idx]))

    return step_num, time_val


def _read_xyz_scales_from_dedalus(top_file, nx, ny, nz):
    with h5py.File(top_file, "r") as f:
        if "scales" in f:
            s = f["scales"]
            x_key = next((k for k in s.keys() if k.startswith("x_")), None)
            y_key = next((k for k in s.keys() if k.startswith("y_")), None)
            z_key = next((k for k in s.keys() if k.startswith("z_")), None)
            if x_key and y_key and z_key:
                x = np.array(s[x_key][:], dtype=np.float64)
                y = np.array(s[y_key][:], dtype=np.float64)
                z = np.array(s[z_key][:], dtype=np.float64)
                if x.size == nx and y.size == ny and z.size == nz:
                    return x, y, z

    # Fallback: assume periodic [0,1)^3 grid.
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, 1.0, ny, endpoint=False, dtype=np.float64)
    z = np.linspace(0.0, 1.0, nz, endpoint=False, dtype=np.float64)
    return x, y, z


def _stitch_dedalus_shards(shard_file, task_name, idx):
    """Stitch distributed Dedalus shard files (*_p*.h5) into a global array."""
    base = os.path.basename(shard_file)
    m = re.match(r"(.+)_p\d+\.h5$", base)
    if not m:
        raise ValueError(f"Not a shard filename: {shard_file}")
    prefix = m.group(1)
    shard_dir = os.path.dirname(shard_file)
    shard_paths = sorted(
        glob.glob(os.path.join(shard_dir, f"{prefix}_p*.h5")),
        key=lambda p: int(re.search(r"_p(\d+)\.h5$", os.path.basename(p)).group(1))
    )
    if not shard_paths:
        raise RuntimeError(f"No shard files found for {shard_file}")

    global_shape = None
    dtype = None
    for p in shard_paths:
        with h5py.File(p, "r") as f:
            dset = f[f"tasks/{task_name}"]
            if global_shape is None:
                global_shape = tuple(int(v) for v in dset.attrs["global_shape"][-3:])
                dtype = dset.dtype

    nx, ny, nz = global_shape
    arr = np.zeros((3, nx, ny, nz), dtype=dtype)

    for p in shard_paths:
        with h5py.File(p, "r") as f:
            dset = f[f"tasks/{task_name}"]
            chunk = np.array(dset[idx], dtype=np.float64)
            start = np.array(dset.attrs["local_start"], dtype=int)
            count = np.array(dset.attrs["local_shape"], dtype=int)
            xs = slice(start[-3], start[-3] + count[-3])
            ys = slice(start[-2], start[-2] + count[-2])
            zs = slice(start[-1], start[-1] + count[-1])
            arr[:, xs, ys, zs] = chunk

    return arr


def read_dedalus_velocity_h5(input_path, snapshot_index=-1, task_name="u"):
    """
    Read Dedalus-style velocity output:
      tasks/u shape = (nt, 3, nx, ny, nz) (stitched) or distributed shards.
    Returns the same tuple shape used by the rest of this script.
    """
    filename = resolve_dedalus_input_file(input_path)
    print(f"Reading Dedalus velocity from: {filename}")

    is_shard = re.search(r"_p\d+\.h5$", os.path.basename(filename)) is not None

    with h5py.File(filename, "r") as f:
        if "tasks" not in f or task_name not in f["tasks"]:
            raise ValueError(f"Dedalus file missing tasks/{task_name}: {filename}")
        dset = f[f"tasks/{task_name}"]
        nt = int(dset.shape[0])

    idx = snapshot_index if snapshot_index >= 0 else (nt + snapshot_index)
    if idx < 0 or idx >= nt:
        raise IndexError(f"snapshot-index {snapshot_index} is out of bounds for nt={nt}")

    # If this is a shard file, stitch from all shard pieces.
    if is_shard:
        u_all = _stitch_dedalus_shards(filename, task_name, idx)
        # Try to find the stitched/top-level file for metadata and scales.
        shard_dir = os.path.dirname(filename)
        set_name = re.sub(r"_p\d+\.h5$", "", os.path.basename(filename))
        top_file = os.path.join(os.path.dirname(shard_dir), f"{set_name}.h5")
        if not os.path.isfile(top_file):
            top_file = filename
    else:
        with h5py.File(filename, "r") as f:
            u_all = np.array(f[f"tasks/{task_name}"][idx], dtype=np.float64)
        top_file = filename

    if u_all.ndim != 4 or u_all.shape[0] != 3:
        raise ValueError(f"Expected tasks/{task_name}[idx] shape (3,nx,ny,nz), got {u_all.shape}")

    nx, ny, nz = u_all.shape[1], u_all.shape[2], u_all.shape[3]
    x_coords, y_coords, z_coords = _read_xyz_scales_from_dedalus(top_file, nx, ny, nz)
    dx = x_coords[1] - x_coords[0] if nx > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if ny > 1 else 1.0
    dz = z_coords[1] - z_coords[0] if nz > 1 else 1.0

    step_number, time_value = _read_time_cycle_from_dedalus(top_file, idx)
    print(f"  Snapshot index: {idx}/{nt-1}")
    print(f"  Step: {step_number}, Time: {time_value:.6e}")
    print(f"  Grid dimensions: {nx} × {ny} × {nz}")
    print(f"  Grid spacing: dx={dx:.8f}, dy={dy:.8f}, dz={dz:.8f}")

    vx = u_all[0]
    vy = u_all[1]
    vz = u_all[2]
    return vx, vy, vz, x_coords, y_coords, z_coords, dx, dy, dz, step_number, time_value


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
    print("Step 1: Forward FFT of velocity")
    n_tot = vx.size

    ux_k = fft.fftn(vx)
    uy_k = fft.fftn(vy)
    uz_k = fft.fftn(vz)

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


def compute_tensor_diagonal_spectrum_binned(ux_k, uy_k, uz_k):
    """
    Compute shell-binned diagonal spectral-tensor contributions using the
    same integer-wavenumber binning style as ComputeSpectraCompressiveVorticalModes.py.

    Returns shell-summed:
      E11 = 0.5 * sum_shell |u_hat_x|^2
      E22 = 0.5 * sum_shell |u_hat_y|^2
      E33 = 0.5 * sum_shell |u_hat_z|^2
      E_total = E11 + E22 + E33
    with Fourier coefficients normalized by N = nx*ny*nz.
    """
    nx, ny, nz = ux_k.shape
    n_tot = nx * ny * nz

    ux_n = ux_k / n_tot
    uy_n = uy_k / n_tot
    uz_n = uz_k / n_tot

    phi11 = np.abs(ux_n) ** 2
    phi22 = np.abs(uy_n) ** 2
    phi33 = np.abs(uz_n) ** 2

    kx_int = np.fft.fftfreq(nx, 1.0 / nx).astype(int)
    ky_int = np.fft.fftfreq(ny, 1.0 / ny).astype(int)
    kz_int = np.fft.fftfreq(nz, 1.0 / nz).astype(int)
    KX_int, KY_int, KZ_int = np.meshgrid(kx_int, ky_int, kz_int, indexing="ij")
    kmag = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)

    from math import ceil
    k_max_int = ceil(nx * 0.5 * np.sqrt(3.0))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * np.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    k_flat = kmag.ravel()
    E11 = np.histogram(k_flat, bins=k_bin_edges, weights=0.5 * phi11.ravel())[0]
    E22 = np.histogram(k_flat, bins=k_bin_edges, weights=0.5 * phi22.ravel())[0]
    E33 = np.histogram(k_flat, bins=k_bin_edges, weights=0.5 * phi33.ravel())[0]
    E_total = E11 + E22 + E33
    return k_centers, E11, E22, E33, E_total


def compute_tensor_cross_spectrum_binned(ux_k, uy_k, uz_k):
    """
    Compute shell-binned off-diagonal spectral-tensor terms on the same
    integer shell bins used for diagonal spectra.

    Returns shell-summed real parts and magnitudes:
      Re(Phi12), Re(Phi13), Re(Phi23), |Phi12|, |Phi13|, |Phi23|.
    """
    nx, ny, nz = ux_k.shape
    n_tot = nx * ny * nz

    ux_n = ux_k / n_tot
    uy_n = uy_k / n_tot
    uz_n = uz_k / n_tot

    phi12 = ux_n * np.conj(uy_n)
    phi13 = ux_n * np.conj(uz_n)
    phi23 = uy_n * np.conj(uz_n)

    kx_int = np.fft.fftfreq(nx, 1.0 / nx).astype(int)
    ky_int = np.fft.fftfreq(ny, 1.0 / ny).astype(int)
    kz_int = np.fft.fftfreq(nz, 1.0 / nz).astype(int)
    KX_int, KY_int, KZ_int = np.meshgrid(kx_int, ky_int, kz_int, indexing="ij")
    kmag = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)

    from math import ceil
    k_max_int = ceil(nx * 0.5 * np.sqrt(3.0))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * np.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    k_flat = kmag.ravel()
    re12 = np.histogram(k_flat, bins=k_bin_edges, weights=np.real(phi12).ravel())[0]
    re13 = np.histogram(k_flat, bins=k_bin_edges, weights=np.real(phi13).ravel())[0]
    re23 = np.histogram(k_flat, bins=k_bin_edges, weights=np.real(phi23).ravel())[0]
    ab12 = np.histogram(k_flat, bins=k_bin_edges, weights=np.abs(phi12).ravel())[0]
    ab13 = np.histogram(k_flat, bins=k_bin_edges, weights=np.abs(phi13).ravel())[0]
    ab23 = np.histogram(k_flat, bins=k_bin_edges, weights=np.abs(phi23).ravel())[0]
    return k_centers, re12, re13, re23, ab12, ab13, ab23


def compute_tensor_cross_spectrum_2d_slice(ux_k, uy_k, uz_k, dx, dy, dz, kz_index=None):
    """
    Compute 2D kx-ky fields of off-diagonal spectral tensor terms on a fixed kz slice.
    Returns fftshifted kx, ky grids and slice fields for Phi12, Phi13, Phi23.
    """
    nx, ny, nz = ux_k.shape
    n_tot = nx * ny * nz

    ux_n = ux_k / n_tot
    uy_n = uy_k / n_tot
    uz_n = uz_k / n_tot

    phi12 = ux_n * np.conj(uy_n)
    phi13 = ux_n * np.conj(uz_n)
    phi23 = uy_n * np.conj(uz_n)

    kx = 2.0 * np.pi * fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * fft.fftfreq(nz, d=dz)

    if kz_index is None:
        kz_index = int(np.argmin(np.abs(kz)))
    else:
        kz_index = int(kz_index) % nz

    p12 = phi12[:, :, kz_index]
    p13 = phi13[:, :, kz_index]
    p23 = phi23[:, :, kz_index]

    # Center the wavenumber origin in the 2D map.
    kx_s = np.fft.fftshift(kx)
    ky_s = np.fft.fftshift(ky)
    p12_s = np.fft.fftshift(p12, axes=(0, 1))
    p13_s = np.fft.fftshift(p13, axes=(0, 1))
    p23_s = np.fft.fftshift(p23, axes=(0, 1))

    return {
        "kx": kx_s,
        "ky": ky_s,
        "kz_value": float(kz[kz_index]),
        "kz_index": kz_index,
        "phi12": p12_s,
        "phi13": p13_s,
        "phi23": p23_s,
    }


def compute_e11_from_r11_longitudinal(fg, k_shell, k0):
    """
    Compute E11(k1) from longitudinal correlation R11(r1):
      E11(k1) = (2/pi) <u1^2> int f(r1) cos(k1 r1) dr1
              = (2/pi) int R11(r1) cos(k1 r1) dr1
    where f(r1) = R11(r1)/R11(0).
    """
    r = np.asarray(fg["r"], dtype=np.float64)
    r11 = np.asarray(fg["f_x"], dtype=np.float64)
    if r.size < 2:
        raise ValueError("Need at least two r points to compute E11 from R11.")

    # Use the formula exactly as provided by the user.
    u1_var = r11[0]
    if np.abs(u1_var) > 0.0:
        f_long = r11 / u1_var
    else:
        f_long = np.zeros_like(r11)

    k_shell = np.asarray(k_shell, dtype=np.float64)
    k_phys = k_shell * float(k0)
    e11 = np.empty_like(k_shell, dtype=np.float64)
    pref = 2.0 / np.pi
    for i, k in enumerate(k_phys):
        integrand = f_long * np.cos(k * r)
        e11[i] = pref * u1_var * np.trapz(integrand, r)

    return e11


def compute_ek_from_e11_derivative(k_shell, e11, k0):
    """
    Compute isotropic E(k) from E11(k):
      E(k) = 1/2 * k^3 * d/dk [ (1/k) * dE11/dk ].
    """
    k_shell = np.asarray(k_shell, dtype=np.float64)
    k = k_shell * float(k0)
    e11 = np.asarray(e11, dtype=np.float64)
    if k.size < 3:
        raise ValueError("Need at least 3 k points to compute derivative-based E(k).")

    dedk = np.gradient(e11, k)

    ek = np.full_like(e11, np.nan, dtype=np.float64)
    mask = k > 0.0
    if np.count_nonzero(mask) < 3:
        return ek

    q = dedk[mask] / k[mask]
    dqdk = np.gradient(q, k[mask])
    ek[mask] = 0.5 * (k[mask] ** 3) * dqdk
    return ek


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


def compute_taylor_microscales_from_curves(r, f_curve, g_curve, prefactor=1.0):
    """
    Compute Taylor microscales from normalized/unnormalized correlation curves:
      lambda_f = sqrt(-prefactor * f(0) / f''(0))
      lambda_g = sqrt(-prefactor * g(0) / g''(0))

    Default uses prefactor=1.0 from the small-r expansion:
      C(r) = C(0) + 0.5 C''(0) r^2 + ...
      lambda^2 = -C(0)/C''(0)
    """
    d2f0 = second_derivative_at_origin(r, f_curve)
    d2g0 = second_derivative_at_origin(r, g_curve)

    f0 = f_curve[0]
    g0 = g_curve[0]

    lam_f = np.sqrt(-prefactor * f0 / d2f0) if d2f0 < 0.0 else np.nan
    lam_g = np.sqrt(-prefactor * g0 / d2g0) if d2g0 < 0.0 else np.nan

    return lam_f, lam_g, d2f0, d2g0


def compute_component_taylor_microscales_from_f(fg, eps=1e-14, prefactor=1.0):
    """
    Compute componentwise longitudinal Taylor microscales from:
      f_x(r)=R11(r e_x), f_y(r)=R22(r e_y), f_z(r)=R33(r e_z)
    using lambda_b = sqrt(-prefactor * f_b(0) / f_b''(0)).

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
        lam = np.sqrt(-prefactor * c0 / d2) if d2 < 0.0 else np.nan
        status = "ok" if np.isfinite(lam) else "invalid_curvature"
        result[tag] = {"lambda": lam, "d2": d2, "c0": c0, "status": status}

    lam_avg = (result["x"]["lambda"] + result["y"]["lambda"] + result["z"]["lambda"]) / 3.0
    return result, lam_avg


def compute_component_taylor_microscales_from_g(fg, eps=1e-14, prefactor=1.0):
    """
    Compute componentwise transverse Taylor microscales from:
      g_x(r)=0.5*(R22(r e_x)+R33(r e_x)),
      g_y(r)=0.5*(R11(r e_y)+R33(r e_y)),
      g_z(r)=0.5*(R11(r e_z)+R22(r e_z))
    using lambda_b = sqrt(-prefactor * g_b(0) / g_b''(0)).
    """
    r = fg["r"]
    result = {}
    for tag in ("x", "y", "z"):
        curve = fg[f"g_{tag}"]
        c0 = curve[0]
        if np.abs(c0) <= eps:
            result[tag] = {"lambda": 0.0, "d2": 0.0, "c0": c0, "status": "zero_component"}
            continue

        d2 = second_derivative_at_origin(r, curve)
        lam = np.sqrt(-prefactor * c0 / d2) if d2 < 0.0 else np.nan
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


def compute_taylor_microscale_gradient_spectral_transverse(vx, vy, vz, dx, dy, dz):
    """
    Componentwise transverse microscales from spectral gradients.

    For each separation direction b, use the two velocity components transverse
    to b and derivatives with respect to b:
      lambda_t,b = sqrt(<u_t^2>_b / <(du_t/dx_b)^2>_b)
    where <u_t^2>_x = 0.5(<u_y^2> + <u_z^2>), etc.
    """
    nx, ny, nz = vx.shape
    n_tot = nx * ny * nz
    KX, KY, KZ = make_kgrids(nx, ny, nz, dx, dy, dz)

    ux_k = fft.fftn(vx)
    uy_k = fft.fftn(vy)
    uz_k = fft.fftn(vz)

    fac = 1.0 / (n_tot * n_tot)
    u2_x = fac * np.sum(np.abs(ux_k)**2)
    u2_y = fac * np.sum(np.abs(uy_k)**2)
    u2_z = fac * np.sum(np.abs(uz_k)**2)

    dux_dx2 = fac * np.sum((KX**2) * (np.abs(ux_k)**2))
    duy_dx2 = fac * np.sum((KX**2) * (np.abs(uy_k)**2))
    duz_dx2 = fac * np.sum((KX**2) * (np.abs(uz_k)**2))

    dux_dy2 = fac * np.sum((KY**2) * (np.abs(ux_k)**2))
    duy_dy2 = fac * np.sum((KY**2) * (np.abs(uy_k)**2))
    duz_dy2 = fac * np.sum((KY**2) * (np.abs(uz_k)**2))

    dux_dz2 = fac * np.sum((KZ**2) * (np.abs(ux_k)**2))
    duy_dz2 = fac * np.sum((KZ**2) * (np.abs(uy_k)**2))
    duz_dz2 = fac * np.sum((KZ**2) * (np.abs(uz_k)**2))

    ut2_x = 0.5 * (u2_y + u2_z)
    ut2_y = 0.5 * (u2_x + u2_z)
    ut2_z = 0.5 * (u2_x + u2_y)

    dut2_x = 0.5 * (duy_dx2 + duz_dx2)
    dut2_y = 0.5 * (dux_dy2 + duz_dy2)
    dut2_z = 0.5 * (dux_dz2 + duy_dz2)

    lam_x = np.sqrt(ut2_x / dut2_x) if dut2_x > 0.0 else 0.0
    lam_y = np.sqrt(ut2_y / dut2_y) if dut2_y > 0.0 else 0.0
    lam_z = np.sqrt(ut2_z / dut2_z) if dut2_z > 0.0 else 0.0
    lam_avg = (lam_x + lam_y + lam_z) / 3.0

    return {
        "x": {"lambda": lam_x, "u2": ut2_x, "du2": dut2_x},
        "y": {"lambda": lam_y, "u2": ut2_y, "du2": dut2_y},
        "z": {"lambda": lam_z, "u2": ut2_z, "du2": dut2_z},
        "avg": lam_avg,
    }


def lambda_sq_ratio(lambda_num, lambda_den):
    """Compute (lambda_num^2 / lambda_den^2) with safe zero handling."""
    if (not np.isfinite(lambda_num)) or (not np.isfinite(lambda_den)) or abs(lambda_den) <= 1e-30:
        return np.nan
    return (lambda_num * lambda_num) / (lambda_den * lambda_den)


def compute_s2_from_f_model(fg, ek):
    """
    Build the averaged second-order model curve:
      S2_from_f(r) = (4/3) * Ek * (1 - f(r))
    where Ek = 0.5 * <u_i u_i>.
    """
    r = np.asarray(fg["r"], dtype=np.float64)
    f = np.asarray(fg["f"], dtype=np.float64)
    s2 = (4.0 / 3.0) * float(ek) * (1.0 - f)
    return {"r": r, "f": f, "s2": s2, "Ek": float(ek)}


def compute_s2_local_longitudinal(vx, vy, vz, dx, dy, dz, max_r_points=None):
    """
    Compute local second-order longitudinal structure functions with the same
    periodic shift approach used in ComputeStructureFunctions3D.py:
      S2_x(r) = <(u(x+r e_x) - u(x))^2>
      S2_y(r) = <(v(x+r e_y) - v(x))^2>
      S2_z(r) = <(w(x+r e_z) - w(x))^2>
    and an averaged curve obtained on the common r grid used for f(r).
    """
    nx, ny, nz = vx.shape
    nmax = min(nx, ny, nz) // 2
    if max_r_points is not None:
        nmax = min(nmax, max_r_points)

    shift_idx = np.arange(nmax + 1, dtype=int)
    sx = np.zeros_like(shift_idx, dtype=np.float64)
    sy = np.zeros_like(shift_idx, dtype=np.float64)
    sz = np.zeros_like(shift_idx, dtype=np.float64)

    for s in range(1, nmax + 1):
        dux = np.roll(vx, -s, axis=0) - vx
        duy = np.roll(vy, -s, axis=1) - vy
        duz = np.roll(vz, -s, axis=2) - vz
        sx[s] = np.mean(dux * dux)
        sy[s] = np.mean(duy * duy)
        sz[s] = np.mean(duz * duz)

    rx = shift_idx * float(dx)
    ry = shift_idx * float(dy)
    rz = shift_idx * float(dz)
    r_common = shift_idx * float((dx + dy + dz) / 3.0)

    def interp_to_common(r_src, s_src):
        if len(r_src) < 2:
            return np.full_like(r_common, np.nan, dtype=np.float64)
        return np.interp(r_common, r_src, s_src, left=np.nan, right=np.nan)

    sx_c = interp_to_common(rx, sx)
    sy_c = interp_to_common(ry, sy)
    sz_c = interp_to_common(rz, sz)
    s_avg = np.nanmean(np.vstack([sx_c, sy_c, sz_c]), axis=0)

    return {
        "r": r_common,
        "s2_x": sx,
        "s2_y": sy,
        "s2_z": sz,
        "s2_x_common": sx_c,
        "s2_y_common": sy_c,
        "s2_z_common": sz_c,
        "s2_avg": s_avg,
    }


def summarize_s2_mismatch(s2_model, s2_local):
    mask = (
        np.isfinite(s2_model["r"]) & np.isfinite(s2_model["s2"])
        & np.isfinite(s2_local["r"]) & np.isfinite(s2_local["s2_avg"])
    )
    if not np.any(mask):
        return {
            "n_overlap": 0,
            "max_abs": np.nan,
            "rms": np.nan,
            "mean_abs": np.nan,
            "l2_abs": np.nan,
            "l2_rel": np.nan,
        }
    diff = s2_local["s2_avg"][mask] - s2_model["s2"][mask]
    r = s2_model["r"][mask]
    l2_abs = float(np.sqrt(np.trapz(diff * diff, r))) if diff.size > 1 else float(np.abs(diff[0]))
    ref = s2_local["s2_avg"][mask]
    ref_l2 = float(np.sqrt(np.trapz(ref * ref, r))) if ref.size > 1 else float(np.abs(ref[0]))
    return {
        "n_overlap": int(np.count_nonzero(mask)),
        "max_abs": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "l2_abs": l2_abs,
        "l2_rel": (l2_abs / ref_l2) if ref_l2 > 0.0 else np.nan,
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
    s2_model = compute_s2_from_f_model(fg, 0.5 * fg["f_raw_avg"][0] * 3.0)
    columns = [
        ("r", fg["r"]),
        ("f_norm", fg["f"]),
        ("g_norm", fg["g"]),
        ("s2_from_f_avg", s2_model["s2"]),
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


def plot_s2_comparison(s2_model, s2_local, step_number, time_value, show=True):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(s2_model["r"], s2_model["s2"], color="black", lw=2.2, label=r"$\frac{4}{3}E_k(1-f(r))$")
    ax.plot(s2_local["r"], s2_local["s2_avg"], color="tab:blue", lw=2.0, ls="--", label=r"Local $S_2(r)$")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$S_2(r)$")
    ax.set_title(
        rf"$S_2(r)$ comparison, step={step_number}, t={time_value:.4e}"
    )
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="best")
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


def plot_tensor_diagonal_spectrum(k, E11, E22, E33, Etotal, step_number, time_value, show=True):
    """Separate log-log plot of binned diagonal spectral-tensor contributions."""
    mask_t = (k > 0.0) & (Etotal > 0.0)
    mask11 = (k > 0.0) & (E11 > 0.0)
    mask22 = (k > 0.0) & (E22 > 0.0)
    mask33 = (k > 0.0) & (E33 > 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(k[mask_t], Etotal[mask_t], color="black", lw=2.2, label=r"$\frac{1}{2}\mathrm{Tr}(\Phi)$ shell sum")
    ax.loglog(k[mask11], E11[mask11], color="tab:blue", lw=1.8, ls="--", label=r"$\frac{1}{2}\Phi_{11}$ shell sum")
    ax.loglog(k[mask22], E22[mask22], color="tab:orange", lw=1.8, ls="-.", label=r"$\frac{1}{2}\Phi_{22}$ shell sum")
    ax.loglog(k[mask33], E33[mask33], color="tab:green", lw=1.8, ls=":", label=r"$\frac{1}{2}\Phi_{33}$ shell sum")
    ax.set_xlabel("Integer shell wavenumber k")
    ax.set_ylabel("Shell-summed spectral energy")
    ax.set_title(f"Diagonal Spectral Tensor Spectrum, step={step_number}, t={time_value:.4e}")
    ax.grid(True, which="both", alpha=0.3, ls="--")
    ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_tensor_cross_spectrum(k, re12, re13, re23, ab12, ab13, ab23, step_number, time_value, show=True):
    """Plot shell-binned off-diagonal spectral tensor terms."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    ax = axes[0]
    ax.plot(k, re12, lw=1.8, color="tab:blue", label=r"$\mathrm{Re}\,\Phi_{12}$ shell sum")
    ax.plot(k, re13, lw=1.8, color="tab:orange", label=r"$\mathrm{Re}\,\Phi_{13}$ shell sum")
    ax.plot(k, re23, lw=1.8, color="tab:green", label=r"$\mathrm{Re}\,\Phi_{23}$ shell sum")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Integer shell wavenumber k")
    ax.set_ylabel("Signed shell sum")
    ax.set_title("Cross-Spectrum (Real Parts)")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="best")

    ax2 = axes[1]
    m12 = (k > 0.0) & (ab12 > 0.0)
    m13 = (k > 0.0) & (ab13 > 0.0)
    m23 = (k > 0.0) & (ab23 > 0.0)
    ax2.loglog(k[m12], ab12[m12], lw=1.8, color="tab:blue", label=r"$|\Phi_{12}|$ shell sum")
    ax2.loglog(k[m13], ab13[m13], lw=1.8, color="tab:orange", label=r"$|\Phi_{13}|$ shell sum")
    ax2.loglog(k[m23], ab23[m23], lw=1.8, color="tab:green", label=r"$|\Phi_{23}|$ shell sum")
    ax2.set_xlabel("Integer shell wavenumber k")
    ax2.set_ylabel("Positive shell sum")
    ax2.set_title("Cross-Spectrum (Magnitudes)")
    ax2.grid(True, which="both", alpha=0.3, ls="--")
    ax2.legend(loc="best")

    fig.suptitle(f"Off-Diagonal Spectral Tensor, step={step_number}, t={time_value:.4e}")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_tensor_cross_spectrum_2d(cross2d, step_number, time_value, show=True):
    """Plot 2D kx-ky fields of cross-spectrum on a fixed kz slice."""
    kx = cross2d["kx"]
    ky = cross2d["ky"]
    kz_val = cross2d["kz_value"]
    phi12 = cross2d["phi12"]
    phi13 = cross2d["phi13"]
    phi23 = cross2d["phi23"]

    re12 = np.real(phi12)
    re13 = np.real(phi13)
    re23 = np.real(phi23)
    ab12 = np.abs(phi12)
    ab13 = np.abs(phi13)
    ab23 = np.abs(phi23)

    vmax_re = max(
        float(np.nanmax(np.abs(re12))),
        float(np.nanmax(np.abs(re13))),
        float(np.nanmax(np.abs(re23))),
    )
    if vmax_re <= 0.0:
        vmax_re = 1.0
    eps = 1e-300

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)

    fields_re = [(re12, r"$\mathrm{Re}\,\Phi_{12}$"), (re13, r"$\mathrm{Re}\,\Phi_{13}$"), (re23, r"$\mathrm{Re}\,\Phi_{23}$")]
    fields_ab = [(ab12, r"$\log_{10}|\Phi_{12}|$"), (ab13, r"$\log_{10}|\Phi_{13}|$"), (ab23, r"$\log_{10}|\Phi_{23}|$")]

    for j, (fld, ttl) in enumerate(fields_re):
        im = axes[0, j].pcolormesh(kx, ky, fld.T, shading="auto", cmap="RdBu_r",
                                   vmin=-vmax_re, vmax=vmax_re)
        axes[0, j].set_title(ttl)
        axes[0, j].set_xlabel(r"$k_x$")
        if j == 0:
            axes[0, j].set_ylabel(r"$k_y$")
        fig.colorbar(im, ax=axes[0, j], shrink=0.9)

    for j, (fld, ttl) in enumerate(fields_ab):
        im = axes[1, j].pcolormesh(kx, ky, np.log10(fld.T + eps), shading="auto", cmap="viridis")
        axes[1, j].set_title(ttl)
        axes[1, j].set_xlabel(r"$k_x$")
        if j == 0:
            axes[1, j].set_ylabel(r"$k_y$")
        fig.colorbar(im, ax=axes[1, j], shrink=0.9)

    fig.suptitle(
        f"2D Cross-Spectral Tensor Slice (kz={kz_val:.4e}), step={step_number}, t={time_value:.4e}"
    )
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_e11_and_ek_from_r11(
        k_shell, e11, ek, step_number, time_value,
        k_regular=None, e_regular=None, show=True):
    """Separate plot for E11(k1) from R11 and derived E(k), with optional regular-spectrum overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax1 = axes[0]
    ax1.plot(k_shell, e11, color="tab:blue", lw=2.0)
    ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax1.set_xlabel("Shell wavenumber k")
    ax1.set_ylabel("E11(k1)")
    ax1.set_title(r"$E_{11}(k_1)$ from $R_{11}(r_1)$")
    ax1.grid(True, alpha=0.3, ls="--")

    ax2 = axes[1]
    knd = np.asarray(k_shell, dtype=np.float64)
    pos = (knd > 0.0) & np.isfinite(ek) & (ek > 0.0)
    neg = (knd > 0.0) & np.isfinite(ek) & (ek < 0.0)
    if np.any(pos):
        ax2.loglog(knd[pos], ek[pos], color="black", lw=2.0, label="Derived E(k) > 0")
    if np.any(neg):
        ax2.loglog(knd[neg], -ek[neg], color="red", marker="x", ls="None", label="-Derived E(k), E(k)<0")

    if k_regular is not None and e_regular is not None:
        kr = np.asarray(k_regular, dtype=np.float64)
        er = np.asarray(e_regular, dtype=np.float64)
        reg_mask = (kr > 0.0) & (er > 0.0) & np.isfinite(kr) & np.isfinite(er)
        if np.any(reg_mask):
            # k_regular from shell binning is already dimensionless shell index.
            ax2.loglog(kr[reg_mask], er[reg_mask], color="tab:blue", lw=1.8, ls="--",
                       label="Regular shell spectrum")

    ax2.set_xlabel("Shell wavenumber k")
    ax2.set_ylabel("E(k)")
    ax2.set_title(r"$E(k)=\frac{1}{2}k^3\frac{d}{dk}\left[\frac{1}{k}\frac{dE_{11}}{dk}\right]$")
    ax2.grid(True, which="both", alpha=0.3, ls="--")
    if np.any(pos) or np.any(neg):
        ax2.legend(loc="best")

    fig.suptitle(f"Derived Spectrum from R11, step={step_number}, t={time_value:.4e}")
    fig.tight_layout()
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


def run_tgv_verification(
        verify_n, max_r_points, out_prefix, no_plot, plot_ek,
        plot_cross_spectrum, cross_kz_index, plot_s2):
    print(f"\n{'='*60}")
    print(f"VERIFY MODE: Taylor-Green vortex (N={verify_n})")
    print(f"{'='*60}")

    vx, vy, vz, x_coords, y_coords, z_coords, dx, dy, dz = build_taylor_green_vortex(verify_n)
    print(f"  Generated grid: {vx.shape}, dx={dx:.6e}, dy={dy:.6e}, dz={dz:.6e}")

    R, hats = compute_tensor_correlations(vx, vy, vz)
    fg = extract_f_g(R, dx, dy, dz, max_r_points)
    ke_check = compute_ke_consistency_from_tensor(vx, vy, vz)
    s2_model = compute_s2_from_f_model(fg, ke_check["ke_phys"])
    s2_local = compute_s2_local_longitudinal(vx, vy, vz, dx, dy, dz, max_r_points=max_r_points)
    s2_cmp = summarize_s2_mismatch(s2_model, s2_local)

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

    lam_f_axis, lam_g_axis, d2f_axis, d2g_axis = compute_taylor_microscales_from_curves(
        r, f_num, g_num, prefactor=1.0
    )
    ratio_axis = lambda_sq_ratio(lam_g_axis, lam_f_axis)
    ratio_axis_err = abs(ratio_axis - 0.5) if np.isfinite(ratio_axis) else np.nan
    lam_exact = 1.0 / (2.0 * np.pi)

    lam_f_avg, lam_g_avg, d2f_avg, d2g_avg = compute_taylor_microscales_from_curves(
        fg["r"], fg["f"], fg["g"], prefactor=1.0
    )
    ratio_avg = lambda_sq_ratio(lam_g_avg, lam_f_avg)
    ratio_avg_err = abs(ratio_avg - 0.5) if np.isfinite(ratio_avg) else np.nan

    f_comp, f_comp_avg = compute_component_taylor_microscales_from_f(fg, prefactor=1.0)
    g_comp, g_comp_avg = compute_component_taylor_microscales_from_g(fg, prefactor=1.0)
    L_comp, L_comp_avg = compute_component_integral_length_scales_from_f(fg)
    L_avg_curve = compute_integral_length_scale_from_curve(fg["r"], fg["f"])
    grad_comp = compute_taylor_microscale_gradient_spectral(vx, vy, vz, dx, dy, dz)
    grad_comp_trans = compute_taylor_microscale_gradient_spectral_transverse(
        vx, vy, vz, dx, dy, dz
    )

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
            ["lambda_longitudinal axis (from f)", f"{lam_f_axis:.8e}"],
            ["lambda_transverse axis (from g)", f"{lam_g_axis:.8e}"],
            ["(lambda_transverse^2/lambda_longitudinal^2) axis", f"{ratio_axis:.8e}"],
            ["|ratio_axis-0.5|", f"{ratio_axis_err:.8e}"],
            ["lambda exact", f"{lam_exact:.8e}"],
            ["f_avg''(0)", f"{d2f_avg:.8e}"],
            ["g_avg''(0)", f"{d2g_avg:.8e}"],
            ["lambda_longitudinal_avg (from f)", f"{lam_f_avg:.8e}"],
            ["lambda_transverse_avg (from g)", f"{lam_g_avg:.8e}"],
            ["(lambda_transverse^2/lambda_longitudinal^2)_avg", f"{ratio_avg:.8e}"],
            ["|ratio_avg-0.5|", f"{ratio_avg_err:.8e}"],
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
    avg_rel = (
        (avg_diff / abs(grad_comp["avg"]) * 100.0) if abs(grad_comp["avg"]) > 0.0 else 0.0
    )
    comp_rows.append(
        [
            "avg/3",
            f"{f_comp_avg:.8e}",
            f"{grad_comp['avg']:.8e}",
            f"{avg_diff:.3e}",
            f"{avg_rel:.3e}%",
        ]
    )
    print("Componentwise longitudinal Taylor microscale comparison:")
    print_table(["Comp", "lambda_longitudinal_from_f", "lambda_longitudinal_from_grad", "abs_diff", "rel_diff"], comp_rows)

    comp_rows_t = []
    for tag in ("x", "y", "z"):
        lt = g_comp[tag]["lambda"]
        lg = grad_comp_trans[tag]["lambda"]
        diff = abs(lt - lg)
        rel = (diff / abs(lg) * 100.0) if abs(lg) > 0.0 else 0.0
        comp_rows_t.append([tag, f"{lt:.8e}", f"{lg:.8e}", f"{diff:.3e}", f"{rel:.3e}%"])
    avg_diff_t = abs(g_comp_avg - grad_comp_trans["avg"])
    avg_rel_t = (
        (avg_diff_t / abs(grad_comp_trans["avg"]) * 100.0) if abs(grad_comp_trans["avg"]) > 0.0 else 0.0
    )
    comp_rows_t.append(
        ["avg/3", f"{g_comp_avg:.8e}", f"{grad_comp_trans['avg']:.8e}", f"{avg_diff_t:.3e}", f"{avg_rel_t:.3e}%"]
    )
    print("Componentwise transverse Taylor microscale comparison:")
    print_table(["Comp", "lambda_transverse_from_g", "lambda_transverse_from_grad", "abs_diff", "rel_diff"], comp_rows_t)

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
    print("S2 comparison:")
    print_table(
        ["Metric", "Value"],
        [
            ["model", "S2_from_f(r) = (4/3) * Ek * (1 - f(r))"],
            ["Ek", f"{s2_model['Ek']:.12e}"],
            ["N_overlap", str(s2_cmp["n_overlap"])],
            ["max_abs", f"{s2_cmp['max_abs']:.3e}"],
            ["rms", f"{s2_cmp['rms']:.3e}"],
            ["mean_abs", f"{s2_cmp['mean_abs']:.3e}"],
            ["L2_abs", f"{s2_cmp['l2_abs']:.3e}"],
            ["L2_rel", f"{100.0*s2_cmp['l2_rel']:.3e}%"],
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
        if plot_s2:
            plot_s2_comparison(s2_model, s2_local, f"tgv_verify_N{verify_n}", 0.0, show=True)
            return passed

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
        print("Step 3 (optional): Compute diagonal spectral-tensor spectrum")
        kdiag, E11, E22, E33, Etot = compute_tensor_diagonal_spectrum_binned(
            hats[0], hats[1], hats[2]
        )
        print("Step 3b (optional): Compute E11(k1) from R11 and derivative E(k)")
        k0_x = 2.0 * np.pi / (vx.shape[0] * dx)
        e11_r11 = compute_e11_from_r11_longitudinal(fg, kdiag, k0_x)
        ek_r11 = compute_ek_from_e11_derivative(kdiag, e11_r11, k0_x)
        if not no_plot:
            plot_tensor_diagonal_spectrum(kdiag, E11, E22, E33, Etot, "tgv_verify", 0.0, show=True)
            plot_e11_and_ek_from_r11(
                kdiag, e11_r11, ek_r11, "tgv_verify", 0.0,
                k_regular=kdiag, e_regular=Etot, show=True
            )

    if plot_cross_spectrum:
        print("Step 3c (optional): Compute cross-correlation spectrum (off-diagonal Phi_ij)")
        kx, re12, re13, re23, ab12, ab13, ab23 = compute_tensor_cross_spectrum_binned(
            hats[0], hats[1], hats[2]
        )
        cross2d = compute_tensor_cross_spectrum_2d_slice(
            hats[0], hats[1], hats[2], dx, dy, dz, kz_index=cross_kz_index
        )
        if not no_plot:
            plot_tensor_cross_spectrum_2d(cross2d, "tgv_verify", 0.0, show=True)
            plot_tensor_cross_spectrum(kx, re12, re13, re23, ab12, ab13, ab23, "tgv_verify", 0.0, show=True)

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
    parser.add_argument("--snapshot-index", type=int, default=-1,
                        help="For Dedalus HDF5 tasks/u: snapshot index in time dimension (default: -1 last)")
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
    parser.add_argument("--plot-cross-spectrum", action="store_true",
                        help="Also compute and plot off-diagonal cross-correlation spectra")
    parser.add_argument("--plot-s2", action="store_true",
                        help="Plot S2 comparison: local shift-based S2(r) and (2/3) * Ek * (1 - f(r))")
    parser.add_argument("--cross-kz-index", type=int, default=None,
                        help="kz-slice index for 2D cross-spectrum field (default: index nearest kz=0)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not open interactive plots (files are still saved)")
    args = parser.parse_args()

    if args.verify:
        run_tgv_verification(
            verify_n=args.verify_n,
            max_r_points=args.max_r_points,
            out_prefix=args.out_prefix,
            no_plot=args.no_plot,
            plot_ek=args.plot_ek,
            plot_cross_spectrum=args.plot_cross_spectrum,
            cross_kz_index=args.cross_kz_index,
            plot_s2=args.plot_s2,
        )
        return

    if args.data_file is None:
        parser.error("data_file is required unless --verify is used")

    print(f"\n{'='*60}")
    print(f"ANALYZING: {args.data_file}")
    print(f"{'='*60}")

    dedalus_candidate = None
    try:
        dedalus_candidate = resolve_dedalus_input_file(args.data_file)
    except Exception:
        dedalus_candidate = None

    if dedalus_candidate and is_dedalus_velocity_h5(dedalus_candidate):
        vx, vy, vz, x_coords, y_coords, z_coords, dx, dy, dz, step_number, time_value = (
            read_dedalus_velocity_h5(
                args.data_file,
                snapshot_index=args.snapshot_index,
                task_name="u"
            )
        )
    else:
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
    s2_model = compute_s2_from_f_model(fg, ke_check["ke_phys"])
    s2_local = compute_s2_local_longitudinal(vx, vy, vz, dx, dy, dz, max_r_points=args.max_r_points)
    s2_cmp = summarize_s2_mismatch(s2_model, s2_local)
    lam_f, lam_g, d2f0, d2g0 = compute_taylor_microscales_from_curves(
        fg["r"], fg["f"], fg["g"], prefactor=1.0
    )
    ratio_avg = lambda_sq_ratio(lam_g, lam_f)
    ratio_avg_err = abs(ratio_avg - 0.5) if np.isfinite(ratio_avg) else np.nan

    f_comp, f_comp_avg = compute_component_taylor_microscales_from_f(fg, prefactor=1.0)
    g_comp, g_comp_avg = compute_component_taylor_microscales_from_g(fg, prefactor=1.0)
    L_comp, L_comp_avg = compute_component_integral_length_scales_from_f(fg)
    L_avg_curve = compute_integral_length_scale_from_curve(fg["r"], fg["f"])
    grad_comp = compute_taylor_microscale_gradient_spectral(vx, vy, vz, dx, dy, dz)
    grad_comp_trans = compute_taylor_microscale_gradient_spectral_transverse(
        vx, vy, vz, dx, dy, dz
    )
    print("Taylor microscale summary:")
    print_table(
        ["Metric", "Value"],
        [
            ["f''(0) avg", f"{d2f0:.8e}"],
            ["g''(0) avg", f"{d2g0:.8e}"],
            ["lambda_longitudinal_avg (from f)", f"{lam_f:.8e}"],
            ["lambda_transverse_avg (from g)", f"{lam_g:.8e}"],
            ["(lambda_transverse^2/lambda_longitudinal^2)_avg", f"{ratio_avg:.8e}"],
            ["|ratio_avg-0.5|", f"{ratio_avg_err:.8e}"],
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
    avg_rel = (
        (avg_diff / abs(grad_comp["avg"]) * 100.0) if abs(grad_comp["avg"]) > 0.0 else 0.0
    )
    comp_rows.append(
        ["avg/3", f"{f_comp_avg:.8e}", f"{grad_comp['avg']:.8e}", f"{avg_diff:.3e}", f"{avg_rel:.3e}%"]
    )
    print("Componentwise longitudinal Taylor microscale comparison:")
    print_table(["Comp", "lambda_longitudinal_from_f", "lambda_longitudinal_from_grad", "abs_diff", "rel_diff"], comp_rows)

    comp_rows_t = []
    for tag in ("x", "y", "z"):
        lt = g_comp[tag]["lambda"]
        lg = grad_comp_trans[tag]["lambda"]
        diff = abs(lt - lg)
        rel = (diff / abs(lg) * 100.0) if abs(lg) > 0.0 else 0.0
        comp_rows_t.append([tag, f"{lt:.8e}", f"{lg:.8e}", f"{diff:.3e}", f"{rel:.3e}%"])
    avg_diff_t = abs(g_comp_avg - grad_comp_trans["avg"])
    avg_rel_t = (
        (avg_diff_t / abs(grad_comp_trans["avg"]) * 100.0) if abs(grad_comp_trans["avg"]) > 0.0 else 0.0
    )
    comp_rows_t.append(
        ["avg/3", f"{g_comp_avg:.8e}", f"{grad_comp_trans['avg']:.8e}", f"{avg_diff_t:.3e}", f"{avg_rel_t:.3e}%"]
    )
    print("Componentwise transverse Taylor microscale comparison:")
    print_table(["Comp", "lambda_transverse_from_g", "lambda_transverse_from_grad", "abs_diff", "rel_diff"], comp_rows_t)

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
    print("S2 comparison:")
    print_table(
        ["Metric", "Value"],
        [
            ["model", "S2_from_f(r) = (4/3) * Ek * (1 - f(r))"],
            ["Ek", f"{s2_model['Ek']:.12e}"],
            ["N_overlap", str(s2_cmp["n_overlap"])],
            ["max_abs", f"{s2_cmp['max_abs']:.3e}"],
            ["rms", f"{s2_cmp['rms']:.3e}"],
            ["mean_abs", f"{s2_cmp['mean_abs']:.3e}"],
            ["L2_abs", f"{s2_cmp['l2_abs']:.3e}"],
            ["L2_rel", f"{100.0*s2_cmp['l2_rel']:.3e}%"],
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

    if not args.no_plot:
        if args.plot_s2:
            plot_s2_comparison(s2_model, s2_local, step_number, time_value, show=True)
        else:
            plot_fg(fg, step_number, time_value, show=True)
    else:
        print("Plotting disabled (--no-plot).")

    if args.plot_ek:
        print("Step 3 (optional): Compute diagonal spectral-tensor spectrum")
        kdiag, E11, E22, E33, Etot = compute_tensor_diagonal_spectrum_binned(
            hats[0], hats[1], hats[2]
        )
        print("Step 3b (optional): Compute E11(k1) from R11 and derivative E(k)")
        k0_x = 2.0 * np.pi / (vx.shape[0] * dx)
        e11_r11 = compute_e11_from_r11_longitudinal(fg, kdiag, k0_x)
        ek_r11 = compute_ek_from_e11_derivative(kdiag, e11_r11, k0_x)
        if not args.no_plot:
            plot_tensor_diagonal_spectrum(kdiag, E11, E22, E33, Etot, step_number, time_value, show=True)
            plot_e11_and_ek_from_r11(
                kdiag, e11_r11, ek_r11, step_number, time_value,
                k_regular=kdiag, e_regular=Etot, show=True
            )

    if args.plot_cross_spectrum:
        print("Step 3c (optional): Compute cross-correlation spectrum (off-diagonal Phi_ij)")
        kx, re12, re13, re23, ab12, ab13, ab23 = compute_tensor_cross_spectrum_binned(
            hats[0], hats[1], hats[2]
        )
        cross2d = compute_tensor_cross_spectrum_2d_slice(
            hats[0], hats[1], hats[2], dx, dy, dz, kz_index=args.cross_kz_index
        )
        if not args.no_plot:
            plot_tensor_cross_spectrum_2d(cross2d, step_number, time_value, show=True)
            plot_tensor_cross_spectrum(kx, re12, re13, re23, ab12, ab13, ab23, step_number, time_value, show=True)


if __name__ == "__main__":
    main()
