#!/usr/bin/env python3
"""
Compute 3D structure functions for MFEM sampled files and Dedalus tasks/u data.
By default this script calls FluidSF directly (same API used in ex_3d):
  fluidsf.generate_structure_functions_3d(...)

This script reuses the data-loading flow from ComputeFGFromVelocityFFT.py:
1) header detection / parsing (for text)
2) chunked read + grid reconstruction
3) auto periodic endpoint drop for sampled text data
4) optional Dedalus snapshot selection via --snapshot-index

Install dependency:
  python3 -m pip install fluidsf
  # Optional fallback without fluidsf:
  #   use --backend local

How to run:
  1) MFEM sampled text:
     python3 python_scripts/ComputeStructureFunctions3D.py \
       SamplePointsVelocity_Re400NumPtsPerDir8RefLv2P2/cycle_0/SampledData0.txt \
       --header-lines 6

  2) Dedalus stitched HDF5 (tasks/u):
     python3 python_scripts/ComputeStructureFunctions3D.py \
       spectralDNS_tgv_incomp_Re400NumPtsPerDir128/tgv_out_Re400NumPtsPerDir128_fields/tgv_out_Re400NumPtsPerDir128_fields_s1.h5 \
       --snapshot-index 0

  3) Dedalus directory auto-detection:
     python3 python_scripts/ComputeStructureFunctions3D.py \
       spectralDNS_tgv_incomp_Re400NumPtsPerDir128 \
       --snapshot-index -1

  4) Headless / CSV only:
     python3 python_scripts/ComputeStructureFunctions3D.py <input_path> --no-plot

  5) KE check from 2nd-order SF (prints 3/4 <SF_LL>_avg estimate):
     python3 python_scripts/ComputeStructureFunctions3D.py <input_path> \
       --sf-type LL --ke-tail-frac 0.25 --no-plot

  6) Local SF_LL and compare with (2/3)*Ek*(1-f):
     python3 python_scripts/ComputeStructureFunctions3D.py <input_path> \
       --backend local --sf-type LL

  7) Plot only S2_from_f(r) = (2/3)*Ek*(1-f(r)):
     python3 python_scripts/ComputeStructureFunctions3D.py <input_path> \
       --plot-s2-from-f-only --no-csv

  8) Match FluidSF ex_3d boundary choice (periodic-x, periodic-y):
     python3 python_scripts/ComputeStructureFunctions3D.py <input_path> \
       --boundary periodic-x periodic-y

Snapshot index examples:
  --snapshot-index 0   = first saved snapshot
  --snapshot-index 5   = 6th saved snapshot
  --snapshot-index -1  = last saved snapshot (default)
  --snapshot-index -2  = second-to-last saved snapshot
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S2_FROM_F_PREFAC = 2.0 / 3.0

try:
    import fluidsf
    HAS_FLUIDSF = True
    FLUIDSF_IMPORT_ERROR = None
except Exception as exc:
    fluidsf = None
    HAS_FLUIDSF = False
    FLUIDSF_IMPORT_ERROR = exc

from ComputeFGFromVelocityFFT import (
    compute_tensor_correlations,
    detect_header_lines,
    extract_f_g,
    is_dedalus_velocity_h5,
    print_table,
    read_data_file_chunked,
    read_data_file_header,
    read_dedalus_velocity_h5,
    resolve_dedalus_input_file,
)


def _normalize_boundary(boundary):
    if boundary is None:
        return {"periodic_x": False, "periodic_y": False, "periodic_z": False}
    if isinstance(boundary, str):
        tokens = [boundary]
    else:
        tokens = list(boundary)

    p_all = any("periodic-all" in t for t in tokens)
    p_x = p_all or any("periodic-x" in t for t in tokens)
    p_y = p_all or any("periodic-y" in t for t in tokens)
    p_z = p_all or any("periodic-z" in t for t in tokens)
    return {"periodic_x": p_x, "periodic_y": p_y, "periodic_z": p_z}


def _shift_axis(arr, shift, axis, periodic):
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    src = [slice(None), slice(None), slice(None)]
    dst = [slice(None), slice(None), slice(None)]

    dst[axis] = slice(0, -shift)
    src[axis] = slice(shift, None)
    out[tuple(dst)] = arr[tuple(src)]

    if periodic:
        dst[axis] = slice(-shift, None)
        src[axis] = slice(0, shift)
        out[tuple(dst)] = arr[tuple(src)]
    return out


def _sep_range(n, periodic, max_sep=None):
    upper_exclusive = int(n / 2) if periodic else int(n - 1)
    if max_sep is not None:
        upper_exclusive = min(upper_exclusive, max_sep + 1)
    if upper_exclusive <= 1:
        return []
    return list(range(1, upper_exclusive))


def calculate_advection_3d(vx, vy, vz, dx, dy, dz):
    dudx, dudy, dudz = np.gradient(vx, dx, dy, dz, axis=(0, 1, 2))
    dvdx, dvdy, dvdz = np.gradient(vy, dx, dy, dz, axis=(0, 1, 2))
    dwdx, dwdy, dwdz = np.gradient(vz, dx, dy, dz, axis=(0, 1, 2))
    adv_x = vx * dudx + vy * dudy + vz * dudz
    adv_y = vx * dvdx + vy * dvdy + vz * dvdz
    adv_z = vx * dwdx + vy * dwdy + vz * dwdz
    return adv_x, adv_y, adv_z


def _has_type(sf_type, token):
    return any(token in t for t in sf_type)


def _compute_directional_sf(
    direction,
    shift,
    vx,
    vy,
    vz,
    adv_x,
    adv_y,
    adv_z,
    periodic_x,
    periodic_y,
    periodic_z,
    sf_type,
):
    if direction == "x":
        u_s = _shift_axis(vx, shift, axis=0, periodic=periodic_x)
        v_s = _shift_axis(vy, shift, axis=0, periodic=periodic_x)
        w_s = _shift_axis(vz, shift, axis=0, periodic=periodic_x)
        ax_s = _shift_axis(adv_x, shift, axis=0, periodic=periodic_x) if adv_x is not None else None
        ay_s = _shift_axis(adv_y, shift, axis=0, periodic=periodic_x) if adv_y is not None else None
        az_s = _shift_axis(adv_z, shift, axis=0, periodic=periodic_x) if adv_z is not None else None
        dL = u_s - vx
        dT1 = v_s - vy
        dT2 = w_s - vz
    elif direction == "y":
        u_s = _shift_axis(vx, shift, axis=1, periodic=periodic_y)
        v_s = _shift_axis(vy, shift, axis=1, periodic=periodic_y)
        w_s = _shift_axis(vz, shift, axis=1, periodic=periodic_y)
        ax_s = _shift_axis(adv_x, shift, axis=1, periodic=periodic_y) if adv_x is not None else None
        ay_s = _shift_axis(adv_y, shift, axis=1, periodic=periodic_y) if adv_y is not None else None
        az_s = _shift_axis(adv_z, shift, axis=1, periodic=periodic_y) if adv_z is not None else None
        dL = v_s - vy
        dT1 = u_s - vx
        dT2 = w_s - vz
    else:
        u_s = _shift_axis(vx, shift, axis=2, periodic=periodic_z)
        v_s = _shift_axis(vy, shift, axis=2, periodic=periodic_z)
        w_s = _shift_axis(vz, shift, axis=2, periodic=periodic_z)
        ax_s = _shift_axis(adv_x, shift, axis=2, periodic=periodic_z) if adv_x is not None else None
        ay_s = _shift_axis(adv_y, shift, axis=2, periodic=periodic_z) if adv_y is not None else None
        az_s = _shift_axis(adv_z, shift, axis=2, periodic=periodic_z) if adv_z is not None else None
        dL = w_s - vz
        dT1 = u_s - vx
        dT2 = v_s - vy

    out = {}
    if _has_type(sf_type, "LL"):
        out[f"SF_LL_{direction}"] = np.nanmean(dL**2)
    if _has_type(sf_type, "TT"):
        out[f"SF_TT_{direction}"] = np.nanmean(dT1**2 + dT2**2)
    if _has_type(sf_type, "LLL"):
        out[f"SF_LLL_{direction}"] = np.nanmean(dL**3)
    if _has_type(sf_type, "LTT"):
        out[f"SF_LTT_{direction}"] = np.nanmean(dL * (dT1**2 + dT2**2))
    if _has_type(sf_type, "ASF_V"):
        out[f"SF_advection_velocity_{direction}"] = np.nanmean(
            (ax_s - adv_x) * (u_s - vx)
            + (ay_s - adv_y) * (v_s - vy)
            + (az_s - adv_z) * (w_s - vz)
        )
    return out


def _bin_data(dd, sf, nbins):
    df = pd.DataFrame({"dd": dd, "sf": sf})
    means = df.groupby(pd.cut(df["dd"], nbins, duplicates="drop"), observed=True).mean()
    return means["dd"].values, means["sf"].values


def generate_structure_functions_3d_fluidsf(vx, vy, vz, x, y, z, sf_type, boundary, nbins=None):
    """
    Direct FluidSF backend.

    FluidSF's internal axis convention for 3D fields is (z, y, x),
    while this repo reconstructs velocity as (x, y, z). Transpose before call.
    """
    if not HAS_FLUIDSF:
        raise ImportError(
            "fluidsf (or one of its dependencies) is not importable in this environment. "
            "Install with: python3 -m pip install fluidsf "
            "or run this script with --backend local."
        ) from FLUIDSF_IMPORT_ERROR

    u = np.transpose(vx, (2, 1, 0))
    v = np.transpose(vy, (2, 1, 0))
    w = np.transpose(vz, (2, 1, 0))

    print("  Starting FluidSF structure-function computation...")
    t0 = time.perf_counter()
    sf = fluidsf.generate_structure_functions_3d(
        u, v, w, x, y, z,
        sf_type=sf_type,
        boundary=boundary,
        nbins=nbins,
    )
    t1 = time.perf_counter()
    print(f"  FluidSF computation finished in {t1 - t0:.2f} s")
    return {k: np.asarray(vv, dtype=np.float64) for k, vv in sf.items()}


def generate_structure_functions_3d(vx, vy, vz, x, y, z, sf_type, boundary, nbins=None,
                                    max_sep=None, max_sep_x=None, max_sep_y=None, max_sep_z=None):
    bc = _normalize_boundary(boundary)
    periodic_x = bc["periodic_x"]
    periodic_y = bc["periodic_y"]
    periodic_z = bc["periodic_z"]

    sep_x = _sep_range(len(x), periodic_x, max_sep=max_sep_x if max_sep_x is not None else max_sep)
    sep_y = _sep_range(len(y), periodic_y, max_sep=max_sep_y if max_sep_y is not None else max_sep)
    sep_z = _sep_range(len(z), periodic_z, max_sep=max_sep_z if max_sep_z is not None else max_sep)

    xd = np.zeros((len(sep_x) + 1,), dtype=np.float64)
    yd = np.zeros((len(sep_y) + 1,), dtype=np.float64)
    zd = np.zeros((len(sep_z) + 1,), dtype=np.float64)

    keys = []
    if _has_type(sf_type, "LL"):
        keys.extend(["SF_LL_x", "SF_LL_y", "SF_LL_z"])
    if _has_type(sf_type, "TT"):
        keys.extend(["SF_TT_x", "SF_TT_y", "SF_TT_z"])
    if _has_type(sf_type, "LLL"):
        keys.extend(["SF_LLL_x", "SF_LLL_y", "SF_LLL_z"])
    if _has_type(sf_type, "LTT"):
        keys.extend(["SF_LTT_x", "SF_LTT_y", "SF_LTT_z"])
    if _has_type(sf_type, "ASF_V"):
        keys.extend([
            "SF_advection_velocity_x",
            "SF_advection_velocity_y",
            "SF_advection_velocity_z",
        ])

    data = {k: None for k in keys}
    for k in keys:
        if k.endswith("_x"):
            data[k] = np.zeros((len(sep_x) + 1,), dtype=np.float64)
        elif k.endswith("_y"):
            data[k] = np.zeros((len(sep_y) + 1,), dtype=np.float64)
        else:
            data[k] = np.zeros((len(sep_z) + 1,), dtype=np.float64)

    adv_x = adv_y = adv_z = None
    if _has_type(sf_type, "ASF_V"):
        dx = float(np.abs(x[1] - x[0])) if len(x) > 1 else 1.0
        dy = float(np.abs(y[1] - y[0])) if len(y) > 1 else 1.0
        dz = float(np.abs(z[1] - z[0])) if len(z) > 1 else 1.0
        adv_x, adv_y, adv_z = calculate_advection_3d(vx, vy, vz, dx, dy, dz)

    for shift in sep_x:
        sf = _compute_directional_sf(
            "x", shift, vx, vy, vz, adv_x, adv_y, adv_z,
            periodic_x, periodic_y, periodic_z, sf_type
        )
        for key, val in sf.items():
            if key.endswith("_x") and key in data:
                data[key][shift] = val
        xd[shift] = np.abs(x[shift] - x[0])

    for shift in sep_y:
        sf = _compute_directional_sf(
            "y", shift, vx, vy, vz, adv_x, adv_y, adv_z,
            periodic_x, periodic_y, periodic_z, sf_type
        )
        for key, val in sf.items():
            if key.endswith("_y") and key in data:
                data[key][shift] = val
        yd[shift] = np.abs(y[shift] - y[0])

    for shift in sep_z:
        sf = _compute_directional_sf(
            "z", shift, vx, vy, vz, adv_x, adv_y, adv_z,
            periodic_x, periodic_y, periodic_z, sf_type
        )
        for key, val in sf.items():
            if key.endswith("_z") and key in data:
                data[key][shift] = val
        zd[shift] = np.abs(z[shift] - z[0])

    out = dict(data)
    out["x-diffs"] = xd
    out["y-diffs"] = yd
    out["z-diffs"] = zd

    if nbins is not None:
        for prefix, dd_key in [("x", "x-diffs"), ("y", "y-diffs"), ("z", "z-diffs")]:
            dd = out[dd_key]
            for key in [k for k in list(out.keys()) if k.endswith(f"_{prefix}")]:
                dd_bin, sf_bin = _bin_data(dd, out[key], nbins)
                out[dd_key] = dd_bin
                out[key] = sf_bin
    return out


def save_sf_csv(out_csv, sf):
    ordered = [
        "x-diffs", "y-diffs", "z-diffs",
        "SF_LL_x", "SF_LL_y", "SF_LL_z",
        "SF_TT_x", "SF_TT_y", "SF_TT_z",
        "SF_LLL_x", "SF_LLL_y", "SF_LLL_z",
        "SF_LTT_x", "SF_LTT_y", "SF_LTT_z",
        "SF_advection_velocity_x", "SF_advection_velocity_y", "SF_advection_velocity_z",
    ]
    cols = [k for k in ordered if k in sf]
    max_n = max(len(sf[k]) for k in cols)

    colw = 26
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(", ".join(name.rjust(colw) for name in cols) + "\n")
        for i in range(max_n):
            vals = []
            for key in cols:
                arr = sf[key]
                val = arr[i] if i < len(arr) else np.nan
                vals.append(f"{val:>{colw}.16e}")
            f.write(", ".join(vals) + "\n")


def _plot_signed_curve(ax, x, y, color, label):
    x = np.asarray(x)
    y = np.asarray(y)
    pos = (x > 0.0) & np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    neg = (x > 0.0) & np.isfinite(x) & np.isfinite(y) & (y < 0.0)
    if np.any(pos):
        ax.loglog(x[pos], y[pos], color=color, label=label)
    if np.any(neg):
        ax.loglog(x[neg], -y[neg], color=color, marker="x", linestyle="None")


def plot_sf_ex3d_style(sf, step_number, time_value, show=True):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharey=False)

    if "SF_LL_x" in sf:
        _plot_signed_curve(ax[0], sf["x-diffs"], sf["SF_LL_x"], "tab:blue", "x")
        _plot_signed_curve(ax[0], sf["y-diffs"], sf["SF_LL_y"], "tab:red", "y")
        _plot_signed_curve(ax[0], sf["z-diffs"], sf["SF_LL_z"], "tab:green", "z")
        ax[0].set_title("Second-order SF$_{LL}$")
    ax[0].set_xlabel("Separation distance")
    ax[0].set_ylabel("SF value")
    ax[0].legend(loc="best")

    if "SF_LLL_x" in sf:
        _plot_signed_curve(ax[1], sf["x-diffs"], sf["SF_LLL_x"], "tab:blue", "x")
        _plot_signed_curve(ax[1], sf["y-diffs"], sf["SF_LLL_y"], "tab:red", "y")
        _plot_signed_curve(ax[1], sf["z-diffs"], sf["SF_LLL_z"], "tab:green", "z")
        ax[1].set_title("Third-order SF$_{LLL}$")
    ax[1].set_xlabel("Separation distance")
    ax[1].legend(loc="best")

    if "SF_advection_velocity_x" in sf:
        _plot_signed_curve(
            ax[2], sf["x-diffs"], sf["SF_advection_velocity_x"], "tab:blue", "x"
        )
        _plot_signed_curve(
            ax[2], sf["y-diffs"], sf["SF_advection_velocity_y"], "tab:red", "y"
        )
        _plot_signed_curve(
            ax[2], sf["z-diffs"], sf["SF_advection_velocity_z"], "tab:green", "z"
        )
        ax[2].set_title("Advective SF")
    ax[2].set_xlabel("Separation distance")
    ax[2].legend(loc="best")

    fig.suptitle(f"3D structure functions, step={step_number}, t={time_value:.4e}")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sf_with_f_comparison(sf, s2_compare, step_number, time_value, show=True):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _plot_signed_curve(ax, sf["x-diffs"], sf["SF_LL_x"], "tab:blue", "SF_LL_x local")
    _plot_signed_curve(ax, sf["y-diffs"], sf["SF_LL_y"], "tab:red", "SF_LL_y local")
    _plot_signed_curve(ax, sf["z-diffs"], sf["SF_LL_z"], "tab:green", "SF_LL_z local")

    ax.plot(s2_compare["r_fft"], s2_compare["pred_x"], "--", color="tab:blue", lw=1.8, label="(2/3)*Ek*(1-f_x)")
    ax.plot(s2_compare["r_fft"], s2_compare["pred_y"], "--", color="tab:red", lw=1.8, label="(2/3)*Ek*(1-f_y)")
    ax.plot(s2_compare["r_fft"], s2_compare["pred_z"], "--", color="tab:green", lw=1.8, label="(2/3)*Ek*(1-f_z)")

    if "r_avg" in s2_compare:
        _plot_signed_curve(ax, s2_compare["r_avg"], s2_compare["s2_local_avg"], "black", "SF_LL_avg local")
        _plot_signed_curve(ax, s2_compare["r_avg"], s2_compare["s2_pred_avg"], "gray", "(2/3)*Ek*(1-f_avg)")

    ax.set_xlabel("Separation distance")
    ax.set_ylabel("Second-order SF")
    ax.set_title(f"SF_LL local vs correlation model, step={step_number}, t={time_value:.4e}")
    ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_s2_from_f_only(vx, vy, vz, dx, dy, dz):
    """
    Build the single averaged correlation-model curve:
      S2_from_f(r) = (2/3) * Ek * (1 - f(r))
    where f(r) is the averaged longitudinal correlation extracted from Rij.
    """
    R, _ = compute_tensor_correlations(vx, vy, vz)
    fg = extract_f_g(R, dx, dy, dz, max_r_points=None)
    Ek = 0.5 * np.mean(vx * vx + vy * vy + vz * vz)
    s2 = S2_FROM_F_PREFAC * Ek * (1.0 - np.asarray(fg["f"], dtype=np.float64))
    return {
        "r": np.asarray(fg["r"], dtype=np.float64),
        "f": np.asarray(fg["f"], dtype=np.float64),
        "s2": s2,
        "Ek": float(Ek),
    }


def plot_s2_from_f_only(s2_model, step_number, time_value, show=True):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(s2_model["r"], s2_model["s2"], color="black", lw=2.2)
    ax.set_xlabel("Separation distance")
    ax.set_ylabel("S2_from_f(r)")
    ax.set_title(f"S2_from_f(r), step={step_number}, t={time_value:.4e}")
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_ke_from_second_order_sf(sf, vx, vy, vz, ke_tail_frac=0.25):
    """
    Estimate KE from second-order longitudinal SF using:
      KE_est = (3/4) * <SF_LL>_avg
    where <SF_LL>_avg is averaged over x/y/z and over a tail window in r.

    The KE reference is the fluctuation kinetic energy:
      KE_fluct = 0.5 * <u'^2 + v'^2 + w'^2>.
    """
    required = ("SF_LL_x", "SF_LL_y", "SF_LL_z")
    if not all(k in sf for k in required):
        return None

    if ke_tail_frac <= 0.0 or ke_tail_frac > 1.0:
        raise ValueError("--ke-tail-frac must be in (0, 1].")

    ux = vx - np.mean(vx)
    uy = vy - np.mean(vy)
    uz = vz - np.mean(vz)
    ke_fluct = 0.5 * np.mean(ux * ux + uy * uy + uz * uz)

    def tail_mean_std(arr):
        arr = np.asarray(arr, dtype=np.float64)
        finite_idx = np.where(np.isfinite(arr))[0]
        if finite_idx.size == 0:
            return np.nan, np.nan, 0
        vals = arr[finite_idx]
        n_tail = max(1, int(np.ceil(ke_tail_frac * vals.size)))
        tail = vals[-n_tail:]
        return float(np.nanmean(tail)), float(np.nanstd(tail)), int(n_tail)

    s2x_m, s2x_s, nx = tail_mean_std(sf["SF_LL_x"])
    s2y_m, s2y_s, ny = tail_mean_std(sf["SF_LL_y"])
    s2z_m, s2z_s, nz = tail_mean_std(sf["SF_LL_z"])

    s2_avg = np.nanmean([s2x_m, s2y_m, s2z_m])
    ke_est = 0.75 * s2_avg
    abs_diff = abs(ke_est - ke_fluct)
    rel_diff = abs_diff / abs(ke_fluct) if abs(ke_fluct) > 0.0 else np.nan

    return {
        "s2x_tail_mean": s2x_m,
        "s2y_tail_mean": s2y_m,
        "s2z_tail_mean": s2z_m,
        "s2x_tail_std": s2x_s,
        "s2y_tail_std": s2y_s,
        "s2z_tail_std": s2z_s,
        "tail_count_x": nx,
        "tail_count_y": ny,
        "tail_count_z": nz,
        "s2_avg_tail_mean": s2_avg,
        "ke_est_from_3over4_s2": ke_est,
        "ke_fluct": float(ke_fluct),
        "abs_diff": float(abs_diff),
        "rel_diff": float(rel_diff),
        "ke_tail_frac": float(ke_tail_frac),
    }


def _interp_with_nan(x_src, y_src, x_dst):
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    x_dst = np.asarray(x_dst, dtype=np.float64)

    finite = np.isfinite(x_src) & np.isfinite(y_src)
    if np.count_nonzero(finite) < 2:
        return np.full_like(x_dst, np.nan, dtype=np.float64)

    xs = x_src[finite]
    ys = y_src[finite]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    out = np.full_like(x_dst, np.nan, dtype=np.float64)
    mask = (x_dst >= xs[0]) & (x_dst <= xs[-1]) & np.isfinite(x_dst)
    if np.any(mask):
        out[mask] = np.interp(x_dst[mask], xs, ys)
    return out


def compute_second_order_sf_from_f_comparison(sf, vx, vy, vz, dx, dy, dz):
    """
    Compare second-order longitudinal SF from the script against a correlation-based
    prediction built from the same velocity field:
      S2_from_f(r) = (2/3) * Ek * (1 - f(r))

    Here Ek = 0.5 * <u_i u_i>.
    """
    required = (
        "SF_LL_x", "SF_LL_y", "SF_LL_z",
        "x-diffs", "y-diffs", "z-diffs",
    )
    if not all(key in sf for key in required):
        return None

    R, _ = compute_tensor_correlations(vx, vy, vz)
    fg = extract_f_g(R, dx, dy, dz, max_r_points=None)
    Ek = 0.5 * np.mean(vx * vx + vy * vy + vz * vz)

    pred_curves = {
        "x": S2_FROM_F_PREFAC * Ek * (1.0 - np.asarray(fg["f_x_norm"], dtype=np.float64)),
        "y": S2_FROM_F_PREFAC * Ek * (1.0 - np.asarray(fg["f_y_norm"], dtype=np.float64)),
        "z": S2_FROM_F_PREFAC * Ek * (1.0 - np.asarray(fg["f_z_norm"], dtype=np.float64)),
    }

    comparison = {
        "Ek": float(Ek),
        "r_fft": np.asarray(fg["r"], dtype=np.float64),
        "pred_x": pred_curves["x"],
        "pred_y": pred_curves["y"],
        "pred_z": pred_curves["z"],
    }

    rows = []
    axis_stats = {}
    for direction in ("x", "y", "z"):
        r_local = np.asarray(sf[f"{direction}-diffs"], dtype=np.float64)
        s2_local = np.asarray(sf[f"SF_LL_{direction}"], dtype=np.float64)
        s2_pred = _interp_with_nan(comparison["r_fft"], pred_curves[direction], r_local)
        mask = np.isfinite(r_local) & np.isfinite(s2_local) & np.isfinite(s2_pred)
        if np.any(mask):
            diff = s2_local[mask] - s2_pred[mask]
            max_abs = float(np.max(np.abs(diff)))
            rms = float(np.sqrt(np.mean(diff * diff)))
            mean_abs = float(np.mean(np.abs(diff)))
            npts = int(np.count_nonzero(mask))
        else:
            max_abs = np.nan
            rms = np.nan
            mean_abs = np.nan
            npts = 0

        axis_stats[direction] = {
            "r": r_local,
            "s2_local": s2_local,
            "s2_pred": s2_pred,
            "npts": npts,
            "max_abs": max_abs,
            "rms": rms,
            "mean_abs": mean_abs,
        }
        rows.append([direction, str(npts), f"{max_abs:.6e}", f"{rms:.6e}", f"{mean_abs:.6e}"])

    same_grid = (
        len(sf["x-diffs"]) == len(sf["y-diffs"]) == len(sf["z-diffs"])
        and np.allclose(sf["x-diffs"], sf["y-diffs"])
        and np.allclose(sf["x-diffs"], sf["z-diffs"])
    )
    if same_grid:
        r_avg = np.asarray(sf["x-diffs"], dtype=np.float64)
        s2_local_stack = np.vstack([
            np.asarray(sf["SF_LL_x"], dtype=np.float64),
            np.asarray(sf["SF_LL_y"], dtype=np.float64),
            np.asarray(sf["SF_LL_z"], dtype=np.float64),
        ])
        s2_pred_stack = np.vstack([
            axis_stats["x"]["s2_pred"],
            axis_stats["y"]["s2_pred"],
            axis_stats["z"]["s2_pred"],
        ])
        valid_axes = np.isfinite(s2_pred_stack)
        s2_local_avg = np.nanmean(np.where(valid_axes, s2_local_stack, np.nan), axis=0)
        s2_pred_avg = np.nanmean(np.where(valid_axes, s2_pred_stack, np.nan), axis=0)
        mask = np.isfinite(r_avg) & np.isfinite(s2_local_avg) & np.isfinite(s2_pred_avg)
        if np.any(mask):
            diff = s2_local_avg[mask] - s2_pred_avg[mask]
            max_abs = float(np.max(np.abs(diff)))
            rms = float(np.sqrt(np.mean(diff * diff)))
            mean_abs = float(np.mean(np.abs(diff)))
            npts = int(np.count_nonzero(mask))
        else:
            max_abs = np.nan
            rms = np.nan
            mean_abs = np.nan
            npts = 0
        rows.append(["avg", str(npts), f"{max_abs:.6e}", f"{rms:.6e}", f"{mean_abs:.6e}"])
        comparison["r_avg"] = r_avg
        comparison["s2_local_avg"] = s2_local_avg
        comparison["s2_pred_avg"] = s2_pred_avg

    comparison["rows"] = rows
    comparison["axis_stats"] = axis_stats
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compute 3D structure functions from velocity data (default backend: fluidsf)."
    )
    parser.add_argument("data_file", type=str, help="Input text/HDF5 velocity file or Dedalus directory")
    parser.add_argument("--header-lines", type=int, default=None,
                        help="Header lines for text files (auto-detect if omitted)")
    parser.add_argument("--snapshot-index", type=int, default=-1,
                        help="For Dedalus HDF5 tasks/u: snapshot index in time dimension (default: -1)")
    parser.add_argument("--chunk-size", type=int, default=5_000_000,
                        help="Chunk size for loading text/HDF5 sampled data")
    parser.add_argument("--decimals", type=int, default=10,
                        help="Coordinate rounding precision for grid reconstruction")
    parser.add_argument("--sf-type", nargs="+", default=["ASF_V", "LL", "LLL", "LTT"],
                        help="Structure function types, e.g. ASF_V LL LLL LTT")
    parser.add_argument("--backend", type=str, default="fluidsf", choices=["fluidsf", "local", "auto"],
                        help="Structure-function backend (default: fluidsf)")
    parser.add_argument("--boundary", nargs="+", default=["periodic-all"],
                        help="Boundary tags: periodic-all periodic-x periodic-y periodic-z")
    parser.add_argument("--nbins", type=int, default=None,
                        help="Optional number of bins for SF curves")
    parser.add_argument("--ke-tail-frac", type=float, default=0.25,
                        help="Tail fraction in r used for KE estimate from 3/4<SF_LL>_avg (default: 0.25)")
    parser.add_argument("--max-sep", type=int, default=None,
                        help="Optional max separation index for all directions")
    parser.add_argument("--max-sep-x", type=int, default=None,
                        help="Optional max separation index in x direction")
    parser.add_argument("--max-sep-y", type=int, default=None,
                        help="Optional max separation index in y direction")
    parser.add_argument("--max-sep-z", type=int, default=None,
                        help="Optional max separation index in z direction")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output prefix (default: <input>_sf3d)")
    parser.add_argument("--plot-s2-from-f-only", action="store_true",
                        help="Plot only the averaged model S2_from_f(r) = (2/3)*Ek*(1-f(r)) and skip the multi-panel SF plots")
    parser.add_argument("--no-csv", action="store_true",
                        help="Do not save CSV output")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not open interactive plots")
    args = parser.parse_args()

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
                task_name="u",
            )
        )
    else:
        if args.data_file.endswith(".h5"):
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
            decimals=args.decimals,
        )

    print(f"  Velocity grid shape: {vx.shape}")
    print(f"  Grid spacing: dx={dx:.8f}, dy={dy:.8f}, dz={dz:.8f}")
    print(f"  SF types: {args.sf_type}")
    print(f"  Boundary: {args.boundary}")
    backend = args.backend
    if backend == "auto":
        backend = "fluidsf" if HAS_FLUIDSF else "local"
    print(f"  Backend: {backend}")

    s2_from_f_only = None
    if args.plot_s2_from_f_only:
        s2_from_f_only = compute_s2_from_f_only(
            vx, vy, vz, dx, dy, dz
        )
        print("Single-curve model to plot:")
        print_table(
            ["Metric", "Value"],
            [
                ["model", "S2_from_f(r) = (2/3) * Ek * (1 - f(r))"],
                ["Ek", f"{s2_from_f_only['Ek']:.12e}"],
            ],
        )

    if backend == "fluidsf":
        if args.max_sep is not None or args.max_sep_x is not None or args.max_sep_y is not None or args.max_sep_z is not None:
            print("  Note: --max-sep* options are ignored by fluidsf backend.")
        sf = generate_structure_functions_3d_fluidsf(
            vx, vy, vz, x_coords, y_coords, z_coords,
            sf_type=args.sf_type,
            boundary=args.boundary,
            nbins=args.nbins,
        )
    else:
        sf = generate_structure_functions_3d(
            vx, vy, vz, x_coords, y_coords, z_coords,
            sf_type=args.sf_type,
            boundary=args.boundary,
            nbins=args.nbins,
            max_sep=args.max_sep,
            max_sep_x=args.max_sep_x,
            max_sep_y=args.max_sep_y,
            max_sep_z=args.max_sep_z,
        )

    s2_compare = None
    if not args.plot_s2_from_f_only:
        summary_rows = []
        for key in ["SF_LL_x", "SF_LL_y", "SF_LL_z", "SF_LLL_x", "SF_LLL_y", "SF_LLL_z",
                    "SF_LTT_x", "SF_LTT_y", "SF_LTT_z",
                    "SF_advection_velocity_x", "SF_advection_velocity_y", "SF_advection_velocity_z"]:
            if key in sf:
                arr = sf[key]
                nfinite = int(np.isfinite(arr).sum())
                summary_rows.append([key, str(len(arr)), str(nfinite), f"{np.nanmin(arr):.6e}", f"{np.nanmax(arr):.6e}"])
        print("Computed structure-function ranges:")
        print_table(["Key", "N", "N_finite", "min", "max"], summary_rows)

        ke_sf = compute_ke_from_second_order_sf(
            sf, vx, vy, vz, ke_tail_frac=args.ke_tail_frac
        )
        if ke_sf is not None:
            print("KE from second-order SF (isotropic large-r relation):")
            print_table(
                ["Metric", "Value"],
                [
                    [f"SF_LL_x tail mean (n={ke_sf['tail_count_x']})", f"{ke_sf['s2x_tail_mean']:.12e}"],
                    [f"SF_LL_y tail mean (n={ke_sf['tail_count_y']})", f"{ke_sf['s2y_tail_mean']:.12e}"],
                    [f"SF_LL_z tail mean (n={ke_sf['tail_count_z']})", f"{ke_sf['s2z_tail_mean']:.12e}"],
                    ["SF_LL_avg tail mean", f"{ke_sf['s2_avg_tail_mean']:.12e}"],
                    ["(3/4) * SF_LL_avg tail", f"{ke_sf['ke_est_from_3over4_s2']:.12e}"],
                    ["KE_fluct (0.5<u'^2+v'^2+w'^2>)", f"{ke_sf['ke_fluct']:.12e}"],
                    ["abs_diff", f"{ke_sf['abs_diff']:.3e}"],
                    ["rel_diff", f"{100.0*ke_sf['rel_diff']:.3e}%"],
                    ["tail_fraction", f"{ke_sf['ke_tail_frac']:.3f}"],
                ],
            )
        else:
            print("KE from second-order SF: skipped (SF_LL_x/y/z not requested).")

        if all(key in sf for key in ("SF_LL_x", "SF_LL_y", "SF_LL_z")):
            s2_compare = compute_second_order_sf_from_f_comparison(
                sf, vx, vy, vz, dx, dy, dz
            )
            if s2_compare is not None:
                print("Second-order SF comparison with correlation model:")
                print_table(
                    ["Metric", "Value"],
                    [
                        ["Ek", f"{s2_compare['Ek']:.12e}"],
                        ["model", "S2_from_f = (2/3) * Ek * (1 - f)"],
                    ],
                )
                print("Local SF_LL vs correlation-model mismatch:")
                print_table(["Dir", "N_overlap", "max_abs", "rms", "mean_abs"], s2_compare["rows"])

    input_base = os.path.splitext(args.data_file)[0]
    out_prefix = args.out_prefix if args.out_prefix else input_base + "_sf3d"
    out_csv = out_prefix + ".csv"

    if not args.no_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        save_sf_csv(out_csv, sf)
        print(f"Saved structure-functions CSV: {out_csv}")
    else:
        print("CSV saving disabled (--no-csv).")

    if not args.no_plot:
        if args.plot_s2_from_f_only and s2_from_f_only is not None:
            plot_s2_from_f_only(s2_from_f_only, step_number, time_value, show=True)
        else:
            plot_sf_ex3d_style(sf, step_number, time_value, show=True)
            if s2_compare is not None:
                plot_sf_with_f_comparison(sf, s2_compare, step_number, time_value, show=True)
    else:
        print("Plotting disabled (--no-plot).")


if __name__ == "__main__":
    main()
