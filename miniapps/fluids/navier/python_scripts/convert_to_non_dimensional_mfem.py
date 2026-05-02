#!/usr/bin/env python3
"""
run_analysis.py
Compute raw dK/dt, effective viscosity/Re, and a few non-dimensional
placeholders from TGV log tables — while removing duplicate/restart rows.

Usage:
    python run_analysis.py --U0 1.0 --domain-length 6.28318 path/to/tgv_data.txt [...]

Notes:
- Deduplication keeps the *last* occurrence of each time (typical restart case).
- If you have near-duplicates (floating noise), use --round-time-decimals (e.g., 12).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional

# ------------------------------ Helpers -------------------------------- #

def find_table_header(fname: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Scan file for the line that starts the numeric table.
    We look for a line that begins with 'time' and contains 'kinetic' and 'enstrophy'.
    Returns (header_idx, header_line) or (None, None) if not found.
    """
    header_idx, header_line = None, None
    with open(fname, "r") as fh:
        for i, line in enumerate(fh):
            s = line.strip().lower()
            if s.startswith("time") and "kinetic" in s and "enstrophy" in s:
                header_idx, header_line = i, line.rstrip("\n")
                break
    return header_idx, header_line


def normalize_header_names(header_line: str) -> list:
    """
    Build clean, machine-friendly column names from the raw header line.
    Keeps order and splits on whitespace.
    """
    hdr = header_line.lower()
    # Normalize a few expected multi-word column names:
    hdr = hdr.replace("kinetic energy", "kinetic_energy")
    hdr = hdr.replace("taylor length scale (aniso)", "taylor_length_scale_aniso")
    hdr = hdr.replace("taylor length scale", "taylor_length_scale")
    hdr = hdr.replace("kolmogorov length scale", "kolmogorov_length_scale")
    hdr = hdr.replace("kolmogorov time scale", "kolmogorov_time_scale")
    hdr = hdr.replace("dofs per component", "dofs_per_component")
    # Squeeze spaces
    hdr = re.sub(r"\s+", " ", hdr).strip()
    names = hdr.split(" ")
    return names


def read_table(fname: str) -> Tuple[pd.DataFrame, list]:
    """
    Find and read the numeric table (rows under the header line).
    Returns (df, names). Does not dedupe yet.
    """
    header_idx, header_line = find_table_header(fname)
    if header_idx is None:
        raise RuntimeError(f"Could not find table header in {fname}")

    names = normalize_header_names(header_line)

    df = pd.read_csv(
        fname,
        sep=r"\s+",
        engine="python",
        header=None,
        skiprows=header_idx + 1,  # start right after header
        names=names
    )
    return df, names


def dedupe_by_time(
    df: pd.DataFrame,
    round_time_decimals: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    - Coerce key columns to numeric (drops any mid-file headers copied as data).
    - Stable-sort by time.
    - Drop duplicate times, keeping the last one (common after restarts).

    If round_time_decimals is provided (e.g., 12), we round time before dedup
    to collapse near-duplicates caused by tiny FP differences.
    """
    # Columns that must exist to compute metrics later
    required_cols = ['time', 'kinetic_energy', 'enstrophy']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Coerce to numeric to kill mid-file headers; keep original df pointer only for diagnostics
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    before = len(df)
    df = df.dropna(subset=['time', 'kinetic_energy', 'enstrophy'])
    # Stable sort
    df = df.sort_values('time', kind='mergesort')

    # Dedup key
    key = df['time'].round(round_time_decimals) if round_time_decimals is not None else df['time']
    mask = ~key.duplicated(keep='last')
    df = df[mask].copy()

    removed = before - len(df)
    if verbose and removed > 0:
        msg = f"[i] Removed {removed} duplicate/non-numeric rows based on time"
        if round_time_decimals is not None:
            msg += f" (rounded to {round_time_decimals} decimals)"
        print(msg + ".", file=sys.stderr)

    df.reset_index(drop=True, inplace=True)
    return df


# ------------------------------ Processing ------------------------------ #

def process_file(fname: str, U0: float, domain_length: float, round_time_decimals: Optional[int]) -> None:
    """
    Read one file, clean & dedupe rows by time, compute metrics, write *_processed.csv
    """
    # Characteristic length L = domain_length/(2π) (your original definition)
    L = domain_length / (2.0 * np.pi)

    # 1) Read table
    try:
        df, names = read_table(fname)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return

    # Make sure required columns exist; we allow extra columns like 'cycle','cfl'
    required_cols = ['time', 'kinetic_energy', 'enstrophy']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in {fname}: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        return

    # 2) Deduplicate & clean
    df = dedupe_by_time(df, round_time_decimals=round_time_decimals, verbose=True)

    # 3) Ensure numeric types
    df['time'] = df['time'].astype(float)
    df['kinetic_energy'] = df['kinetic_energy'].astype(float)
    df['enstrophy'] = df['enstrophy'].astype(float)

    # 4) Raw derivatives
    #    Guard against zero dt to avoid inf/NaN explosions
    dt = df['time'].diff()
    dt = dt.replace(0.0, np.nan)

    df['dk_dt'] = df['kinetic_energy'].diff() / dt

    # Back/forward fill edge cases so first row isn’t NaN
    df['dk_dt'] = df['dk_dt'].bfill().ffill()

    # 5) Effective viscosity choices
    # Using small epsilon to avoid division by zero when enstrophy ≈ 0
    eps_n = 1e-30
    ens_safe = df['enstrophy'].where(df['enstrophy'].abs() > eps_n, np.nan)

    # ν_eff_half = -(dK/dt) / (2 Ω)
    df['effective_nu'] = -df['dk_dt'] / (2.0 * ens_safe)

    df[['effective_nu']] = \
        df[['effective_nu']].bfill().ffill()

    # 6) Reynolds numbers
    # Re_half = U0*L / ν_eff_half ; Re = U0*L / ν_eff
    df['Re'] = (U0 * L) / df['effective_nu']

    # 7) Example alternative “ReL” diagnostic (safe denominator)
    eps = 1e-10
    denom = np.maximum(df['dk_dt']**2, eps)
    df['ReL'] = 2.0 * df['enstrophy'] * (df['kinetic_energy']**2) / denom

    # 8) Non-dimensional placeholders
    # t* = t * U0 / L
    df['t_star'] = df['time'] * U0 / L

    # (dK/dt)* = (dK/dt) / (U0^3 / L)
    df['dk_dt_star'] = df['dk_dt'] / (U0**3 / L)

    # enstrophy* = 2 Ω / (U0/L)^2
    df['enstrophy_star'] = df['enstrophy'] * 2.0 / (U0 / L)**2

    # K* = K / U0^2
    df['kinetic_energy_star'] = df['kinetic_energy'] / (U0**2)

    denom = np.maximum(-df['dk_dt'], eps)
    df["L_int"] = (df['kinetic_energy']**1.5) / denom

    # 9) Write output (comma-separated, with nice spacing in header)
    out_dir = os.path.dirname(os.path.abspath(fname))
    base, _ = os.path.splitext(os.path.basename(fname))
    out_path = os.path.join(out_dir, f"{base}_processed.csv")

    with open(out_path, 'w') as f:
        # Header with extra spaces after commas (as requested)
        header = ',   '.join(df.columns)
        f.write(header + '\n')
        # Data
        df.to_csv(f, index=False, header=False, float_format="%.12e")

    print(f"[✓] Processed {fname} -> {out_path}")
    print(f"    Output columns: {list(df.columns)}", file=sys.stderr)


# -------------------------------  CLI  ---------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="Compute raw and normalized metrics from TGV log tables, removing duplicate/restart rows."
    )
    p.add_argument(
        "--U0", type=float, required=True,
        help="Initial velocity scale U0"
    )
    p.add_argument(
        "--domain-length", type=float, required=True,
        help="Domain length (e.g., 2π for a 0..2π box)"
    )
    p.add_argument(
        "--round-time-decimals", type=int, default=None,
        help="If set (e.g., 12), round 'time' before dedup to collapse near-duplicates."
    )
    p.add_argument(
        "files", nargs="+",
        help="One or more input files containing the TGV table"
    )

    args = p.parse_args()
    print(f"Processing with U0={args.U0}, domain_length={args.domain_length}, "
          f"round_time_decimals={args.round_time_decimals}")

    for f in args.files:
        if not os.path.isfile(f):
            print(f"[!] Warning: '{f}' not found, skipping", file=sys.stderr)
            continue
        process_file(f, args.U0, args.domain_length, args.round_time_decimals)


if __name__ == "__main__":
    main()
