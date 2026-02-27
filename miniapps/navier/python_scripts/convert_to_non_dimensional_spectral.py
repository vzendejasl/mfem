#!/usr/bin/env python3
"""
spectral_run_analysis.py

Compute raw dK/dt, effective viscosity/Re, and non-dimensional
placeholders from TGV CSV output such as:

    Time, Cycle, KineticEnergy, Enstrophy
    0.0, 0.0, 1.25e-1, 1.48e+01
    ...

Usage:
    python spectral_run_analysis.py --U0 1.0 --domain-length 6.28318 path/to/tgv_out.csv [...]
"""

import os
import sys
import argparse
from typing import List

import pandas as pd
import numpy as np


# ------------------------------ Helpers -------------------------------- #

def normalize_column_names(cols: List[str]) -> List[str]:
    """
    Normalize header names to: time, cycle, kinetic_energy, enstrophy, etc.
    Handles variants like:
      - Time, time, TIME
      - Cycle, cycle
      - KineticEnergy, kinetic_energy, ke, KE
      - Enstrophy, enstrophy
    """
    norm = []
    for c in cols:
        c0 = c.strip().lower()

        if c0 == "time":
            norm.append("time")
        elif c0 == "cycle":
            norm.append("cycle")
        # Explicit mapping for kinetic energy
        elif c0 in ("ke", "kineticenergy", "kinetic_energy"):
            norm.append("kinetic_energy")
        elif "kinetic" in c0 and "energy" in c0:
            norm.append("kinetic_energy")
        elif "enstrophy" in c0:
            norm.append("enstrophy")
        else:
            norm.append(c0.replace(" ", "_"))
    return norm


def dedupe_by_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate rows by exact 'cycle', keeping the last occurrence of each cycle.
    This follows the idea of the original script that used the cycle number.
    """
    required_cols = ["cycle", "time", "kinetic_energy", "enstrophy"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Coerce to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["cycle", "time", "kinetic_energy", "enstrophy"])

    # Stable sort by cycle
    df = df.sort_values("cycle", kind="mergesort")

    # Drop duplicates in cycle, keep last
    mask = ~df["cycle"].duplicated(keep="last")
    df = df[mask].copy()

    removed = before - len(df)
    if removed > 0:
        print(f"[i] Removed {removed} duplicate/non-numeric rows based on cycle.", file=sys.stderr)

    df.reset_index(drop=True, inplace=True)
    return df


# ------------------------------ Processing ------------------------------ #

def process_file(
    fname: str,
    U0: float,
    domain_length: float,
) -> None:
    """
    Read one CSV file, clean & dedupe rows by cycle, compute metrics, write *_processed.csv
    """
    # Characteristic length L = domain_length/(2π)
    L = domain_length / (2.0 * np.pi)

    # 1) Read CSV and normalize headers
    try:
        raw_df = pd.read_csv(fname, skipinitialspace=True)
    except Exception as e:
        print(f"ERROR reading {fname}: {e}", file=sys.stderr)
        return

    if raw_df.empty:
        print(f"ERROR: {fname} appears empty.", file=sys.stderr)
        return

    raw_cols = list(raw_df.columns)
    norm_cols = normalize_column_names(raw_cols)
    raw_df.columns = norm_cols

    required_cols = ["time", "cycle", "kinetic_energy", "enstrophy"]
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in {fname}: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {list(raw_df.columns)}", file=sys.stderr)
        return

    # 2) Deduplicate by cycle
    df = dedupe_by_cycle(raw_df)

    # 3) Ensure numeric types (already numeric from dedupe, but enforce)
    df["time"] = df["time"].astype(float)
    df["kinetic_energy"] = df["kinetic_energy"].astype(float)
    df["enstrophy"] = df["enstrophy"].astype(float)

    # 4) Raw dK/dt, guard against zero dt
    dt = df["time"].diff()
    dt = dt.replace(0.0, np.nan)
    df["dk_dt"] = df["kinetic_energy"].diff() / dt
    df["dk_dt"] = df["dk_dt"].bfill().ffill()

    # 5) Effective viscosity
    eps_n = 1e-30
    ens_safe = df["enstrophy"].where(df["enstrophy"].abs() > eps_n, np.nan)

    df["effective_nu_half"] = -df["dk_dt"] / (2.0 * ens_safe)
    df["effective_nu"] = -df["dk_dt"] / (ens_safe)

    df[["effective_nu_half", "effective_nu"]] = \
        df[["effective_nu_half", "effective_nu"]].bfill().ffill()

    # 6) Reynolds numbers
    df["Re_half"] = (U0 * L) / df["effective_nu_half"]
    df["Re"] = (U0 * L) / df["effective_nu"]

    # 7) Alternative ReL diagnostic
    eps = 1e-10
    denom = np.maximum(df["dk_dt"]**2, eps)
    df["ReL"] = 2.0 * df["enstrophy"] * (df["kinetic_energy"]**2) / denom

    # 8) Non dimensional placeholders
    df["t_star"] = df["time"] * U0 / L
    df["dk_dt_star"] = df["dk_dt"] / (U0**3 / L)
    df["enstrophy_star"] = df["enstrophy"] * 2.0 / (U0 / L)**2
    df["kinetic_energy_star"] = df["kinetic_energy"] / (U0**2)

    # 9) Write output
    out_dir = os.path.dirname(os.path.abspath(fname))
    base, _ = os.path.splitext(os.path.basename(fname))
    out_path = os.path.join(out_dir, f"{base}_processed.csv")

    with open(out_path, "w") as f:
        header = ",   ".join(df.columns)
        f.write(header + "\n")
        df.to_csv(f, index=False, header=False, float_format="%.12e")

    print(f"[✓] Processed {fname} -> {out_path}")
    print(f"    Output columns: {list(df.columns)}", file=sys.stderr)


# -------------------------------  CLI  ---------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="Compute raw and normalized metrics from TGV CSV output."
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
        "files", nargs="+",
        help="One or more input CSV files"
    )
    args = p.parse_args()

    print(f"Processing with U0={args.U0}, domain_length={args.domain_length}")

    for f in args.files:
        if not os.path.isfile(f):
            print(f"[!] Warning: '{f}' not found, skipping", file=sys.stderr)
            continue
        process_file(f, args.U0, args.domain_length)


if __name__ == "__main__":
    main()
