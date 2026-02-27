#!/usr/bin/env python3
"""
run_analysis.py

Compute raw dk/dt, effective viscosity and Re, plus insert
placeholders for non‐dimensional (%_star) quantities.

Usage:
    python run_analysis.py --mach M --domain-length L path/to/tgv_data.csv [...]
"""
import os
import sys
import argparse

import pandas as pd
import numpy as np

def process_file(fname: str, mach: float, domain_length: float) -> None:
    L = domain_length/(2*np.pi)

    # --- 1) Read and clean up
    df = pd.read_csv(fname, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['time'] = df['time'].astype(float)

    # --- 2) Raw derivatives
    dt = df['time'].diff()                    # Δt, first row = NaN
    df['dk_dt'] = df['kinetic_energy'].diff() / dt
    df['effective_nu_half'] = -df['dk_dt'] / (2.0 * df['enstrophy'])
    df['effective_nu'] = -df['dk_dt'] / (df['enstrophy'])
    # NOTE: we'll compute Re once U0 is defined below
    # back‐fill first‐row NaNs so nothing is missing
    df[['dk_dt','effective_nu']] = df[['dk_dt','effective_nu']].bfill()
    df[['dk_dt','effective_nu_half']] = df[['dk_dt','effective_nu_half']].bfill()

    # --- 2b) ReL as requested:
    # ReL = 2 * enstrophy * (kinetic_energy**2) / (-(dk_dt**2))
    # Add a tiny epsilon to avoid division by zero
    eps = 1e-10
    denom = df['dk_dt']**2 - eps
    df['ReL'] = 2.0 * df['enstrophy'] * (df['kinetic_energy']**2) / denom

    # --- 3) Placeholders for non‐dimensionalization
    U0  = 1.0

    # Reynolds number
    df['Re'] = (U0 * L) / df['effective_nu']
    df['Re_half'] = (U0 * L) / df['effective_nu_half']

    # Example non‐dimensional quantities (fill or adjust as needed):
    # t* = t * U0 / L
    df['t_star'] = df['time'] * U0 / L

    # (dK/dt)* = (dK/dt) / (U0^3 / L)
    df['dk_dt_star'] = df['dk_dt'] / (U0**3 / L)

    # ω* = 2.0*ω / (U0/L)^2
    df['enstrophy_star'] = df['enstrophy'] * 2.0 / (U0 / L)**2
        
    df['kinetic_energy_star'] = df['kinetic_energy'] / (U0)**2

    # --- 5) Write out full processed file
    out_dir   = os.path.dirname(os.path.abspath(fname))
    base, ext = os.path.splitext(os.path.basename(fname))
    out_name  = f"{base}_processed{ext}"
    out_path  = os.path.join(out_dir, out_name)

    with open(out_path, 'w') as f:
        # Write header with extra spaces after commas
        header = ',   '.join(df.columns)
        f.write(header + '\n')
        # Write the data rows (no header)
        df.to_csv(f, index=False, header=False, float_format="%.12e")
    print(f"[✔] Processed {fname} → {out_path}")

    # --- 6) Energy conservation file
    # Create a second DataFrame from the already cleaned df
    df2 = df[['time', 't_star', 'cycle', 'kinetic_energy', 'internal_energy']].copy()

    # Total energy
    df2['IE + KE'] = df2['internal_energy'] + df2['kinetic_energy']

    # Normalized total energy relative to initial value
    # Use the first row as the reference
    initial_total_energy = df2['IE + KE'].iloc[0]
    if initial_total_energy == 0:
        # Avoid division by zero; you can choose a different behavior if you prefer
        df2['(IE + KE)_normalized'] = np.nan
    else:
        df2['(IE + KE)_normalized'] = df2['IE + KE'] / initial_total_energy

    # Write out second CSV
    out_name  = f"{base}_processed_energy_conservation{ext}"
    out_path  = os.path.join(out_dir, out_name)
    
    with open(out_path, 'w') as f:
        header = ',   '.join(df2.columns)
        f.write(header + '\n')
        df2.to_csv(f, index=False, header=False, float_format="%.12e")
    print(f"[✔] Processed {fname} → {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="Compute raw & placeholder‐normalized metrics from tgv_data.csv"
    )
    p.add_argument(
        "--mach", type=float, required=True,
        help="Turbulent Mach number M"
    )
    p.add_argument(
        "--domain-length", type=float, required=True,
        help="Domain length "
    )
    p.add_argument(
        "files", nargs="+",
        help="One or more input CSV files"
    )
    args = p.parse_args()

    for f in args.files:
        if not os.path.isfile(f):
            print(f"[!] Warning: '{f}' not found, skipping", file=sys.stderr)
            continue
        process_file(f, args.mach, args.domain_length)

if __name__ == "__main__":
    main()
