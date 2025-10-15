#!/usr/bin/env python3
"""
run_analysis.py
Compute raw dk/dt, effective viscosity and Re, plus insert
placeholders for non?dimensional (%_star) quantities.

Usage:
    python run_analysis.py --U0 1.0 --domain-length 6.28318 path/to/tgv_data.csv [...]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

def process_file(fname: str, U0: float, domain_length: float) -> None:
    L = domain_length/(2*np.pi)
    
    # --- 1) Read and clean up
    df = pd.read_csv(fname, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    # FIX: Handle column names with spaces by replacing with underscores
    df.columns = df.columns.str.replace(' ', '_')
    
    # Verify we have the required columns
    required_cols = ['time', 'kinetic_energy', 'enstrophy']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in {fname}: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        return
    
    df['time'] = df['time'].astype(float)
    
    # --- 2) Raw derivatives
    dt = df['time'].diff()                    # ?t, first row = NaN
    df['dk_dt'] = df['kinetic_energy'].diff() / dt
    # I believe this is correct
    df['effective_nu_half'] = -df['dk_dt'] / (2.0 * df['enstrophy'])

    # I believe this is not correct but standard in definitions
    df['effective_nu'] = -df['dk_dt'] / (df['enstrophy'])
    
    # back?fill first?row NaNs so nothing is missing
    df[['dk_dt','effective_nu']] = df[['dk_dt','effective_nu']].bfill()
    df[['dk_dt','effective_nu_half']] = df[['dk_dt','effective_nu_half']].bfill()
    
    # Reynolds number
    # I believe this is correct
    df['Re_half'] = (U0 * L) / df['effective_nu_half']

    # I believe this is not correct but standard in definitions
    df['Re'] = (U0 * L) / df['effective_nu']

    # ReL = 2 * enstrophy * (kinetic_energy**2) / (-(dk_dt**2))
    # Add a tiny epsilon to avoid division by zero
    eps = 1e-10
    denom = df['dk_dt']**2 - eps
    df['ReL'] = 2.0 * df['enstrophy'] * (df['kinetic_energy']**2) / denom
    
    # Example non?dimensional quantities:
    # t* = t * U0 / L
    df['t_star'] = df['time'] * U0 / L

    # (dK/dt)* = (dK/dt) / (U0^3 / L)
    df['dk_dt_star'] = df['dk_dt'] / (U0**3 / L)

    # ?* = 2.0*? / (U0/L)^2
    df['enstrophy_star'] = df['enstrophy'] * 2.0 / (U0/L)**2

    df['kinetic_energy_star'] = df['kinetic_energy']/ (U0)**2

    # --- 4) Write out
    out_dir   = os.path.dirname(os.path.abspath(fname))
    base, ext = os.path.splitext(os.path.basename(fname))
    out_name  = f"{base}_processed{ext}"
    out_path  = os.path.join(out_dir, out_name)
    
    # Write CSV with spaced headers
    with open(out_path, 'w') as f:
        # Write header with spaces after commas
        header = ',   '.join(df.columns)
        f.write(header + '\n')
        
        # Write the data (without header since we already wrote it)
        df.to_csv(f, index=False, header=False, float_format="%.12e")
    
    print(f"[?] Processed {fname} ? {out_path}")
    print(f"    Output columns: {list(df.columns)}", file=sys.stderr)

def main():
    p = argparse.ArgumentParser(
        description="Compute raw & placeholder?normalized metrics from tgv_data.csv"
    )
    p.add_argument(
        "--U0", type=float, required=True,
        help="Initial velocity U0"
    )
    p.add_argument(
        "--domain-length", type=float, required=True,
        help="Domain length"
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

