#!/usr/bin/env python3
"""
run_analysis.py

Computes effective viscosity and effective Reynolds number in two ways:

1) Baseline (from tgv_data):
   nu_eff = - (dk/dt) / (2 * enstrophy)

2) Optional visc-only numerator (from turb_diag):
   nu_eff_visc = (avg_de_dt_visc_rk2) / (2 * enstrophy_from_tgv_data)

3) Always, if turb_diag is provided:
   Re_{lambda_g,visc} and related quantities, using viscous dE/dt.

INF/NAN handling:
- effective_nu_visc is undefined when avg_de_dt_visc_rk2 is ~0, this yields inf in Re_eff_visc.
- We provide three policies to handle this, controlled by CLI flags:

  A) Default: "nan" policy
     If |avg_de_dt_visc_rk2| < --min-abs-dedt, set avg_de_dt_visc_rk2 to NaN,
     producing NaN effective_nu_visc and Re_eff_visc (no inf).

  B) "floor_nu" policy
     Compute effective_nu_visc, then clamp to >= --nu-floor before inverting.

  C) "cap_re" policy
     Compute Re_eff_visc, then clamp to <= --re-cap.

You can choose the policy with --visc-re-policy.

Outputs:
- {tgv_base}_processed.csv
- If --turb-diag is provided:
  {turb_diag_base}_processed_visc_effective_Re.csv
- If internal_energy exists in tgv_data, also writes:
  {tgv_base}_processed_energy_conservation.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Literal

import numpy as np
import pandas as pd


def read_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def write_csv_with_spaced_header(df: pd.DataFrame, out_path: str) -> None:
    """Keeps the existing style used previously, a header line with extra spaces."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",   ".join(df.columns) + "\n")
        df.to_csv(f, index=False, header=False, float_format="%.12e")


def to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def is_tgv_data(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"time", "kinetic_energy", "enstrophy"}.issubset(cols)


def is_turb_diag(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"time", "avg_de_dt_visc"}.issubset(cols)


def build_enstrophy_lookup(df_tgv: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup DataFrame from TGV data, containing time, cycle (if present),
    enstrophy, and kinetic_energy, to be merged into turb_diag.
    """
    keep = [c for c in ["cycle", "time", "enstrophy", "kinetic_energy"] if c in df_tgv.columns]
    out = df_tgv[keep].copy()
    to_numeric_inplace(out, ["cycle", "time", "enstrophy", "kinetic_energy"])
    out = out.dropna(subset=["enstrophy"])

    if "cycle" in out.columns:
        out["cycle"] = pd.to_numeric(out["cycle"], errors="coerce").astype("Int64")

    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out = out.dropna(subset=["time"])
    return out


def merge_enstrophy(diag: pd.DataFrame, ens: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer merge on cycle if present in both dataframes, otherwise merge by nearest time.
    Merges both enstrophy and kinetic_energy if available in ens.
    """
    diag = diag.copy()
    ens = ens.copy()

    if "cycle" in diag.columns and "cycle" in ens.columns:
        to_numeric_inplace(diag, ["cycle"])
        diag["cycle"] = pd.to_numeric(diag["cycle"], errors="coerce").astype("Int64")

        merge_cols = [c for c in ["cycle", "enstrophy", "kinetic_energy"] if c in ens.columns]
        merged = diag.merge(ens[merge_cols], on="cycle", how="left")
        return merged

    # Nearest-time merge
    diag = diag.sort_values("time")
    ens = ens.sort_values("time")

    diag_time = diag["time"].to_numpy(dtype=float)
    ens_time = ens["time"].to_numpy(dtype=float)

    tol: Optional[float] = None
    if len(ens_time) >= 2:
        dt = np.diff(ens_time)
        dt_med = float(np.nanmedian(dt)) if np.isfinite(np.nanmedian(dt)) else np.nan
        if np.isfinite(dt_med) and dt_med > 0:
            tol = 0.51 * dt_med

    merge_cols = [c for c in ["time", "enstrophy", "kinetic_energy"] if c in ens.columns]

    if tol is None:
        merged = diag.merge(ens[merge_cols], on="time", how="left")
    else:
        merged = pd.merge_asof(
            diag,
            ens[merge_cols],
            on="time",
            direction="nearest",
            tolerance=tol,
        )
    return merged


def process_tgv_data(tgv_path: str, df: pd.DataFrame, U0: float, Lref: float) -> pd.DataFrame:
    df = df.copy()
    to_numeric_inplace(df, ["time", "cycle", "kinetic_energy", "enstrophy", "internal_energy"])
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].astype(float)

    # Baseline dk/dt
    dt = df["time"].diff().replace(0.0, np.nan)
    df["dk_dt"] = df["kinetic_energy"].diff() / dt

    # Effective viscosity (baseline)
    eps = 1e-30
    denom = (2.0 * df["enstrophy"]).replace(0.0, np.nan)
    df["effective_nu"] = -df["dk_dt"] / (denom + eps)

    # Fill initial NaNs from derivative and resulting nu
    df[["dk_dt", "effective_nu"]] = df[["dk_dt", "effective_nu"]].bfill()

    # Effective Reynolds number
    df["Re_effective"] = (U0 * Lref) / df["effective_nu"]

    # Optional nondimensional extras
    df["t_star"] = df["time"] * U0 / Lref
    df["dk_dt_star"] = df["dk_dt"] / (U0**3 / Lref)
    df["enstrophy_star"] = df["enstrophy"] * 2.0 / (U0 / Lref) ** 2
    df["kinetic_energy_star"] = df["kinetic_energy"] / (U0**2)

    out_dir = os.path.dirname(os.path.abspath(tgv_path))
    base, ext = os.path.splitext(os.path.basename(tgv_path))
    out_path = os.path.join(out_dir, f"{base}_processed{ext}")
    write_csv_with_spaced_header(df, out_path)
    print(f"[OK] Wrote {out_path}")

    # Energy conservation output if present
    if "internal_energy" in df.columns:
        cols_need = [
            c for c in ["time", "t_star", "cycle", "kinetic_energy", "internal_energy"] if c in df.columns
        ]
        if {"time", "kinetic_energy", "internal_energy"}.issubset(set(cols_need)):
            df2 = df[cols_need].copy()
            df2["IE + KE"] = df2["internal_energy"] + df2["kinetic_energy"]
            initial = df2["IE + KE"].iloc[0]
            if pd.isna(initial) or initial == 0:
                df2["(IE + KE)_normalized"] = np.nan
            else:
                df2["(IE + KE)_normalized"] = df2["IE + KE"] / initial

            out_path2 = os.path.join(out_dir, f"{base}_processed_energy_conservation{ext}")
            write_csv_with_spaced_header(df2, out_path2)
            print(f"[OK] Wrote {out_path2}")

    return df


ViscPolicy = Literal["nan", "floor_nu", "cap_re"]


def apply_visc_re_bounding(
    merged: pd.DataFrame,
    *,
    U0: float,
    Lref: float,
    min_abs_dedt: float,
    visc_re_policy: ViscPolicy,
    nu_floor: float,
    re_cap: float,
) -> pd.DataFrame:
    """
    Compute effective_nu_visc and Re_eff_visc with configurable inf/large-value handling.

    - "nan": if |dE/dt| < min_abs_dedt -> set dE/dt to NaN, outputs become NaN (no inf)
    - "floor_nu": clamp effective_nu_visc >= nu_floor before inversion
    - "cap_re": compute Re_eff_visc then clamp to <= re_cap
    """
    out = merged.copy()

    eps = 1e-10
    denom = (2.0 * out["enstrophy"]).replace(0.0, np.nan)

    # Ensure column exists, else all NaN
    if "avg_de_dt_visc_rk2" not in out.columns:
        out["effective_nu_visc"] = np.nan
        out["Re_eff_visc"] = np.nan
        return out

    d = out["avg_de_dt_visc_rk2"].astype(float)

    if visc_re_policy == "nan":
        bad = d.abs() < float(min_abs_dedt)
        d = d.mask(bad, np.nan)
        out["avg_de_dt_visc_rk2"] = d
        out["effective_nu_visc"] = d / (denom + eps)
        out["Re_eff_visc"] = (U0 * Lref) / out["effective_nu_visc"]

    elif visc_re_policy == "floor_nu":
        out["effective_nu_visc"] = d / (denom + eps)
        out["effective_nu_visc"] = out["effective_nu_visc"].clip(lower=float(nu_floor))
        out["Re_eff_visc"] = (U0 * Lref) / out["effective_nu_visc"]

    elif visc_re_policy == "cap_re":
        out["effective_nu_visc"] = d / (denom + eps)
        out["Re_eff_visc"] = (U0 * Lref) / out["effective_nu_visc"]
        out["Re_eff_visc"] = out["Re_eff_visc"].clip(upper=float(re_cap))

    else:
        raise ValueError(f"Unknown visc_re_policy: {visc_re_policy}")

    # Clean up any inf that may remain from other pathologies
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def process_turb_diag_visc(
    turb_path: str,
    df_diag: pd.DataFrame,
    enstrophy_lookup: pd.DataFrame,
    U0: float,
    Lref: float,
    *,
    visc_re_policy: ViscPolicy,
    min_abs_dedt: float,
    nu_floor: float,
    re_cap: float,
) -> None:
    df_diag = df_diag.copy()
    to_numeric_inplace(
        df_diag,
        ["time", "cycle", "avg_de_dt_visc", "avg_de_dt_visc_rk2", "taylor_micro_scale"],
    )
    df_diag = df_diag.dropna(subset=["time"])
    df_diag["time"] = df_diag["time"].astype(float)

    merged = merge_enstrophy(df_diag, enstrophy_lookup)

    # Compute effective_nu_visc + Re_eff_visc with bounding policy
    merged = apply_visc_re_bounding(
        merged,
        U0=U0,
        Lref=Lref,
        min_abs_dedt=min_abs_dedt,
        visc_re_policy=visc_re_policy,
        nu_floor=nu_floor,
        re_cap=re_cap,
    )

    merged["t_star"] = merged["time"] * U0 / Lref

    # Always attempt to compute Re_lambda_* when turb_diag is present
    missing_cols = [
        c
        for c in ["taylor_micro_scale", "enstrophy", "kinetic_energy", "avg_de_dt_visc_rk2"]
        if c not in merged.columns
    ]
    if missing_cols:
        print(
            "[WARN] Cannot compute Re_lambda_g_visc, missing columns: " + ", ".join(missing_cols),
            file=sys.stderr,
        )
        merged["Re_lambda_g_visc"] = np.nan
    else:
        lambda_f = merged["taylor_micro_scale"].astype(float).copy()
        lambda_g = (lambda_f / np.sqrt(2.0)).copy()
        Omega = merged["enstrophy"]
        Ek = merged["kinetic_energy"]
        DEkDt_visc = merged["avg_de_dt_visc_rk2"]

        merged["lambda_g"] = lambda_g
        merged["lambda_f"] = lambda_f

        eps = 1e-10
        # Guard against zero DEkDt_visc for the Re_lambda expressions
        denom_Re = DEkDt_visc.replace(0.0, 1e-5)

        Re_lambda_f = 2.0 * lambda_f * Omega * np.sqrt((2.0 / 3.0) * Ek)
        Re_lambda_f /= (denom_Re + eps)
        merged["Re_lambda_f_visc"] = Re_lambda_f

        Re_lambda_g = 2.0 * lambda_g * Omega * np.sqrt((2.0 / 3.0) * Ek)
        Re_lambda_g /= (denom_Re + eps)
        merged["Re_lambda_g_visc"] = Re_lambda_g

        Re_lambda_ke = np.sqrt(40.0 * Omega / 3.0) * Ek
        Re_lambda_ke /= (denom_Re + eps)
        merged["Re_lambda_ke_visc"] = Re_lambda_ke

        merged["taylor_micro_scale_ke"] = np.sqrt(5.0 * Ek / Omega)

        # Integral length scale
        merged["L_int"] = (Ek ** 1.5) / (DEkDt_visc + eps)

        denom = np.maximum(DEkDt_visc**2, eps)
        merged["Re_L_int_visc"] = 2.0 * Omega * (Ek**2) / denom

        # Integral Length scale
        merged["kolm_lng_scl"] = ((merged["effective_nu_visc"] ** 3) 
                / merged["avg_de_dt_visc_rk2"].abs())**0.25

    keep_base = [
        "time",
        "cycle",
        "t_star",
        "enstrophy",
        "avg_de_dt_visc",
        "effective_nu_visc",
        "Re_eff_visc",
    ]
    keep_extra = []
    for c in [
        "taylor_micro_scale",
        "lambda_f",
        "lambda_g",
        "taylor_micro_scale_ke",
        "avg_de_dt_visc_rk2",
        "kinetic_energy",
        "Re_lambda_f_visc",
        "Re_lambda_g_visc",
        "Re_lambda_ke_visc",
        "L_int",
        "Re_L_int_visc",
        "kolm_lng_scl",
    ]:
        if c in merged.columns:
            keep_extra.append(c)

    keep = [c for c in keep_base + keep_extra if c in merged.columns]
    out = merged[keep].copy()

    n_miss = int(out["enstrophy"].isna().sum()) if "enstrophy" in out.columns else 0
    if n_miss:
        print(
            f"[WARN] {n_miss} turb_diag rows could not be matched to enstrophy, results are NaN for those rows.",
            file=sys.stderr,
        )

    out_dir = os.path.dirname(os.path.abspath(turb_path))
    base, ext = os.path.splitext(os.path.basename(turb_path))
    out_path = os.path.join(out_dir, f"{base}_processed_visc_effective_Re{ext}")
    write_csv_with_spaced_header(out, out_path)
    print(f"[OK] Wrote {out_path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute effective viscosity and effective Reynolds number from TGV CSV outputs."
    )
    p.add_argument("--U0", type=float, required=True, help="Reference velocity scale U0")
    p.add_argument(
        "--domain-length",
        type=float,
        required=True,
        help="Domain length (same units as used in the simulation)",
    )
    p.add_argument(
        "tgv_file",
        help="Primary tgv_data CSV, must include time, kinetic_energy, enstrophy",
    )
    p.add_argument(
        "--turb-diag",
        default=None,
        help="Optional turb_diag CSV, must contain time and avg_de_dt_visc. "
        "Uses enstrophy and kinetic_energy from the primary tgv_file. "
        "If present, also computes Re_lambda_g_visc using taylor_micro_scale and avg_de_dt_visc_rk2.",
    )

    # New: bounding controls for visc-only Re
    p.add_argument(
        "--visc-re-policy",
        choices=["nan", "floor_nu", "cap_re"],
        default="nan",
        help="How to handle near-zero viscous dE/dt that causes inf/huge Re_eff_visc. "
        "'nan' sets small |dE/dt| to NaN, 'floor_nu' clamps effective_nu_visc, "
        "'cap_re' clamps Re_eff_visc.",
    )
    p.add_argument(
        "--min-abs-dedt",
        type=float,
        default=1e-12,
        help="Threshold for |avg_de_dt_visc_rk2| below which values are treated as invalid (nan policy).",
    )
    p.add_argument(
        "--nu-floor",
        type=float,
        default=1e-8,
        help="Minimum effective_nu_visc when using --visc-re-policy floor_nu.",
    )
    p.add_argument(
        "--re-cap",
        type=float,
        default=1e6,
        help="Maximum Re_eff_visc when using --visc-re-policy cap_re.",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.U0 <= 0:
        raise ValueError("--U0 must be > 0")
    if args.domain_length <= 0:
        raise ValueError("--domain-length must be > 0")

    if not os.path.isfile(args.tgv_file):
        raise FileNotFoundError(f"Primary file not found: {args.tgv_file}")

    if args.turb_diag is not None and (not os.path.isfile(args.turb_diag)):
        raise FileNotFoundError(f"turb_diag file not found: {args.turb_diag}")

    Lref = args.domain_length / (2.0 * np.pi)

    # Read and validate primary tgv_data file
    tgv = read_clean_csv(args.tgv_file)
    if not is_tgv_data(tgv):
        raise ValueError(
            f"Primary file '{args.tgv_file}' must contain columns: time, kinetic_energy, enstrophy"
        )

    # If turb_diag requested, enforce enstrophy exists in primary file
    if args.turb_diag is not None and "enstrophy" not in tgv.columns:
        raise ValueError(
            f"--turb-diag was provided, but primary file '{args.tgv_file}' has no 'enstrophy' column"
        )

    # Baseline processing
    process_tgv_data(args.tgv_file, tgv, args.U0, Lref)

    # Optional visc-only numerator processing
    if args.turb_diag is not None:
        diag = read_clean_csv(args.turb_diag)
        if not is_turb_diag(diag):
            raise ValueError(
                f"turb_diag file '{args.turb_diag}' must contain columns: time, avg_de_dt_visc"
            )

        enst_lookup = build_enstrophy_lookup(tgv)
        if len(enst_lookup) == 0:
            raise ValueError(f"No valid enstrophy rows found in primary file '{args.tgv_file}'")

        process_turb_diag_visc(
            args.turb_diag,
            diag,
            enst_lookup,
            args.U0,
            Lref,
            visc_re_policy=args.visc_re_policy,
            min_abs_dedt=args.min_abs_dedt,
            nu_floor=args.nu_floor,
            re_cap=args.re_cap,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
