#!/usr/bin/env python3
"""
Parallel reading of a velocity field with 6-line header, computing TKE
and verifying correctness versus a serial read.

To run (example):
  mpiexec -n 4 python DebugParallelTKE.py
"""

import os
import re
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------- SETUP -------------------
file_directory = '/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_64/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P2/'
filename = os.path.join(file_directory, 'cycle_9000', 'element_centers_9000.txt')

# Number of header lines to skip (known from your data format)
NUM_HEADER_LINES = 6

if rank == 0:
    print("="*70)
    print("[Rank 0] Parallel TKE Computation with 6 Header Lines to Skip")
    print(f"[Rank 0] File: {filename}")
    print(f"[Rank 0] MPI size: {size}")
    print("="*70)

comm.Barrier()

# ------------------- STEP 0: Count data lines (Rank 0 only) -------------------
# We'll skip the 6 header lines, then count how many data lines remain.
total_data_lines = None
if rank == 0:
    # Open file
    with open(filename, 'r') as f:
        # Skip the first NUM_HEADER_LINES lines
        for _ in range(NUM_HEADER_LINES):
            f.readline()
        # Count the rest
        line_count = 0
        for _ in f:
            line_count += 1
    total_data_lines = line_count
    print(f"[Rank 0] Detected {total_data_lines} data lines (excluding {NUM_HEADER_LINES} header lines).")

# Broadcast total_data_lines to all ranks
total_data_lines = comm.bcast(total_data_lines, root=0)

# ------------------- STEP 1: Partition data lines among ranks -------------------
# We'll do an even split, plus leftover for the first 'remainder' ranks
lines_per_rank = total_data_lines // size
remainder = total_data_lines % size

start_idx = lines_per_rank * rank + min(rank, remainder)
end_idx   = start_idx + lines_per_rank + (1 if rank < remainder else 0)

# Debug printing
print(f"[Rank {rank}] Will read data lines [{start_idx}, {end_idx}) out of {total_data_lines}")

comm.Barrier()

# ------------------- STEP 2: Parallel reading & partial TKE -------------------
local_count = 0
local_tke = 0.0

with open(filename, 'r') as f:
    # 2A) Skip the NUM_HEADER_LINES first
    for _ in range(NUM_HEADER_LINES):
        f.readline()

    # 2B) Skip 'start_idx' more lines
    for _ in range(start_idx):
        line = f.readline()
        if not line:
            break  # in case of unexpected EOF

    # 2C) Read lines from [start_idx..end_idx)
    for _ in range(start_idx, end_idx):
        line = f.readline()
        if not line:
            break
        parts = line.strip().split()
        # We expect 6 columns: x, y, z, u, v, w
        if len(parts) < 6:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            u = float(parts[3])
            v = float(parts[4])
            w = float(parts[5])
        except ValueError:
            # If there's any malformed line, skip or handle as needed
            continue

        # Kinetic energy at this point = 0.5*(u^2+v^2+w^2)
        local_tke += 0.5 * (u*u + v*v + w*w)
        local_count += 1

# ------------------- STEP 3: Combine partial TKE & counts -------------------
global_tke = comm.reduce(local_tke, op=MPI.SUM, root=0)
global_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# ------------------- STEP 4: Verify on Rank 0 with a Serial Pass -------------------
if rank == 0:
    print(f"[Rank 0] Summed partial TKE across ranks. Parallel assigned lines = {global_count}.")
    # Now do a quick serial pass (skip 6 lines, parse all, compute TKE) to compare
    serial_count = 0
    serial_tke = 0.0
    with open(filename, 'r') as f:
        # skip 6 header lines
        for _ in range(NUM_HEADER_LINES):
            f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                u = float(parts[3])
                v = float(parts[4])
                w = float(parts[5])
            except ValueError:
                continue
            serial_tke += 0.5 * (u*u + v*v + w*w)
            serial_count += 1

    # Print results
    print(f"[Rank 0] total_data_lines = {total_data_lines}, global_count = {global_count}, serial_count = {serial_count}")
    print(f"[Rank 0] TKE (parallel) = {global_tke:.6f}")
    print(f"[Rank 0] TKE (serial)   = {serial_tke:.6f}")

    if global_count != serial_count:
        print("[Rank 0] WARNING: parallel line count != serial line count")

    tolerance = 1e-12
    diff = abs(global_tke - serial_tke)
    if diff < tolerance:
        print("[Rank 0] SUCCESS: Parallel TKE matches serial TKE within tolerance.")
    else:
        print(f"[Rank 0] MISMATCH: TKE difference = {diff:.6e} > tol={tolerance:.1e}")

MPI.Finalize()

