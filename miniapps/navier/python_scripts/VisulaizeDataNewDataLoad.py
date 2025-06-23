#!/usr/bin/env python
"""
Read an MFEM velocity dump (block-wise [x y z u v w]), reconstruct grid,
visualise velocity magnitude, and compute TKE – now compatible with
the new Vector::Print() file format.

Run with
   srun -n 4 python VisualiseField.py /path/to/SamplePointsAtDoFs_step0.txt
"""
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import re, os, subprocess, argparse

# ------------------------------------------------------------------ #
#  MPI + helper utilities  (unchanged)
# ------------------------------------------------------------------ #
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

def log_memory_rank0(msg):
    if rank == 0:
        pid = os.getpid()
        out = subprocess.check_output(f"ps -p {pid} -o rss",
                                      shell=True).decode().splitlines()[1]
        print(f"[Rank 0] {msg} – RSS: {int(out)/1024:.2f} MB")

def log_memory_global(msg):
    pid = os.getpid()
    try:
        out = subprocess.check_output(f"ps -p {pid} -o rss",
                                      shell=True).decode().splitlines()[1]
        rss = int(out)/1024
    except Exception:
        rss = 0.0
    tot = comm.reduce(rss, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"{msg} – Total RSS: {tot:.2f} MB")

def get_local_slice(n, r, p):
    counts = [n//p + (1 if rr < n % p else 0) for rr in range(p)]
    start  = sum(counts[:r])
    return start, start + counts[r]

# ------------------------------------------------------------------ #
#  Parse CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(
    description="Visualise MFEM velocity magnitude field.")
parser.add_argument("data_file", help="Path to SamplePointsAtDoFs*.txt")
args  = parser.parse_args()
fname = args.data_file

# ------------------------------------------------------------------ #
#  Rank-0: read header and load flat vector
# ------------------------------------------------------------------ #
nx = ny = nz = None
step_number_extracted = None
time_extracted        = None
tke_physical          = None

if rank == 0:
    print(f"[Rank 0] Reading file: {fname}")

    # ---- 1. header ----
    header_lines = []
    with open(fname, "r") as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                break
            header_lines.append(line)
    for ln in header_lines:
        m = re.search(r"Step\s+(\d+)", ln)
        if m: step_number_extracted = m.group(1)
        m = re.search(r"Time\s+([0-9.+-Ee]+)", ln)
        if m: time_extracted = float(m.group(1))
    if step_number_extracted is None:
        step_number_extracted = "Unknown"
    if time_extracted is None:
        time_extracted = 0.0
    print(f"[Rank 0] Header: Step {step_number_extracted}, "
          f"Time {time_extracted:.3e}")

    # ---- 2. numerical payload ----
    # np.loadtxt skips lines that start with '#'
    flat = np.loadtxt(fname, comments="#", dtype=np.float64).ravel()
    if flat.size % 6 != 0:
        raise ValueError("Data length not divisible by 6 – corrupted file?")
    ND = flat.size // 6

    # Slice blocks: [x | y | z | u | v | w]
    xpos = flat[0*ND : 1*ND]
    ypos = flat[1*ND : 2*ND]
    zpos = flat[2*ND : 3*ND]
    velx = flat[3*ND : 4*ND]
    vely = flat[4*ND : 5*ND]
    velz = flat[5*ND : 6*ND]
    log_memory_rank0("After reading & slicing flat vector")

    # ---- 3. round coords, build unique grids ----
    xpos_r = np.round(xpos, 10)
    ypos_r = np.round(ypos, 10)
    zpos_r = np.round(zpos, 10)

    x_unique = np.unique(xpos_r)
    y_unique = np.unique(ypos_r)
    z_unique = np.unique(zpos_r)
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

    print(f"[Rank 0] Grid size = {nx} × {ny} × {nz}")
    if nx * ny * nz != ND:
        print("[Rank 0] Warning: node count mismatch!")

    # ---- 4. populate 3-D velocity arrays ----
    velx_grid = np.full((nx, ny, nz), np.nan)
    vely_grid = np.full((nx, ny, nz), np.nan)
    velz_grid = np.full((nx, ny, nz), np.nan)
    xi = {v:i for i,v in enumerate(x_unique)}
    yi = {v:i for i,v in enumerate(y_unique)}
    zi = {v:i for i,v in enumerate(z_unique)}
    for k in range(ND):
        velx_grid[xi[xpos_r[k]], yi[ypos_r[k]], zi[zpos_r[k]]] = velx[k]
        vely_grid[xi[xpos_r[k]], yi[ypos_r[k]], zi[zpos_r[k]]] = vely[k]
        velz_grid[xi[xpos_r[k]], yi[ypos_r[k]], zi[zpos_r[k]]] = velz[k]
    velx_grid = np.nan_to_num(velx_grid)
    vely_grid = np.nan_to_num(vely_grid)
    velz_grid = np.nan_to_num(velz_grid)
    log_memory_rank0("After reconstructing 3-D grids")

    # ---- 5. physical-space KE and |u| ----
    tke_physical = 0.5*np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    print(f"[Rank 0] ⟨KE⟩ = {tke_physical:.6f}")
    velocity_magnitude = np.sqrt(velx_grid**2 + vely_grid**2 + velz_grid**2)

    # ---- 6. distribute slabs along x ----
    for r in range(1, size):
        s,e = get_local_slice(nx, r, size)
        comm.Send(velx_grid[s:e], dest=r, tag=100)
        comm.Send(vely_grid[s:e], dest=r, tag=101)
        comm.Send(velz_grid[s:e], dest=r, tag=102)
else:
    # non-root: will receive slices later
    pass

# ------------------------------------------------------------------ #
#  Broadcast scalar meta-data
# ------------------------------------------------------------------ #
nx = comm.bcast(nx, root=0)
ny = comm.bcast(ny, root=0)
nz = comm.bcast(nz, root=0)
step_number_extracted = comm.bcast(step_number_extracted, root=0)
time_extracted        = comm.bcast(time_extracted, root=0)
tke_physical          = comm.bcast(tke_physical, root=0)

# ------------------------------------------------------------------ #
#  Non-root ranks receive their slabs
# ------------------------------------------------------------------ #
local_start, local_stop = get_local_slice(nx, rank, size)
local_shape = (local_stop - local_start, ny, nz)
local_velx = np.empty(local_shape)
local_vely = np.empty(local_shape)
local_velz = np.empty(local_shape)

if rank == 0:
    local_velx[:] = velx_grid[local_start:local_stop]
    local_vely[:] = vely_grid[local_start:local_stop]
    local_velz[:] = velz_grid[local_start:local_stop]
else:
    comm.Recv(local_velx, source=0, tag=100)
    comm.Recv(local_vely, source=0, tag=101)
    comm.Recv(local_velz, source=0, tag=102)

local_velocity_magnitude = np.sqrt(local_velx**2 + local_vely**2 + local_velz**2)
comm.Barrier()
log_memory_global("After distributing local slabs")

# ------------------------------------------------------------------ #
#  Plot (rank 0) – identical to your original
# ------------------------------------------------------------------ #
if rank == 0:
    print("[Rank 0] Plotting velocity magnitude …")
    X, Y, Z = np.meshgrid(x_unique, y_unique, z_unique, indexing='ij')
    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(),
                     c=velocity_magnitude.ravel(),
                     cmap='viridis', marker='.')
    plt.colorbar(sc, ax=ax, label='|u|')
    ax.set(xlabel='X', ylabel='Y', zlabel='Z',
           title=(f'Velocity magnitude\n'
                  f'⟨KE⟩={tke_physical:.6f}  '
                  f'Step {step_number_extracted}, '
                  f'Time {time_extracted:.3e}'))
    plt.tight_layout()
    plt.show()

comm.Barrier()
log_memory_global("Before MPI finalize")
MPI.Finalize()

