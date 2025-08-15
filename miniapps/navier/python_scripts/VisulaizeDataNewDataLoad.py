#!/usr/bin/env python
"""
Read an MFEM velocity dump (row-wise [x y z u v w]), reconstruct grid,
visualise velocity magnitude, and compute TKE – now compatible with
the new row-wise file format.
Run with
   srun -n 4 python VisualiseField.py /path/to/SamplePointsAtDoFs_step0.txt
"""
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import re, os, subprocess, argparse

# ------------------------------------------------------------------ #
#  MPI + helper utilities
# ------------------------------------------------------------------ #
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

def log_memory_rank0(msg):
    if rank == 0:
        pid = os.getpid()
        try:
            out = subprocess.check_output(f"ps -p {pid} -o rss",
                                          shell=True).decode().splitlines()[1]
            print(f"[Rank 0] {msg} – RSS: {int(out)/1024:.2f} MB")
        except Exception:
            print(f"[Rank 0] {msg} – RSS: unavailable")

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
    """Compute start:end slice for rank r out of p ranks for array of size n"""
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
#  Rank-0: read header and load data
# ------------------------------------------------------------------ #
nx = ny = nz = None
step_number_extracted = None
time_extracted        = None
tke_physical          = None
x_unique = y_unique = z_unique = None
velocity_magnitude = None

if rank == 0:
    print(f"[Rank 0] Reading file: {fname}")
    
    # ---- 1. Parse header ----
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
    
    # ---- 2. Load data in new row-wise format ----
    # Each row is [x y z u v w]
    print("[Rank 0] Loading data...")
    data = np.loadtxt(fname, comments="#", dtype=np.float64)
    
    if data.ndim == 1:
        # Single row case
        data = data.reshape(1, -1)
    
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Expected data with 6 columns [x y z u v w], got shape {data.shape}")
    
    ND = data.shape[0]
    print(f"[Rank 0] Loaded {ND} data points")

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
    
    tke_before_process = 0.5 * np.mean(velx**2 + vely**2 + velz**2)
    print(f"[Rank 0] Total Kinetic Energy in Physical Space (TKE before process): {tke_before_process:.6f}")

    log_memory_rank0("After reading & extracting columns")
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    
    print(f"[Rank 0] Grid size = {nx} × {ny} × {nz} = {nx*ny*nz}")
    print(f"[Rank 0] Data points = {ND}")
    
    if nx * ny * nz != ND:
        print(f"[Rank 0] Warning: node count mismatch! Grid={nx*ny*nz}, Data={ND}")
        print("[Rank 0] This might indicate irregular grid or missing data")
    
    # ---- 4. Populate 3-D velocity arrays ----
    print("[Rank 0] Reconstructing 3D velocity grids...")
    velx_grid = np.full((nx, ny, nz), np.nan)
    vely_grid = np.full((nx, ny, nz), np.nan)
    velz_grid = np.full((nx, ny, nz), np.nan)
    
    # Create index mappings for faster lookup
    xi = {v:i for i,v in enumerate(x_unique)}
    yi = {v:i for i,v in enumerate(y_unique)}
    zi = {v:i for i,v in enumerate(z_unique)}
    
    # Populate grids
    missing_points = 0
    for k in range(ND):
        try:
            i_idx = xi[xpos_r[k]]
            j_idx = yi[ypos_r[k]] 
            k_idx = zi[zpos_r[k]]
            velx_grid[i_idx, j_idx, k_idx] = velx[k]
            vely_grid[i_idx, j_idx, k_idx] = vely[k]
            velz_grid[i_idx, j_idx, k_idx] = velz[k]
        except KeyError:
            missing_points += 1
    
    if missing_points > 0:
        print(f"[Rank 0] Warning: {missing_points} points could not be mapped to grid")
    
    # Check for missing data
    nan_count = np.sum(np.isnan(velx_grid))
    if nan_count > 0:
        print(f"[Rank 0] Warning: {nan_count} grid points have NaN values (will be set to 0)")
    
    # Replace NaN with zeros
    velx_grid = np.nan_to_num(velx_grid)
    vely_grid = np.nan_to_num(vely_grid)
    velz_grid = np.nan_to_num(velz_grid)
    
    log_memory_rank0("After reconstructing 3-D grids")
    
    # ---- 5. Compute physical-space KE and velocity magnitude ----
    print("[Rank 0] Computing kinetic energy and velocity magnitude...")
    tke_physical = 0.5*np.mean(velx_grid**2 + vely_grid**2 + velz_grid**2)
    print(f"[Rank 0] ⟨KE⟩ = {tke_physical:.6f}")
    
    velocity_magnitude = np.sqrt(velx_grid**2 + vely_grid**2 + velz_grid**2)
    print(f"[Rank 0] |u| range: [{np.min(velocity_magnitude):.4f}, {np.max(velocity_magnitude):.4f}]")
    
    # ---- 6. Distribute slabs along x-direction to other ranks ----
    if size > 1:
        print(f"[Rank 0] Distributing data to {size-1} other ranks...")
        for r in range(1, size):
            s, e = get_local_slice(nx, r, size)
            comm.Send(velx_grid[s:e], dest=r, tag=100)
            comm.Send(vely_grid[s:e], dest=r, tag=101)
            comm.Send(velz_grid[s:e], dest=r, tag=102)
        print("[Rank 0] Data distribution complete")

# ------------------------------------------------------------------ #
#  Broadcast scalar meta-data to all ranks
# ------------------------------------------------------------------ #
nx = comm.bcast(nx, root=0)
ny = comm.bcast(ny, root=0)
nz = comm.bcast(nz, root=0)
step_number_extracted = comm.bcast(step_number_extracted, root=0)
time_extracted        = comm.bcast(time_extracted, root=0)
tke_physical          = comm.bcast(tke_physical, root=0)
x_unique = comm.bcast(x_unique, root=0)
y_unique = comm.bcast(y_unique, root=0)
z_unique = comm.bcast(z_unique, root=0)

# ------------------------------------------------------------------ #
#  All ranks receive their local data slabs
# ------------------------------------------------------------------ #
local_start, local_stop = get_local_slice(nx, rank, size)
local_shape = (local_stop - local_start, ny, nz)

if rank != 0:
    print(f"[Rank {rank}] Receiving data slice {local_start}:{local_stop}")

local_velx = np.empty(local_shape, dtype=np.float64)
local_vely = np.empty(local_shape, dtype=np.float64)
local_velz = np.empty(local_shape, dtype=np.float64)

if rank == 0:
    # Rank 0 keeps its own slice
    local_velx[:] = velx_grid[local_start:local_stop]
    local_vely[:] = vely_grid[local_start:local_stop]
    local_velz[:] = velz_grid[local_start:local_stop]
else:
    # Other ranks receive their slices
    comm.Recv(local_velx, source=0, tag=100)
    comm.Recv(local_vely, source=0, tag=101)
    comm.Recv(local_velz, source=0, tag=102)

# Compute local velocity magnitude
local_velocity_magnitude = np.sqrt(local_velx**2 + local_vely**2 + local_velz**2)

comm.Barrier()
log_memory_global("After distributing local slabs")

if rank != 0:
    print(f"[Rank {rank}] Local data received: shape {local_shape}")

# ------------------------------------------------------------------ #
#  Visualization (rank 0 only)
# ------------------------------------------------------------------ #
if rank == 0:
    print("[Rank 0] Creating visualization...")
    
    try:
        # Create meshgrid for plotting
        X, Y, Z = np.meshgrid(x_unique, y_unique, z_unique, indexing='ij')
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot with velocity magnitude as color
        sc = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(),
                       c=velocity_magnitude.ravel(),
                       cmap='viridis', marker='.', s=1, alpha=0.6)
        
        # Add colorbar
        plt.colorbar(sc, ax=ax, label='|u|', shrink=0.8)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Velocity Magnitude\n'
                    f'⟨KE⟩={tke_physical:.6f}  '
                    f'Step {step_number_extracted}, '
                    f'Time {time_extracted:.3e}')
        
        plt.tight_layout()
        
        # # Save figure
        # output_name = f"velocity_magnitude_step_{step_number_extracted}.png"
        # plt.savefig(output_name, dpi=150, bbox_inches='tight')
        # print(f"[Rank 0] Figure saved as: {output_name}")
        
        # Show plot (comment out if running without display)
        try:
            plt.show()
        except Exception as e:
            print(f"[Rank 0] Could not display plot (no display available): {e}")
            
    except Exception as e:
        print(f"[Rank 0] Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

# ------------------------------------------------------------------ #
#  Cleanup and finalize
# ------------------------------------------------------------------ #
comm.Barrier()
log_memory_global("Before MPI finalize")

if rank == 0:
    print("[Rank 0] Visualization complete!")

MPI.Finalize()
