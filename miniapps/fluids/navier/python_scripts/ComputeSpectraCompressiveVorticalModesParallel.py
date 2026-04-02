#!/usr/bin/env python3
"""
Distributed HeFFTe/MPI version of the spectra workflow.

This first parallel version keeps file I/O on rank 0, reconstructs the
periodic grid there, and scatters structured subdomains to all ranks.
The FFTs, Helmholtz-Hodge decomposition, verification, and spectrum binning
are then performed in parallel with HeFFTe.

Example:
    mpirun -n 4 python ComputeSpectraCompressiveVorticalModesParallel.py data.h5 --backend fftw --no-plot
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from mpi4py import MPI

try:
    import heffte
except ImportError as exc:  # pragma: no cover - runtime environment specific
    raise SystemExit(
        "Unable to import heffte. Set PYTHONPATH to the HeFFTe Python wrapper "
        "install path before running this script."
    ) from exc

import ComputeSpectraCompressiveVorticalModes as serial_impl


def split_axis(length, parts):
    """Split a 1D index range into nearly equal contiguous chunks."""
    base, remainder = divmod(length, parts)
    start = 0
    chunks = []
    for i in range(parts):
        stop = start + base + (1 if i < remainder else 0)
        chunks.append((start, stop))
        start = stop
    return chunks


def choose_proc_grid(shape, nranks):
    """Choose a 3D processor grid that roughly minimizes halo surface area."""
    nx, ny, nz = shape
    best = None
    best_score = None
    for px in range(1, nranks + 1):
        if nranks % px != 0:
            continue
        rem = nranks // px
        for py in range(1, rem + 1):
            if rem % py != 0:
                continue
            pz = rem // py
            lx = nx / px
            ly = ny / py
            lz = nz / pz
            score = lx * ly + lx * lz + ly * lz
            if best_score is None or score < best_score:
                best = (px, py, pz)
                best_score = score
    return best


def build_boxes(shape, proc_grid):
    """Create rank-local boxes in the same rank order used by HeFFTe examples."""
    nx, ny, nz = shape
    px, py, pz = proc_grid
    # NumPy uses row-major layout, so the HeFFTe box order must match C-order.
    row_major_order = np.array([2, 1, 0], dtype=np.int32)
    xs = split_axis(nx, px)
    ys = split_axis(ny, py)
    zs = split_axis(nz, pz)

    boxes = []
    for kz in range(pz):
        for ky in range(py):
            for kx in range(px):
                x0, x1 = xs[kx]
                y0, y1 = ys[ky]
                z0, z1 = zs[kz]
                boxes.append(
                    heffte.box3d(
                        [x0, y0, z0],
                        [x1 - 1, y1 - 1, z1 - 1],
                        row_major_order,
                    )
                )
    return boxes


def box_shape(box):
    return tuple(int(hi - lo + 1) for lo, hi in zip(box.low, box.high))


def box_slices(box):
    return tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(box.low, box.high))


def flatten_box(field, box):
    return np.ascontiguousarray(field[box_slices(box)].ravel(order="C"))


def scatter_field(field, boxes, comm):
    """Scatter one global field from rank 0 to local flattened arrays."""
    rank = comm.Get_rank()
    payload = None
    if rank == 0:
        payload = [flatten_box(field, box) for box in boxes]
    local = comm.scatter(payload, root=0)
    return np.ascontiguousarray(local, dtype=np.float64)


def get_backend(backend_name):
    backend_name = backend_name.lower()
    backend_map = {
        "stock": heffte.backend.stock,
        "fftw": heffte.backend.fftw,
    }
    if backend_name not in backend_map:
        raise ValueError(f"Unsupported backend '{backend_name}'. Use one of: {sorted(backend_map)}")

    if backend_name == "fftw" and not getattr(heffte.heffte_config, "enable_fftw", False):
        raise RuntimeError("HeFFTe was built without FFTW support.")
    return backend_map[backend_name]


def forward_field(plan, local_field):
    local_complex = np.empty(plan.size_outbox(), dtype=np.complex128)
    plan.forward(np.ascontiguousarray(local_field.ravel(order="C")), local_complex, heffte.scale.none)
    return local_complex


def backward_field(plan, local_field_k, local_shape):
    local_real = np.empty(plan.size_inbox(), dtype=np.float64)
    plan.backward(np.ascontiguousarray(local_field_k.ravel(order="C")), local_real, heffte.scale.full)
    return local_real.reshape(local_shape, order="C")


def local_wavenumber_mesh(shape, box, dx, dy, dz):
    nx, ny, nz = shape
    sx, sy, sz = box_slices(box)
    kx_phys = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)[sx]
    ky_phys = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)[sy]
    kz_phys = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)[sz]
    return np.meshgrid(kx_phys, ky_phys, kz_phys, indexing="ij")


def local_integer_wavenumber_mesh(shape, box):
    nx, ny, nz = shape
    sx, sy, sz = box_slices(box)
    kx_int = np.fft.fftfreq(nx, 1.0 / nx).astype(int)[sx]
    ky_int = np.fft.fftfreq(ny, 1.0 / ny).astype(int)[sy]
    kz_int = np.fft.fftfreq(nz, 1.0 / nz).astype(int)[sz]
    return np.meshgrid(kx_int, ky_int, kz_int, indexing="ij")


def global_mean_energy(vx, vy, vz, global_points, comm):
    local_sum = np.sum(vx**2 + vy**2 + vz**2, dtype=np.float64)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    return 0.5 * global_sum / float(global_points)


def global_range(values, comm):
    local_min = np.min(values)
    local_max = np.max(values)
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    return global_min, global_max


def print_component_ranges(name, vx, vy, vz, comm, root):
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    vx_rng = global_range(vx, comm)
    vy_rng = global_range(vy, comm)
    vz_rng = global_range(vz, comm)
    vm_rng = global_range(vmag, comm)
    if root:
        print(f"  {name}:")
        print(f"    vx: [{vx_rng[0]:.8f}, {vx_rng[1]:.8f}]")
        print(f"    vy: [{vy_rng[0]:.8f}, {vy_rng[1]:.8f}]")
        print(f"    vz: [{vz_rng[0]:.8f}, {vz_rng[1]:.8f}]")
        print(f"    |v|: [{vm_rng[0]:.8f}, {vm_rng[1]:.8f}]")


def verify_decomposition(plan, local_shape, KX, KY, KZ, vx_c_k, vy_c_k, vz_c_k, vx_r_k, vy_r_k, vz_r_k, comm, root):
    curl_c_x_k = 1j * (KY * vz_c_k - KZ * vy_c_k)
    curl_c_y_k = 1j * (KZ * vx_c_k - KX * vz_c_k)
    curl_c_z_k = 1j * (KX * vy_c_k - KY * vx_c_k)
    curl_c_x = backward_field(plan, curl_c_x_k, local_shape)
    curl_c_y = backward_field(plan, curl_c_y_k, local_shape)
    curl_c_z = backward_field(plan, curl_c_z_k, local_shape)
    curl_c_mag = np.sqrt(curl_c_x**2 + curl_c_y**2 + curl_c_z**2)

    div_r_k = 1j * (KX * vx_r_k + KY * vy_r_k + KZ * vz_r_k)
    div_r = backward_field(plan, div_r_k, local_shape)

    max_curl = comm.allreduce(np.max(np.abs(curl_c_mag)), op=MPI.MAX)
    max_div = comm.allreduce(np.max(np.abs(div_r)), op=MPI.MAX)

    if root:
        print("Verifying decomposition quality...")
        print(f"  Max |curl(v_compressive)|: {max_curl:.2e} (should be ~0)")
        print(f"  Max |div(v_rotational)|:  {max_div:.2e} (should be ~0)")


def compute_energy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    nx, _, _ = shape
    norm = float(np.prod(shape))
    energy_density = 0.5 * (
        np.abs(vx_k / norm) ** 2
        + np.abs(vy_k / norm) ** 2
        + np.abs(vz_k / norm) ** 2
    )

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)

    k_max_int = int(math.ceil(nx * 0.5 * math.sqrt(3.0)))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * math.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=energy_density.ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return k_bin_centers, global_hist


def compute_enstrophy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    omega_x_k = 1j * (KY_int * vz_kn - KZ_int * vy_kn)
    omega_y_k = 1j * (KZ_int * vx_kn - KX_int * vz_kn)
    omega_z_k = 1j * (KX_int * vy_kn - KY_int * vx_kn)

    enstrophy_density = 0.5 * (
        np.abs(omega_x_k) ** 2 + np.abs(omega_y_k) ** 2 + np.abs(omega_z_k) ** 2
    )
    local_total_enstrophy = np.sum(enstrophy_density, dtype=np.float64)
    total_enstrophy = comm.allreduce(local_total_enstrophy, op=MPI.SUM)

    nx, _, _ = shape
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)
    k_max_int = int(math.ceil(nx * 0.5 * math.sqrt(3.0)))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * math.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=enstrophy_density.ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return k_bin_centers, global_hist, total_enstrophy


def compute_helicity_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    omega_x_k = 1j * (KY_int * vz_kn - KZ_int * vy_kn)
    omega_y_k = 1j * (KZ_int * vx_kn - KX_int * vz_kn)
    omega_z_k = 1j * (KX_int * vy_kn - KY_int * vx_kn)

    helicity_density = np.real(
        vx_kn * np.conj(omega_x_k) +
        vy_kn * np.conj(omega_y_k) +
        vz_kn * np.conj(omega_z_k)
    )

    nx, _, _ = shape
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)
    k_max_int = int(math.ceil(nx * 0.5 * math.sqrt(3.0)))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * math.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=helicity_density.ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return k_bin_centers, global_hist


def compute_energy_dissipation_enstrophy(vx_k, vy_k, vz_k, shape, box, comm, root):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    energy_density = 0.5 * (
        np.abs(vx_kn) ** 2 + np.abs(vy_kn) ** 2 + np.abs(vz_kn) ** 2
    )
    local_total_ke = np.sum(energy_density, dtype=np.float64)
    total_ke = comm.allreduce(local_total_ke, op=MPI.SUM)

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    k_squared = KX_int**2 + KY_int**2 + KZ_int**2
    local_diss = np.sum(energy_density * k_squared, dtype=np.float64)
    total_diss = comm.allreduce(local_diss, op=MPI.SUM)

    omega_x_k = 1j * (KY_int * vz_kn - KZ_int * vy_kn)
    omega_y_k = 1j * (KZ_int * vx_kn - KX_int * vz_kn)
    omega_z_k = 1j * (KX_int * vy_kn - KY_int * vx_kn)
    local_enstrophy = 0.5 * np.sum(
        np.abs(omega_x_k) ** 2 + np.abs(omega_y_k) ** 2 + np.abs(omega_z_k) ** 2,
        dtype=np.float64,
    )
    total_enstrophy = comm.allreduce(local_enstrophy, op=MPI.SUM)

    if root:
        print(f"  Total kinetic energy (fourier): {total_ke:.8f}")
        print(f"  Total k^2-weighted energy: {total_diss:.8f}")
        print("  Enstrophy vs total k^2-weighted energy comparison (should be close)")
        print(f"  {total_enstrophy:.8f} {total_diss:.8f}")


def analyze_file_parallel(filename, comm, header_lines=None, chunk_size=5_000_000, backend_name="fftw", visualize=False):
    rank = comm.Get_rank()
    root = rank == 0

    if root:
        print(f"\n{'=' * 60}")
        print(f"ANALYZING: {filename}")
        print(f"{'=' * 60}")

        if filename.endswith(".h5"):
            header_lines = 0
        elif header_lines is None:
            header_lines = serial_impl.detect_header_lines(filename)

        step_number, time_value = serial_impl.read_data_file_header(filename, header_lines)
        grid_vx, grid_vy, grid_vz, x_unique, y_unique, z_unique, dx, dy, dz = serial_impl.read_data_file_chunked(
            filename, chunk_size=chunk_size, skiprows=header_lines
        )
        shape = (len(x_unique), len(y_unique), len(z_unique))
    else:
        step_number = None
        time_value = None
        grid_vx = grid_vy = grid_vz = None
        x_unique = y_unique = z_unique = None
        dx = dy = dz = None
        shape = None
        if filename.endswith(".h5"):
            header_lines = 0

    header_lines = comm.bcast(header_lines, root=0)
    step_number = comm.bcast(step_number, root=0)
    time_value = comm.bcast(time_value, root=0)
    shape = comm.bcast(shape, root=0)
    dx = comm.bcast(dx, root=0)
    dy = comm.bcast(dy, root=0)
    dz = comm.bcast(dz, root=0)

    proc_grid = choose_proc_grid(shape, comm.Get_size())
    boxes = build_boxes(shape, proc_grid)
    local_box = boxes[rank]
    local_shape = box_shape(local_box)
    global_points = int(np.prod(shape))

    if root:
        print(f"Using processor grid: {proc_grid}")
        print(f"Local box size on rank 0: {local_shape}")

    local_vx_flat = scatter_field(grid_vx, boxes, comm)
    local_vy_flat = scatter_field(grid_vy, boxes, comm)
    local_vz_flat = scatter_field(grid_vz, boxes, comm)

    if root:
        del grid_vx, grid_vy, grid_vz

    local_vx = local_vx_flat.reshape(local_shape, order="C")
    local_vy = local_vy_flat.reshape(local_shape, order="C")
    local_vz = local_vz_flat.reshape(local_shape, order="C")

    total_ke = global_mean_energy(local_vx, local_vy, local_vz, global_points, comm)

    KX, KY, KZ = local_wavenumber_mesh(shape, local_box, dx, dy, dz)
    K_squared = KX**2 + KY**2 + KZ**2
    nonzero_mask = K_squared > 0.0

    backend = get_backend(backend_name)
    plan = heffte.fft3d(backend, local_box, local_box, comm)

    if root:
        print("Performing HeFFTe Helmholtz-Hodge decomposition...")

    vx_k = forward_field(plan, local_vx)
    vy_k = forward_field(plan, local_vy)
    vz_k = forward_field(plan, local_vz)

    vx_k = vx_k.reshape(local_shape, order="C")
    vy_k = vy_k.reshape(local_shape, order="C")
    vz_k = vz_k.reshape(local_shape, order="C")

    k_dot_v = KX * vx_k + KY * vy_k + KZ * vz_k
    projection = np.zeros_like(k_dot_v, dtype=np.complex128)
    projection[nonzero_mask] = k_dot_v[nonzero_mask] / K_squared[nonzero_mask]

    vx_c_k = KX * projection
    vy_c_k = KY * projection
    vz_c_k = KZ * projection

    vx_r_k = vx_k - vx_c_k
    vy_r_k = vy_k - vy_c_k
    vz_r_k = vz_k - vz_c_k

    vx_c = backward_field(plan, vx_c_k, local_shape)
    vy_c = backward_field(plan, vy_c_k, local_shape)
    vz_c = backward_field(plan, vz_c_k, local_shape)

    vx_r = backward_field(plan, vx_r_k, local_shape)
    vy_r = backward_field(plan, vy_r_k, local_shape)
    vz_r = backward_field(plan, vz_r_k, local_shape)

    comp_ke = global_mean_energy(vx_c, vy_c, vz_c, global_points, comm)
    rot_ke = global_mean_energy(vx_r, vy_r, vz_r, global_points, comm)

    if root:
        comp_pct = 100.0 * comp_ke / total_ke if total_ke > 0.0 else 0.0
        rot_pct = 100.0 * rot_ke / total_ke if total_ke > 0.0 else 0.0
        print("Energy breakdown:")
        print(f"  Total: {total_ke:.8f}")
        print(f"  Compressive: {comp_ke:.8f} ({comp_pct:.1f}%)")
        print(f"  Rotational: {rot_ke:.8f} ({rot_pct:.1f}%)")
        print(f"  Sum: {comp_ke + rot_ke:.8f}")
        print("Decomposition component ranges:")

    print_component_ranges("Compressive component", vx_c, vy_c, vz_c, comm, root)
    print_component_ranges("Rotational component", vx_r, vy_r, vz_r, comm, root)

    verify_decomposition(plan, local_shape, KX, KY, KZ, vx_c_k, vy_c_k, vz_c_k, vx_r_k, vy_r_k, vz_r_k, comm, root)

    if root:
        print("Computing distributed energy spectra...")
    k_centers, E_total = compute_energy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, comm)
    _, E_comp = compute_energy_spectrum_from_modes(vx_c_k, vy_c_k, vz_c_k, shape, local_box, comm)
    _, E_rot = compute_energy_spectrum_from_modes(vx_r_k, vy_r_k, vz_r_k, shape, local_box, comm)
    _, Enst, total_enstrophy = compute_enstrophy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, comm)
    _, Hel = compute_helicity_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, comm)
    if root:
        print(f"  Total enstrophy (fourier, code convention): {total_enstrophy:.8f}")

    compute_energy_dissipation_enstrophy(vx_k, vy_k, vz_k, shape, local_box, comm, root)

    result = None
    if root:
        serial_impl.save_spectra(
            k_centers,
            E_total,
            E_comp,
            E_rot,
            Enst,
            Hel,
            filename,
            step_number,
            time_value,
            shape[0],
            shape[1],
            shape[2],
            total_ke,
            comp_ke,
            rot_ke,
            total_enstrophy,
        )
        if visualize:
            print("Visualization is not implemented in the parallel script yet.")
        result = (k_centers, E_total, E_comp, E_rot, step_number, time_value)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Distributed HeFFTe Helmholtz-Hodge decomposition and spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mpirun -n 4 python ComputeSpectraCompressiveVorticalModesParallel.py data_file.h5 --backend fftw --no-plot
  mpirun -n 8 python ComputeSpectraCompressiveVorticalModesParallel.py file1.txt file2.txt --header-lines 5
        """,
    )
    parser.add_argument("data_files", type=str, nargs="+", help="One or more velocity data files to analyze")
    parser.add_argument(
        "--header-lines",
        type=int,
        default=None,
        help="Number of header lines to skip/read. If omitted, attempts auto-detection on rank 0.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fftw",
        choices=["fftw", "stock"],
        help="HeFFTe backend to use.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Chunk size used when rank 0 reads the input file.",
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Reserved for future use.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting spectra on rank 0.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    results = []
    for filename in args.data_files:
        result = analyze_file_parallel(
            filename,
            comm,
            header_lines=args.header_lines,
            chunk_size=args.chunk_size,
            backend_name=args.backend,
            visualize=args.visualize,
        )
        if rank == 0:
            results.append(result)
        comm.Barrier()

    if rank == 0:
        if not args.no_plot:
            serial_impl.plot_spectra(results)
        else:
            print(f"\nProcessed {len(results)} files. Spectrum files saved to disk.")
            print("Skipping plot display as requested (--no-plot flag).")


if __name__ == "__main__":
    main()
