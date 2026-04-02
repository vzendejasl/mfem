"""
Parallel MPI version of convert_txt_to_hdf5.py

TXT -> H5: Fully parallel — rank 0 builds a byte-offset index in one scan,
           then all ranks seek+read their own chunks and write to HDF5 in parallel.
H5 -> TXT: Serial on rank 0 — parallel text writing requires pre-computing
           per-row byte offsets (non-trivial), so not worth the complexity.

NOTE: driver='mpio' does not support gzip compression.  Files will be larger
      than the serial gzip'd HDF5 output, but still smaller than raw text
      (binary float64 vs ASCII %.16e).

================================================================================
ENVIRONMENT & DEPENDENCIES
================================================================================

Tested with the 'dedalus3' conda environment, which ships h5py pre-built
against a parallel (MPI-enabled) HDF5 library.

Required packages (all present in dedalus3):
  - mpi4py  >= 4.0    (Python MPI bindings)
  - h5py             (must be the parallel build — h5py.get_config().mpi == True)
  - numpy
  - pandas

To verify your h5py build supports MPI before running:
  python -c "import h5py; print(h5py.get_config().mpi)"
  # Must print: True

To activate the environment:
  conda activate dedalus3

================================================================================
HOW TO RUN
================================================================================

Basic usage (replace N with the number of MPI ranks):
  mpirun -n N python convert_txt_to_hdf5_parallel.py <file1.txt> [file2.txt ...]

Examples:
  # Single file, 4 ranks
  mpirun -n 4 python convert_txt_to_hdf5_parallel.py SampledData0.txt

  # Multiple files, 8 ranks (all ranks cooperate on one file at a time)
  mpirun -n 8 python convert_txt_to_hdf5_parallel.py SampledData0.txt SampledData1.txt

  # Convert HDF5 back to text (serial — runs on rank 0 only regardless of N)
  mpirun -n 1 python convert_txt_to_hdf5_parallel.py SampledData0.h5

Rule of thumb for N: use one rank per physical core up to the number of
CHUNK_SIZE-row blocks in your file.  More ranks than chunks gives no benefit
since idle ranks have no chunks to process.

================================================================================
"""

import sys
import os
import io
import numpy as np
import h5py
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Number of rows per read/write chunk
CHUNK_SIZE = 1_000_000


# ---------------------------------------------------------------------------
# Header detection (unchanged from serial version)
# ---------------------------------------------------------------------------

def get_txt_header(txt_file_path):
    """Rank 0 only: find header length. Returns (header_lines, count)."""
    print("  Analyzing text file header...")
    header_lines, header_count = [], 0
    try:
        with open(txt_file_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    header_lines.append(line)
                    header_count += 1
                    continue
                try:
                    [float(x) for x in stripped.split()]
                    break  # first all-numeric line = data starts here
                except ValueError:
                    header_lines.append(line)
                    header_count += 1
        print(f"  Detected header length: {header_count} lines")
        return header_lines, header_count
    except Exception as e:
        print(f"  Error analyzing file: {e}")
        return None, 0


# ---------------------------------------------------------------------------
# Chunk index — rank 0 scans once, broadcasts to all
# ---------------------------------------------------------------------------

def build_chunk_index(txt_path, skip_count):
    """
    Rank 0 only.  Scan file after the header and record
    (byte_offset, num_rows) for every CHUNK_SIZE block of rows.

    Opens in binary mode so f.tell() is reliable on all platforms.
    Python's buffered I/O makes this fast despite the readline loop.
    """
    chunks = []
    with open(txt_path, 'rb') as f:
        for _ in range(skip_count):
            f.readline()  # skip header
        while True:
            offset = f.tell()
            count = 0
            for _ in range(CHUNK_SIZE):
                if not f.readline():
                    break
                count += 1
            if count == 0:
                break
            chunks.append((offset, count))
    return chunks


# ---------------------------------------------------------------------------
# Per-rank text reading
# ---------------------------------------------------------------------------

def read_chunk_at_offset(txt_path, byte_offset, num_rows):
    """
    All ranks.  Seek directly to byte_offset, read exactly num_rows lines,
    parse with pandas (fast C parser), return float64 array.
    """
    with open(txt_path, 'rb') as f:
        f.seek(byte_offset)
        buf = b''.join(f.readline() for _ in range(num_rows))
    return pd.read_csv(io.BytesIO(buf), header=None, sep=r'\s+').values.astype(np.float64)


# ---------------------------------------------------------------------------
# TXT -> H5  (fully parallel)
# ---------------------------------------------------------------------------

def convert_txt_to_h5_parallel(txt_path, h5_path):
    header_lines = chunk_index = None
    total_rows = 0

    # --- rank 0: parse header, build chunk index ---
    if rank == 0:
        header_lines, skip_count = get_txt_header(txt_path)
        if header_lines is not None:
            print("  Building chunk index (rank 0 scans file once)...")
            chunk_index = build_chunk_index(txt_path, skip_count)
            total_rows = sum(n for _, n in chunk_index)
            print(f"  {total_rows} rows | {len(chunk_index)} chunks | {size} MPI ranks")

    # --- broadcast metadata to all ranks ---
    header_lines = comm.bcast(header_lines, root=0)
    chunk_index  = comm.bcast(chunk_index,  root=0)
    total_rows   = comm.bcast(total_rows,   root=0)

    if header_lines is None or not chunk_index:
        return None, 0

    n_chunks = len(chunk_index)

    # Compute the global HDF5 row start for each chunk (cumulative sum)
    row_starts = [0] * n_chunks
    for i in range(1, n_chunks):
        row_starts[i] = row_starts[i - 1] + chunk_index[i - 1][1]

    # Round-robin chunk assignment across ranks
    my_chunk_indices = range(rank, n_chunks, size)

    running_sum_sq = 0.0

    if rank == 0:
        print("  Converting TXT -> H5 in parallel (no gzip — driver='mpio' limitation)...")

    # --- Phase 1: all ranks write data in parallel ---
    with h5py.File(h5_path, 'w', driver='mpio', comm=comm) as hf:
        # All ranks must call create_dataset collectively
        dset = hf.create_dataset('data', shape=(total_rows, 6), dtype='float64')

        for ci in my_chunk_indices:
            byte_off, nrows = chunk_index[ci]
            h5_row = row_starts[ci]

            chunk_data = read_chunk_at_offset(txt_path, byte_off, nrows)
            dset[h5_row : h5_row + nrows] = chunk_data

            vx, vy, vz = chunk_data[:, 3], chunk_data[:, 4], chunk_data[:, 5]
            running_sum_sq += np.sum(vx**2 + vy**2 + vz**2)

            if rank == 0:
                print(f"    Rank 0: chunk {ci + 1}/{n_chunks}...", end='\r')

    comm.Barrier()

    # --- Phase 2: rank 0 appends header in a serial re-open ---
    # This avoids the complexity of writing variable-length strings with mpio.
    if rank == 0:
        with h5py.File(h5_path, 'a') as hf:
            dt = h5py.string_dtype(encoding='utf-8')
            hf.create_dataset('header',
                              data=np.array([str(l) for l in header_lines], dtype=object),
                              dtype=dt)
        print(f"\n  Conversion complete. Total rows: {total_rows}")

    # --- reduce TKE across all ranks, broadcast result ---
    global_sum_sq = comm.reduce(running_sum_sq, op=MPI.SUM, root=0)
    tke = None
    if rank == 0:
        tke = 0.5 * (global_sum_sq / total_rows)
    tke = comm.bcast(tke, root=0)

    return tke, total_rows


# ---------------------------------------------------------------------------
# H5 -> TXT  (serial on rank 0 — text writing is the bottleneck anyway)
# ---------------------------------------------------------------------------

def convert_h5_to_txt_chunked(h5_path, txt_path):
    """
    Rank 0 only.  Parallel text writing would require pre-computing per-row
    byte offsets (hard for %.16e whose exponent width varies) — not worth it.
    """
    if rank != 0:
        return None, 0

    try:
        print("  Converting H5 -> TXT (serial, rank 0)...")
        with h5py.File(h5_path, 'r') as hf:
            if 'data' not in hf:
                print("  Error: No 'data' dataset in HDF5 file.")
                return None, 0

            dset = hf['data']
            total_rows_h5 = dset.shape[0]

            header_lines = []
            if 'header' in hf:
                for line in hf['header'][:]:
                    l = line.decode('utf-8') if isinstance(line, bytes) else str(line)
                    header_lines.append(l)
            header_lines.insert(0, "# Data restored from HDF5 conversion (Original source: Text file)\n")

            running_sum_sq = 0.0
            processed_rows = 0

            with open(txt_path, 'w') as f_txt:
                for line in header_lines:
                    f_txt.write(line)
                for i in range(0, total_rows_h5, CHUNK_SIZE):
                    chunk_data = dset[i : i + CHUNK_SIZE]
                    np.savetxt(f_txt, chunk_data, fmt='%.16e', delimiter=' ')
                    vx, vy, vz = chunk_data[:, 3], chunk_data[:, 4], chunk_data[:, 5]
                    running_sum_sq += np.sum(vx**2 + vy**2 + vz**2)
                    processed_rows += len(chunk_data)
                    print(f"    {processed_rows}/{total_rows_h5} rows...", end='\r')

            print(f"\n  Conversion complete.")
            return 0.5 * (running_sum_sq / processed_rows), processed_rows

    except Exception as e:
        print(f"\n  Error during H5->TXT: {e}")
        return None, 0


# ---------------------------------------------------------------------------
# TKE verification  (parallel for H5, serial on rank 0 for TXT)
# ---------------------------------------------------------------------------

def calculate_file_tke_parallel(file_path):
    """Returns the same (tke, total_rows) on all ranks."""
    if rank == 0:
        print(f"  Verifying: {os.path.basename(file_path)}...")

    _, ext = os.path.splitext(file_path)
    running_sum_sq = 0.0
    local_rows = 0

    try:
        if ext == '.h5':
            # All ranks read different slices in parallel
            with h5py.File(file_path, 'r', driver='mpio', comm=comm) as hf:
                dset = hf['data']
                total = dset.shape[0]
                for start in range(rank * CHUNK_SIZE, total, size * CHUNK_SIZE):
                    chunk = dset[start : start + CHUNK_SIZE]
                    vx, vy, vz = chunk[:, 3], chunk[:, 4], chunk[:, 5]
                    running_sum_sq += np.sum(vx**2 + vy**2 + vz**2)
                    local_rows += len(chunk)
        else:
            # TXT: rank 0 reads serially (only needed when verifying H5->TXT output)
            if rank == 0:
                _, skip = get_txt_header(file_path)
                reader = pd.read_csv(file_path, skiprows=skip, header=None,
                                     sep=r'\s+', chunksize=CHUNK_SIZE)
                for chunk_df in reader:
                    chunk = chunk_df.values.astype(np.float64)
                    vx, vy, vz = chunk[:, 3], chunk[:, 4], chunk[:, 5]
                    running_sum_sq += np.sum(vx**2 + vy**2 + vz**2)
                    local_rows += len(chunk)

        global_sum_sq = comm.reduce(running_sum_sq, op=MPI.SUM, root=0)
        total_rows    = comm.reduce(local_rows,     op=MPI.SUM, root=0)

        tke = None
        if rank == 0:
            tke = 0.5 * (global_sum_sq / total_rows) if total_rows > 0 else 0.0
        tke        = comm.bcast(tke,        root=0)
        total_rows = comm.bcast(total_rows, root=0)

        return tke, total_rows

    except Exception as e:
        if rank == 0:
            print(f"  Error verifying file: {e}")
        return None, 0


# ---------------------------------------------------------------------------
# Storage stats
# ---------------------------------------------------------------------------

def print_storage_stats(path1, path2):
    size1 = os.path.getsize(path1)
    size2 = os.path.getsize(path2)
    diff  = size1 - size2
    print("-" * 40)
    print(f"Storage Comparison:")
    print(f"  Input:  {size1 / (1024*1024):.2f} MB")
    print(f"  Output: {size2 / (1024*1024):.2f} MB")
    if diff > 0:
        print(f"  Saved:  {diff / (1024*1024):.2f} MB ({(diff/size1)*100:.2f}%)")
    else:
        print(f"  Growth: {abs(diff) / (1024*1024):.2f} MB")
    print("-" * 40)


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def convert_file(input_path):
    if rank == 0:
        print(f"\nProcessing: {input_path}")

    exists = comm.bcast(os.path.exists(input_path) if rank == 0 else False, root=0)
    if not exists:
        if rank == 0:
            print("Error: File not found.")
        return False

    base, ext = os.path.splitext(input_path)
    input_tke = None
    input_rows = 0
    output_path = ""

    # --- conversion phase ---
    if ext == '.txt':
        output_path = base + '.h5'
        input_tke, input_rows = convert_txt_to_h5_parallel(input_path, output_path)
        # convert_txt_to_h5_parallel already bcasts; values are consistent on all ranks

    elif ext == '.h5':
        output_path = base + '.txt'
        input_tke, input_rows = convert_h5_to_txt_chunked(input_path, output_path)
        # serial — bcast rank 0's result
        input_tke  = comm.bcast(input_tke,  root=0)
        input_rows = comm.bcast(input_rows, root=0)

    else:
        if rank == 0:
            print(f"Error: Unsupported extension '{ext}'.")
        return False

    if input_tke is None:
        return False

    # --- verification phase ---
    output_tke, output_rows = calculate_file_tke_parallel(output_path)
    # calculate_file_tke_parallel already bcasts; values are consistent on all ranks

    if output_tke is None:
        return False

    success = False
    if rank == 0:
        print(f"  Original TKE:       {input_tke:.16f}")
        print(f"  Reconstructed TKE:  {output_tke:.16f}")
        print(f"  Original Rows:      {input_rows}")
        print(f"  Reconstructed Rows: {output_rows}")

        tke_match = np.isclose(input_tke, output_tke, atol=1e-12)
        row_match = (input_rows == output_rows)

        if tke_match and row_match:
            print("  SUCCESS: TKE and Row Counts match.")
            print_storage_stats(input_path, output_path)
            if ext == '.txt':
                print(f"  Deleting original: {input_path}")
                try:
                    os.remove(input_path)
                except OSError as e:
                    print(f"  Warning: Could not delete: {e}")
            success = True
        else:
            print("\n" + "!" * 60)
            print("  !!! ERROR: DATA INTEGRITY FAILED !!!")
            if not tke_match:
                print(f"  TKE mismatch! Diff: {abs(input_tke - output_tke):.2e}")
            if not row_match:
                print(f"  Row mismatch! Input: {input_rows}, Output: {output_rows}")
            print("!" * 60 + "\n")

    return comm.bcast(success, root=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        if rank == 0:
            print("\nUsage: mpirun -n N python convert_txt_to_hdf5_parallel.py <file1> [file2 ...]")
        sys.exit(0)

    files = sys.argv[1:]
    failures = 0

    if rank == 0:
        print(f"Batch processing {len(files)} file(s) with {size} MPI ranks...")

    # All ranks cooperate on one file at a time (intra-file parallelism).
    # For many small files, file-level distribution is faster — see serial version.
    for file_path in files:
        try:
            if not convert_file(file_path):
                failures += 1
        except Exception as e:
            if rank == 0:
                print(f"  CRITICAL ERROR processing {file_path}: {e}")
            failures += 1

    if rank == 0:
        print("\n" + "=" * 40)
        if failures == 0:
            print(f"Batch completed successfully. {len(files)} file(s) processed.")
        else:
            print(f"Batch completed with ERRORS. {failures}/{len(files)} file(s) failed.")

    sys.exit(0 if failures == 0 else 1)
