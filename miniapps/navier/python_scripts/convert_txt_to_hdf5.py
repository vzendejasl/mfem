import sys
import os
import numpy as np
import h5py
import pandas as pd

# Process in chunks of 1 million rows to save memory
CHUNK_SIZE = 1_000_000

def get_txt_header(txt_file_path):
    """
    Analyzes the text file to find the header length.
    Returns: (header_lines_list, header_line_count)
    """
    print("  Analyzing text file header...")
    header_lines = []
    header_count = 0
    
    try:
        with open(txt_file_path, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    header_lines.append(line)
                    header_count += 1
                    continue
                
                try:
                    parts = line_stripped.split()
                    # Try to parse first few tokens as floats
                    [float(x) for x in parts]
                    break # Found data
                except ValueError:
                    header_lines.append(line)
                    header_count += 1
        
        print(f"  Detected header length: {header_count} lines")
        return header_lines, header_count

    except Exception as e:
        print(f"  Error analyzing file structure: {e}")
        return None, 0

def update_tke_stats(chunk_data, current_sum_sq, current_count):
    """
    Updates the running sum of squares for TKE calculation.
    """
    # Columns 3, 4, 5 are vx, vy, vz
    vx = chunk_data[:, 3]
    vy = chunk_data[:, 4]
    vz = chunk_data[:, 5]
    
    sq_sum = np.sum(vx**2 + vy**2 + vz**2)
    return current_sum_sq + sq_sum, current_count + len(chunk_data)

def convert_txt_to_h5_chunked(txt_path, h5_path):
    header_lines, skip_count = get_txt_header(txt_path)
    if header_lines is None: return None, 0

    try:
        # Use Pandas for efficient chunked CSV reading
        reader = pd.read_csv(txt_path, skiprows=skip_count, header=None, sep=r'\s+', chunksize=CHUNK_SIZE)
        
        print("  Converting TXT -> H5 (Chunked)...")
        
        with h5py.File(h5_path, 'w') as hf:
            # Create a resizable dataset. 
            # EXPLICITLY set dtype='float64' to ensure double precision
            dset = hf.create_dataset('data', shape=(0, 6), maxshape=(None, 6), 
                                     dtype='float64', chunks=True, compression="gzip")
            
            # Save header
            dt = h5py.string_dtype(encoding='utf-8')
            header_str = [str(l) for l in header_lines]
            hf.create_dataset('header', data=np.array(header_str, dtype=object), dtype=dt)
            
            running_sum_sq = 0.0
            total_rows = 0
            
            for i, chunk_df in enumerate(reader):
                chunk_data = chunk_df.values.astype(np.float64)
                chunk_len = len(chunk_data)
                
                # Resize dataset to accommodate new chunk
                dset.resize(total_rows + chunk_len, axis=0)
                dset[total_rows : total_rows + chunk_len] = chunk_data
                
                # Update TKE stats
                running_sum_sq, total_rows = update_tke_stats(chunk_data, running_sum_sq, total_rows)
                
                print(f"    Processed chunk {i+1} ({total_rows} rows total)...", end='\r')
            
            print(f"\n  Conversion complete. Total rows: {total_rows}")
            
            # Calculate final TKE for input
            input_tke = 0.5 * (running_sum_sq / total_rows)
            return input_tke, total_rows

    except Exception as e:
        print(f"\n  Error during TXT->H5 conversion: {e}")
        return None, 0

def convert_h5_to_txt_chunked(h5_path, txt_path):
    try:
        print("  Converting H5 -> TXT (Chunked)...")
        
        with h5py.File(h5_path, 'r') as hf:
            if 'data' not in hf:
                print("  Error: No 'data' dataset in HDF5 file.")
                return None, 0
            
            dset = hf['data']
            total_rows_h5 = dset.shape[0]
            
            # Read header
            header_lines = []
            if 'header' in hf:
                header_ds = hf['header'][:]
                for line in header_ds:
                    l = line.decode('utf-8') if isinstance(line, bytes) else str(line)
                    header_lines.append(l)
            
            # ADD COMMENT TO HEADER
            # Prepend a line indicating provenance
            header_lines.insert(0, "# Data restored from HDF5 conversion (Original source: Text file)\n")

            running_sum_sq = 0.0
            processed_rows = 0
            
            with open(txt_path, 'w') as f_txt:
                # Write Header
                for line in header_lines:
                    f_txt.write(line)
                
                # Loop through HDF5 in chunks
                for i in range(0, total_rows_h5, CHUNK_SIZE):
                    chunk_data = dset[i : i + CHUNK_SIZE]
                    
                    # Write to Text
                    np.savetxt(f_txt, chunk_data, fmt='%.16e', delimiter=' ')
                    
                    # Update TKE stats
                    running_sum_sq, processed_rows = update_tke_stats(chunk_data, running_sum_sq, processed_rows)
                    
                    print(f"    Processed {processed_rows}/{total_rows_h5} rows...", end='\r')
            
            print(f"\n  Conversion complete.")
            
            # Calculate final TKE for input
            input_tke = 0.5 * (running_sum_sq / processed_rows)
            return input_tke, processed_rows

    except Exception as e:
        print(f"\n  Error during H5->TXT conversion: {e}")
        return None, 0

def calculate_file_tke_chunked(file_path):
    """
    Independently reads a file (TXT or H5) in chunks to calculate TKE and count rows.
    """
    print(f"  Verifying output file: {os.path.basename(file_path)}...")
    _, ext = os.path.splitext(file_path)
    
    running_sum_sq = 0.0
    total_rows = 0
    
    try:
        if ext == '.h5':
            with h5py.File(file_path, 'r') as hf:
                dset = hf['data']
                rows = dset.shape[0]
                for i in range(0, rows, CHUNK_SIZE):
                    chunk = dset[i : i + CHUNK_SIZE]
                    running_sum_sq, total_rows = update_tke_stats(chunk, running_sum_sq, total_rows)
        
        else: # TXT
            _, skip = get_txt_header(file_path)
            # Handle empty file check if needed, but pd usually handles it
            reader = pd.read_csv(file_path, skiprows=skip, header=None, sep=r'\s+', chunksize=CHUNK_SIZE)
            for chunk_df in reader:
                chunk = chunk_df.values.astype(np.float64)
                running_sum_sq, total_rows = update_tke_stats(chunk, running_sum_sq, total_rows)
                
        if total_rows == 0:
            return 0.0, 0
            
        return 0.5 * (running_sum_sq / total_rows), total_rows
        
    except Exception as e:
        print(f"  Error reading verification file: {e}")
        return None, 0

def print_storage_stats(path1, path2):
    size1 = os.path.getsize(path1)
    size2 = os.path.getsize(path2)
    diff = size1 - size2
    
    print("-" * 40)
    print(f"Storage Comparison:")
    print(f"  Input:  {size1 / (1024*1024):.2f} MB")
    print(f"  Output: {size2 / (1024*1024):.2f} MB")
    if diff > 0:
        print(f"  Saved:  {diff / (1024*1024):.2f} MB ({(diff/size1)*100:.2f}%)")
    else:
        print(f"  Growth: {abs(diff) / (1024*1024):.2f} MB")
    print("-" * 40)

def convert_file(input_path):
    print(f"\nProcessing: {input_path}")
    if not os.path.exists(input_path):
        print("Error: File not found.")
        return False

    base, ext = os.path.splitext(input_path)
    input_tke = None
    input_rows = 0
    output_path = ""
    
    # --- CONVERSION PHASE ---
    if ext == '.txt':
        output_path = base + '.h5'
        input_tke, input_rows = convert_txt_to_h5_chunked(input_path, output_path)
    elif ext == '.h5':
        output_path = base + '.txt' # Restore to clean .txt name
        input_tke, input_rows = convert_h5_to_txt_chunked(input_path, output_path)
    else:
        print(f"Error: Unsupported file extension '{ext}'.")
        return False
        
    if input_tke is None:
        return False

    # --- VERIFICATION PHASE ---
    output_tke, output_rows = calculate_file_tke_chunked(output_path)
    
    if output_tke is None:
        return False
        
    print(f"  Original TKE:      {input_tke:.16f}")
    print(f"  Reconstructed TKE: {output_tke:.16f}")
    print(f"  Original Rows:     {input_rows}")
    print(f"  Reconstructed Rows: {output_rows}")
    
    tke_match = np.isclose(input_tke, output_tke, atol=1e-12)
    row_match = (input_rows == output_rows)
    
    if tke_match and row_match:
        print("  SUCCESS: TKE and Row Counts match.")
        print_storage_stats(input_path, output_path)
        
        # --- DELETE ORIGINAL IF .TXT ---
        if ext == '.txt':
            print(f"  Deleting original text file: {input_path}")
            try:
                os.remove(input_path)
            except OSError as e:
                print(f"  Warning: Could not delete file: {e}")
        
        return True
    else:
        print("\n" + "!"*60)
        print(f"  !!! ERROR: DATA INTEGRITY FAILED !!!")
        if not tke_match:
            print(f"  TKE mismatch detected! Diff: {abs(input_tke - output_tke):.2e}")
        if not row_match:
            print(f"  Row count mismatch! Input: {input_rows}, Output: {output_rows}")
        print("!"*60 + "\n")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python convert_txt_to_hdf5.py <file_path1> [file_path2 ...]")
        sys.exit(0)

    files = sys.argv[1:]
    failures = 0
    
    print(f"Batch processing {len(files)} files...")
    
    for file_path in files:
        try:
            if not convert_file(file_path):
                failures += 1
        except Exception as e:
            print(f"  CRITICAL ERROR processing {file_path}: {e}")
            failures += 1

    print("\n" + "="*40)
    if failures == 0:
        print(f"Batch completed successfully. {len(files)} files processed.")
        sys.exit(0)
    else:
        print(f"Batch completed with ERRORS. {failures}/{len(files)} files failed.")
        sys.exit(1)
