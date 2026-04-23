#!/usr/bin/env python3
"""
convert_tgv_file_to_csv.py (robust)

Supports:
- TGV-like text logs with whitespace-delimited numeric tables and a header line.
- Proper comma-separated CSV inputs.

Always:
- Deduplicates by time (stable sort by time, keep last by default),
  optionally rounding time to collapse near duplicates.

Output:
- If input is .csv, overwrite same file path.
- Otherwise write <input_base>.csv
"""

import sys
import os
import re
import glob
import csv
import argparse
from typing import List, Optional, Tuple, Iterable

SEP_RE = re.compile(r'^[=\-\s]{5,}$')

def is_separator(line: str) -> bool:
    return bool(SEP_RE.match(line.strip()))

def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def is_data_line_whitespace(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    toks = re.split(r'\s+', s)
    if len(toks) < 2:
        return False
    return all(is_float_token(t) for t in toks)

def split_header_cells_ws(header_line: str) -> List[str]:
    """
    Split header cells for fixed-width headers.
    Prefer splits on 2+ spaces or tabs to preserve multi-word column names.
    """
    line = header_line.strip()
    cells = re.split(r'(?:\t+|\s{2,})', line)
    return [c.strip() for c in cells if c.strip()]

def normalize_header_for_matching(h: str) -> str:
    """
    Normalize header label for matching 'time'.
    """
    s = h.strip().lower()
    # remove common punctuation/brackets
    s = re.sub(r'[\[\]\(\)\{\}]', ' ', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def find_time_column(headers: List[str]) -> int:
    """
    Find a time column, case-insensitive, tolerant to units (time, time s, time_sec, etc.).
    Default to 0 if not found.
    """
    norm = [normalize_header_for_matching(h) for h in headers]

    # Exact-ish matches first
    for i, h in enumerate(norm):
        if h == "time":
            return i

    # Contains 'time' (covers "time s", "time sec", "simulation time", etc.)
    for i, h in enumerate(norm):
        if "time" in h.split():
            return i
        if "time" in h:
            return i

    return 0

def output_path_for_input(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    if ext.lower() == ".csv":
        return input_path
    return base + ".csv"

# ----------------------------
# CSV input handling
# ----------------------------

def sniff_is_csv(lines: List[str]) -> bool:
    """
    Content sniff: treat as CSV if any of the first few non-empty lines contain commas.
    """
    checked = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        checked += 1
        if "," in s:
            return True
        if checked >= 10:
            break
    return False

def read_csv_table(path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Read comma-separated CSV. Assumes first row is header.
    Keeps values as strings (dedupe will parse time as float).
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError("Empty CSV file, no header row found.")

        headers = [h.strip() for h in headers]
        rows: List[List[str]] = []
        for row in reader:
            if not row:
                continue
            # pad/trim to header length
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            rows.append([c.strip() for c in row])
    return headers, rows

# ----------------------------
# Whitespace TGV text handling
# ----------------------------

def detect_header_and_data_ws(lines: List[str]) -> Tuple[int, int]:
    """
    Robust header/data detection for whitespace tables.

    Strategy:
    1) Separator line (==== or ----), header is next non-empty line,
       data begins after header (next non-empty line).
    2) Fallback: find a "header-like" line (letters and multi-space gaps) whose
       next non-empty line is numeric data.
    3) Strong fallback: find the first numeric data line, then choose the closest
       previous non-empty line with any letters as the header.
    """
    n = len(lines)

    # 1) explicit separator
    for i, ln in enumerate(lines):
        if is_separator(ln):
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j >= n:
                break
            header_idx = j

            k = j + 1
            while k < n and not lines[k].strip():
                k += 1
            if k < n:
                return header_idx, k

    # 2) heuristic
    for i in range(n - 1):
        header_line = lines[i]
        # find next non-empty
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        if j >= n:
            break

        if re.search(r'[A-Za-z]', header_line) and re.search(r'(\t+|\s{2,})', header_line):
            if is_data_line_whitespace(lines[j]):
                return i, j

    # 3) strong fallback: locate first data line, then best previous header-ish line
    first_data = None
    for i, ln in enumerate(lines):
        if is_data_line_whitespace(ln):
            first_data = i
            break
    if first_data is None:
        raise ValueError("No numeric data lines detected (whitespace mode).")

    # Search backwards for a header line (letters), skip separators
    for h in range(first_data - 1, -1, -1):
        if is_separator(lines[h]):
            continue
        if re.search(r'[A-Za-z]', lines[h]):
            return h, first_data

    raise ValueError("Could not detect header/data boundary (whitespace mode).")

def parse_rows_ws(lines: List[str], data_idx: int, ncols: int) -> List[List[str]]:
    rows: List[List[str]] = []
    for ln in lines[data_idx:]:
        s = ln.strip()
        if not s:
            continue
        if is_separator(s):
            break
        if not is_data_line_whitespace(s):
            continue

        vals = re.split(r'\s+', s)
        if len(vals) >= ncols:
            row = vals[:ncols]
        else:
            row = vals + [''] * (ncols - len(vals))
        rows.append(row)
    return rows

def read_ws_table(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_idx, data_idx = detect_header_and_data_ws(lines)
    headers = split_header_cells_ws(lines[header_idx])
    if not headers:
        raise ValueError("Detected header line but could not split any header cells.")

    rows = parse_rows_ws(lines, data_idx=data_idx, ncols=len(headers))
    return headers, rows

# ----------------------------
# Dedupe
# ----------------------------

def dedupe_rows_by_time(
    rows: List[List[str]],
    time_idx: int,
    round_time_decimals: Optional[int],
    keep: str
) -> Tuple[List[List[str]], int, int]:
    """
    Similar to convert_to_non_dimensional_mfem.py:
    - drop rows where time is non-numeric
    - stable sort by time
    - dedupe by time key, keep first/last
    Returns: (deduped_rows, removed_count, non_numeric_dropped)
    """
    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")

    cleaned: List[Tuple[int, float, float, List[str]]] = []
    non_numeric = 0
    for idx, row in enumerate(rows):
        try:
            t = float(row[time_idx])
        except Exception:
            non_numeric += 1
            continue
        key = round(t, round_time_decimals) if round_time_decimals is not None else t
        cleaned.append((idx, t, key, row))

    # stable sort by time, original index as tie-breaker
    cleaned.sort(key=lambda x: (x[1], x[0]))

    seen = {}
    for idx, t, key, row in cleaned:
        if keep == "first":
            if key not in seen:
                seen[key] = (t, idx, row)
        else:
            seen[key] = (t, idx, row)

    out = list(seen.values())
    out.sort(key=lambda x: (x[0], x[1]))
    deduped_rows = [row for _, _, row in out]

    removed = len(rows) - len(deduped_rows)
    return deduped_rows, removed, non_numeric

# ----------------------------
# Conversion
# ----------------------------

def convert_one_file(input_path: str, round_time_decimals: Optional[int], keep: str) -> Tuple[str, int, int, int]:
    # Read a small chunk for sniffing
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        head_lines = f.readlines(2000)

    ext = os.path.splitext(input_path)[1].lower()
    treat_as_csv = (ext == ".csv") or sniff_is_csv(head_lines)

    if treat_as_csv:
        headers, rows = read_csv_table(input_path)
        mode = "csv"
    else:
        headers, rows = read_ws_table(input_path)
        mode = "whitespace"

    before = len(rows)
    time_idx = find_time_column(headers)

    print(f"[i] {os.path.basename(input_path)}")
    print(f"    mode={mode}")
    print(f"    columns={len(headers)}, parsed_rows_before_dedupe={before}")
    print(f"    time_column_index={time_idx}, time_column_name={headers[time_idx] if headers else 'UNKNOWN'}")

    deduped, removed, non_numeric = dedupe_rows_by_time(
        rows,
        time_idx=time_idx,
        round_time_decimals=round_time_decimals,
        keep=keep
    )
    after = len(deduped)

    if round_time_decimals is None:
        print(f"    dedupe_mode=exact_time, keep={keep}")
    else:
        print(f"    dedupe_mode=rounded_time, round_time_decimals={round_time_decimals}, keep={keep}")
    print(f"    non_numeric_time_rows_dropped={non_numeric}")
    print(f"    duplicates_removed_total={removed}")
    print(f"    parsed_rows_after_dedupe={after}")

    out_path = output_path_for_input(input_path)

    with open(out_path, "w", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        writer.writerow(headers)
        writer.writerows(deduped)

    return out_path, removed, before, after

def main():
    ap = argparse.ArgumentParser(
        description="Convert TGV-like whitespace tables or CSV to CSV, deduplicating by time (robust)."
    )
    ap.add_argument(
        "patterns", nargs="+",
        help='Input glob pattern(s) or explicit files, e.g. "*.txt" "*.csv"'
    )
    ap.add_argument(
        "--keep", choices=["first", "last"], default="last",
        help="When duplicate times occur, keep the first or last occurrence. Default: last."
    )
    ap.add_argument(
        "--round-time-decimals", type=int, default=None,
        help="Round time before dedup (e.g., 12) to collapse near-duplicates."
    )
    args = ap.parse_args()

    # Expand patterns
    inputs: List[str] = []
    for pat in args.patterns:
        matches = glob.glob(pat)
        if matches:
            inputs.extend(matches)
        else:
            if os.path.isfile(pat):
                inputs.append(pat)
            else:
                print(f"[!] Warning: pattern matched no files: {pat}", file=sys.stderr)

    if not inputs:
        print("No input files found. Nothing to do.", file=sys.stderr)
        sys.exit(2)

    total_removed = 0
    total_files = 0

    for path in sorted(set(inputs)):
        try:
            out, removed, before, after = convert_one_file(
                path,
                round_time_decimals=args.round_time_decimals,
                keep=args.keep
            )
            total_removed += removed
            total_files += 1
            print(f"[✓] wrote: {out} | rows {before} -> {after} | removed={removed}")
        except Exception as e:
            print(f"[x] Error converting {path}: {e}", file=sys.stderr)

    print(f"TOTAL: files={total_files}, duplicates_removed={total_removed}")

if __name__ == "__main__":
    main()
