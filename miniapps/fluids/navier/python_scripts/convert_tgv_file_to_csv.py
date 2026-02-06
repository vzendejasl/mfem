#!/usr/bin/env python3
import sys
import os
import re
import glob
import csv
import argparse

SEP_RE = re.compile(r'^[=\-\s]{5,}$')  # lines made of '=' or '-' (with optional spaces)

def is_separator(line: str) -> bool:
    return bool(SEP_RE.match(line.strip()))

def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def is_data_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    toks = re.split(r'\s+', s)
    # require at least 2 numeric tokens and all tokens numeric
    if len(toks) < 2:
        return False
    return all(is_float_token(t) for t in toks)

def split_header_cells(header_line: str):
    """
    Split a column header line that uses fixed-width spacing.
    Prefer splits on 2+ spaces or tabs so we keep multi-word column names.
    """
    line = header_line.strip()
    # Replace any run of whitespace >=2 or tabs with a single delimiter
    cells = re.split(r'(?:\t+|\s{2,})', line)
    # Trim and drop empties
    return [c.strip() for c in cells if c.strip()]

def snake_case(name: str) -> str:
    s = name.strip()
    # Optional light normalization so names are readable in CSV:
    s = s.replace('µ', 'u')
    s = s.replace('×', 'x')
    s = s.replace('*', 'x')
    # drop parentheses but keep contents
    s = s.replace('(', ' ').replace(')', ' ')
    s = s.replace('/', '_per_')
    # replace everything non-alphanumeric with underscores
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s.lower()

def detect_header_and_data(lines):
    """
    Strategy:
    1) Find a separator line (==== or ----). Header is the next non-empty line.
       Data begins after that header (next non-empty line).
    2) If no separator was found, fall back: locate a line with letters & big gaps
       whose next non-empty line is numeric data.
    """
    n = len(lines)

    # 1) look for explicit separator
    for i, ln in enumerate(lines):
        if is_separator(ln):
            # next non-empty is header
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j >= n:
                break
            header_idx = j

            # data starts after header's next non-empty line
            k = j + 1
            while k < n and not lines[k].strip():
                k += 1
            if k < n:
                return header_idx, k

    # 2) fallback heuristic
    for i in range(n - 1):
        header_line = lines[i]
        next_line = lines[i + 1]
        if re.search(r'[A-Za-z]', header_line) and re.search(r'\s{2,}', header_line):
            if is_data_line(next_line):
                return i, i + 1

    raise ValueError("Could not detect header/data boundary.")

def convert_one_file(input_path, snake=False, verbose=False):
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    header_idx, data_idx = detect_header_and_data(lines)
    headers_raw = split_header_cells(lines[header_idx])

    headers = [snake_case(h) for h in headers_raw] if snake else headers_raw

    # decide output path
    base, ext = os.path.splitext(input_path)
    if ext.lower() == '.txt':
        output_path = base + '.csv'
    else:
        output_path = input_path + '.csv'

    if verbose:
        print(f"[parse] {os.path.basename(input_path)}")
        print(f"        header #{len(headers)}: {headers}")
        print(f"        writing -> {os.path.basename(output_path)}")

    with open(output_path, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(headers)

        for ln in lines[data_idx:]:
            s = ln.strip()
            if not s:
                continue
            # stop if we ever hit another separator (unlikely, but safe)
            if is_separator(s):
                break
            if not is_data_line(s):
                # skip non-numeric junk lines at the end, if any
                continue
            vals = re.split(r'\s+', s)
            # trim or pad to match header count
            if len(vals) >= len(headers):
                row = vals[:len(headers)]
            else:
                row = vals + [''] * (len(headers) - len(vals))
            writer.writerow(row)

    return output_path

def main():
    ap = argparse.ArgumentParser(
        description="Convert TGV-like text outputs (variable headers) to CSV."
    )
    ap.add_argument('patterns', nargs='+',
                    help='Input glob pattern(s), e.g. "tgv_Re*" or "tgv_out_*.txt"')
    ap.add_argument('--snake-case', action='store_true',
                    help='Normalize column names to snake_case.')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='Print what is being parsed/written.')
    args = ap.parse_args()

    # expand all patterns
    inputs = []
    for pat in args.patterns:
        matches = glob.glob(pat)
        if not matches:
            print(f"Warning: pattern matched no files: {pat}", file=sys.stderr)
        inputs.extend(matches)

    if not inputs:
        print("No input files found. Nothing to do.", file=sys.stderr)
        sys.exit(2)

    # process files
    ok = 0
    for path in sorted(set(inputs)):
        try:
            out = convert_one_file(path, snake=args.snake_case, verbose=args.verbose)
            if args.verbose:
                print(f"Done: {path} -> {out}")
            ok += 1
        except Exception as e:
            print(f"Error converting {path}: {e}", file=sys.stderr)

    if ok == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()

