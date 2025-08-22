#!/usr/bin/env python3
import sys
import re

"""
example usage:
python convert_tgv.py your_input_file.txt output_data.csv
"""

def convert_tgv_to_csv(input_file, output_file):
    """
    Convert TGV output file to CSV format by skipping header lines
    and adding commas between columns.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Skip the first 6 lines (header)
    data_lines = lines[7:]
    
    # Open output CSV file
    with open(output_file, 'w') as f:
        # Write CSV header
        f.write("time,cycle,kinetic_energy,enstrophy\n")
        
        # Process each data line
        for line in data_lines:
            line = line.strip()
            if line:  # Skip empty lines
                # Split on whitespace and join with commas
                values = line.split()
                if len(values) >= 4:  # Make sure we have all 4 columns
                    csv_line = ','.join(values[:4])  # Take first 4 columns
                    f.write(csv_line + '\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_tgv.py input_file.txt output_file.csv")
        print("Example: python convert_tgv.py tgv_out.txt tgv_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        convert_tgv_to_csv(input_file, output_file)
        print(f"Successfully converted {input_file} to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
