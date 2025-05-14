#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_spectrum_file(filepath):
    """
    Reads a spectrum file of the form:
      # (possibly a comment with 'Step=XXXX, Time=YYYY')
      k1  E1
      k2  E2
      ...
    Returns: step_num, time_val, k_vals, E_vals
             (some may be None if not found)
    """
    step_num = None
    time_val = None
    k_vals = []
    E_vals = []
    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith('#'):
                step_match = re.search(r'Step\s*=\s*(\d+)', line_str)
                if step_match:
                    step_num = int(step_match.group(1))
                time_match = re.search(r'Time\s*=\s*([\d.eE+\-]+)', line_str)
                if time_match:
                    time_val = float(time_match.group(1))
            else:
                parts = line_str.split()
                if len(parts) >= 2:
                    k_vals.append(float(parts[0]))
                    E_vals.append(float(parts[1]))
    if k_vals and E_vals:
        return step_num, time_val, np.array(k_vals), np.array(E_vals)
    return step_num, time_val, None, None

def main():

    dir_styles = [
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/SampledDatavelocityP1/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 512, P = 1",
            "color": "blue",
            "linestyle": "-",
            "file_prefix": "energy_spectrum_include_one_boundary_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/SampledDatavelocityP2/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 256, P = 2",
            "color": "purple",
            "linestyle": "-",
            "file_prefix": "energy_spectrum_include_one_boundary_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/SampledDatavelocityP2/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 256, P = 2 (Over Sample)",
            "color": "purple",
            "linestyle": "--",
            "file_prefix": "energy_spectrum_include_both_boundaries_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/SampledDatavelocityP4/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 128, P = 4",
            "color": "green",
            "linestyle": "-",
            "file_prefix": "energy_spectrum_include_one_boundary_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/SampledDatavelocityP4/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 128, P = 4 (Over Sample)",
            "color": "green",
            "linestyle": "--",
            "file_prefix": "energy_spectrum_include_both_boundaries_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/SampledDatavelocityP8/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 64, P = 8",
            "color": "magenta",
            "linestyle": "-",
            "file_prefix": "energy_spectrum_include_one_boundary_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/SampledDatavelocityP8/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 64, P = 8 (Over Sample)",
            "color": "magenta",
            "linestyle": "--",
            "file_prefix": "energy_spectrum_include_both_boundaries_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/sample_data_2/",
            "step": 9000,
            "label": " Re=1600 Reference (Spectral)",
            "color": "black",
            "linestyle": "-"
        },

        ]

    plt.figure(figsize=(10, 8))
    all_k = []

    for style_info in dir_styles:
        directory = style_info["directory"]
        if "filename" in style_info:
            fname = style_info["filename"]
        else:
            file_prefix = style_info.get("file_prefix", "energy_spectrum_step")
            step = style_info.get("step", 0)
            fname = f"{file_prefix}_{step}.txt"
        
        filepath = os.path.join(directory, fname)
        if not os.path.isfile(filepath):
            print(f"Warning: File '{filepath}' not found. Skipping.")
            continue

        step_num, time_val, k_vals, E_vals = parse_spectrum_file(filepath)
        if k_vals is None or E_vals is None:
            print(f"Warning: No valid data in '{filepath}'. Skipping.")
            continue
        
        legend_label = style_info.get("label", os.path.basename(directory.rstrip('/')))
        if step_num is not None:
            legend_label += f", Step {step_num}"
        if time_val is not None:
            legend_label += f", Time {time_val:.3e}"
        
        plot_kwargs = {k: v for k, v in style_info.items() if k not in ["directory", "step", "label", "filename", "file_prefix"]}
        plot_kwargs["label"] = legend_label
        
        plt.loglog(k_vals, E_vals, **plot_kwargs)
        all_k.append(k_vals)

    if all_k:
        merged_k = np.concatenate(all_k)
        positive_k = merged_k[merged_k > 0]
        if positive_k.size > 0:
            kmin, kmax = positive_k.min(), positive_k.max()
            if kmax > kmin:
                ref_k = np.linspace(kmin, kmax, 200)
                E_ref = 0.1e1
                slope_vals = E_ref * (ref_k / kmin)**(-5.0/3.0)
                plt.loglog(ref_k, slope_vals, 'r--', label='k^-5/3 slope')

    plt.xlabel('Wavenumber k')
    plt.ylabel('E(k)')
    plt.title('Energy Spectra Comparison, t=9')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    ymax = 1e-1  
    ymin = 1e-10
    plt.ylim(ymin, ymax)
    xmax = np.max(2*kmax)
    xmin = 1.5
    plt.xlim(xmin,xmax)


    plt.show()

if __name__ == "__main__":
    main()
