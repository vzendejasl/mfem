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
    # dir_styles = [
    #     {
    #         "directory": "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128/SampledDataVelocityP2/",
    #         "step": 9000,
    #         "label": "Sampled Cell Center N = 128, P = 2",
    #         "color": "blue",
    #         "linestyle": "-",
    #         "file_prefix": "energy_spectrum_step_cell_center"
    #     },
    #     {
    #         "directory": "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128/SampledDataVelocityP2/",
    #         "step": 9000,
    #         "label": "Sampled One boundary N = 128, P = 2",
    #         "color": "magenta",
    #         "linestyle": "-",
    #         "file_prefix": "energy_spectrum_step_one_boundary"
    #     },
    #     {
    #         "directory": "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128/SampledDataVelocityP2/",
    #         "step": 9000,
    #         "label": "Sampled Both boundary N = 128, P = 2",
    #         "color": "red",
    #         "linestyle": "-",
    #         "file_prefix": "energy_spectrum_step_both_boundaries"
    #     },
    #     {
    #         "directory":
    #         "/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128/SampledDataVelocityP3/cycle_9002",
    #         "step": 9002,
    #         "label": "Sampled Both boundary N = 128, P = 3",
    #         "color": "magenta",
    #         "linestyle": "-",
    #         "file_prefix": "energy_spectrum_include_both_boundaries_step"
    #     },
    #     {
    #         "directory":
    #         "/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128/SampledDataVelocityP3/cycle_9002",
    #         "step": 9002,
    #         "label": "Sampled One boundary N = 128, P = 3",
    #         "color": "purple",
    #         "linestyle": "-",
    #         "file_prefix": "energy_spectrum_include_one_boundary_step"
    #     },
    #     {
    #         "directory":
    #         "/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128/SampledDataVelocityP3/cycle_9002",
    #         "step": 9002,
    #         "label": "Sampled Element Center N = 128, P = 3",
    #         "color": "blue",
    #         "linestyle": "--",
    #         "file_prefix": "energy_spectrum_element_centers_step"
    #     },
    #     {
    #         "directory":
    #         "/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128/SampledDataVelocityP4/cycle_9000",
    #         "step": 9000,
    #         "label": "Sampled Both boundary N = 128, P = 4",
    #         "color": "magenta",
    #         "linestyle": "-",
    #         "file_prefix": "energy_spectrum_include_both_boundaries_step"
    #     },
    #     {
    #         "directory":
    #         "/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128/SampledDataVelocityP4/cycle_9000",
    #         "step": 9000,
    #         "label": "Sampled One boundary N = 128, P = 4",
    #         "color": "purple",
    #         "linestyle": "-",
    #         "file_prefix": "energy_spectrum_include_one_boundary_step"
    #     },
    #     {
    #         "directory":
    #         "/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128/SampledDataVelocityP4/cycle_9000",
    #         "step": 9000,
    #         "label": "Sampled Element Center N = 128, P = 4",
    #         "color": "blue",
    #         "linestyle": "--",
    #         "file_prefix": "energy_spectrum_element_centers_step"
    #     },
    #     {
    #         "directory": "/p/lustre1/zendejas/TGV/mfem/sample_data_2/",
    #         "step": 9000,
    #         "label": "Reference (Spectral)",
    #         "color": "black",
    #         "linestyle": "-"
    #     },
    # ]

    dir_styles = [
        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP3/cycle_5000",
        #     "step": 5000,
        #     "label": "Effective Res 768^3, N = 192, P = 3 (P+2 sampling -- include both boundaries)",
        #     "color": "blue",
        #     "linestyle": ":",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },
        # {
        #     "directory":
        #     "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP3/cycle_9997",
        #     "step": 9997,
        #     "label": "Effective Res 768^3, N = 192, P = 3 (P+2 sampling -- include both boundaries)",
        #     "color": "blue",
        #     "linestyle": "dashdot",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },

        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP1/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 768^3, N = 384, P = 1 (P+2 sampling -- include both boundaries)",
        #     "color": "magenta",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },


        # {
        #     "directory":"/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_32/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv0P3/cycle_9000/",
        #     "step": 9000,
        #     "label": "Effective Res 128^3, N = 32, P = 3",
        #     "color": "blue",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_unknown_step"
        # },

        # {
        #     "directory":"/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_64/SampledDataVelocityP3/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 256^3, N = 64, P = 3 ",
        #     "color": "purple",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },

        # {
        #     "directory":"/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P3/cycle_9000/",
        #     "step": 9000,
        #     "label": "Effective Res 512^3, N = 128, P = 3",
        #     "color": "magenta",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_unknown_step"
        # },

        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP3/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 768^3, N = 192, P = 3",
        #     "color": "grey",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },


        #{
        #    "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP3/cycle_9000",
        #    "step": 9000,
        #    "label": "Effective Res 768^3, N = 192, P = 3 (Element center sampling)",
        #    "color": "Blue",
        #    "linestyle": "--",
        #    "file_prefix": "energy_spectrum_element_centers_step"
        #},
        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP1/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 768^3, N = 384, P = 3 (Element center sampling)",
        #     "color": "magenta",
        #     "linestyle": "--",
        #     "file_prefix": "energy_spectrum_element_centers_step"
        # },

        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP4/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 768^3, N = 192, P = 3 (P+3 sampling -- include both boundaries)",
        #     "color": "orange",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },
        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_768/SampledDataVelocityP5/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 768^3, N = 128, P = 5 (P+2 sampling -- include both boundaries)",
        #     "color": "purple",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },

        # {
        #     "directory":"/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128/SampledDataVelocityP2/cycle_9000",
        #     "step": 9000,
        #     "label": "Sampled Both boundary N = 128, P = 2",
        #     "color": "red",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },

        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_256/SampledDataVelocityP3/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 256^3, N = 64, P = 3 (P+2 sampling -- include both boundaries)",
        #     "color": "green",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },
        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_256/SampledDataVelocityP3/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 256^3, N = 64, P = 3 (Element center sampling)",
        #     "color": "green",
        #     "linestyle": "--",
        #     "file_prefix": "energy_spectrum_element_centers_step"
        # },
        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_256/SampledDataVelocityP1/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 256^3, N = 128, P = 1 (P+2 sampling -- include both boundaries)",
        #     "color": "purple",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },
        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_256/SampledDataVelocityP1/cycle_9000",
        #     "step": 9000,
        #     "label": "Effective Res 256^3, N = 128, P = 1 (Element center sampling)",
        #     "color": "purple",
        #     "linestyle": "--",
        #     "file_prefix": "energy_spectrum_element_centers_step"
        # },

        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/Order2_Re3200/tgv_256/SampledDataVelocityP2/cycle_9000",
        #     "step": 9000,
        #     "label": "Re 3200 Effective Res 768^3, N = 256, P = 2 (P+2 sampling -- include both boundaries)",
        #     "color": "brown",
        #     "linestyle": "-",
        #     "file_prefix": "energy_spectrum_include_both_boundaries_step"
        # },
        # {
        #     "directory": "/p/lustre1/zendejas/TGV/mfem/Order2_Re3200/tgv_256/SampledDataVelocityP2/cycle_9000",
        #     "step": 9000,
        #     "label": "Re 3200 Effective Res 768^3, N = 256, P = 2 (Element center sampling)",
        #     "color": "brown",
        #     "linestyle": "--",
        #     "file_prefix": "energy_spectrum_element_centers_step"
        # },

        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P7/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 64, P = 7",
            "color": "blue",
            "linestyle": "-",
            "file_prefix": "energy_spectrum_unknown_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P3/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 128, P = 3",
            "color": "magenta",
            "linestyle": "-",
            "file_prefix": "energy_spectrum_unknown_step"
        },
        {
            "directory": "/p/lustre1/zendejas/TGV/mfem/effective_resolution_512/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv3P1/cycle_9000/",
            "step": 9000,
            "label": "Effective Res 512^3, N = 256, P = 1",
            "color": "purple",
            "linestyle": "-",
            "file_prefix": "energy_spectrum_unknown_step"
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

    ymax = 1e1  
    ymin = 1e-10
    plt.ylim(ymin, ymax)
    xmax = np.max(2*kmax)
    xmin = 1.5
    plt.xlim(xmin,xmax)


    plt.show()

if __name__ == "__main__":
    main()
