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
                # Attempt to extract Step/Time from a comment
                step_match = re.search(r'Step\s*=\s*(\d+)', line_str)
                if step_match:
                    step_num = int(step_match.group(1))
                time_match = re.search(r'Time\s*=\s*([\d.eE+\-]+)', line_str)
                if time_match:
                    time_val = float(time_match.group(1))
            else:
                # data line: "k  E(k)"
                parts = line_str.split()
                if len(parts) >= 2:
                    k_vals.append(float(parts[0]))
                    E_vals.append(float(parts[1]))
    if k_vals and E_vals:
        return step_num, time_val, np.array(k_vals), np.array(E_vals)
    return step_num, time_val, None, None

def main():
    # Dictionary of directories to compare, each with:
    #   'step': int (the step # in the filename)
    #   'label': str (legend base)
    #   plus any valid matplotlib kwargs like 'color', 'linestyle', etc.
    
    # dir_styles = {
    #     "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/": {
    #         "step": 9000,
    #         "label": "Order2",
    #         "color": "blue",
    #         "linestyle": "-"
    #     },
    #     "/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P3/": {
    #         "step": 9002,
    #         "label": "Order3",
    #         "color": "black",
    #         "linestyle": "-"
    #     },
    #     "/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/": {
    #         "step": 9000,
    #         "label": "Order4",
    #         "color": "green",
    #         "linestyle": "-"
    #     },        
    # }

    dir_styles = {
        # "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/": {
        #     "step": 9000,
        #     "label": "Order2 Cell center sampling",
        #     "color": "blue",
        #     "linestyle": "-"
        # },
        # "/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P3/": {
        #     "step": 9002,
        #     "label": "Order3 Cell center sampling",
        #     "color": "black",
        #     "linestyle": "-"
        # },
        # "/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/": {
        #     "step": 9000,
        #     "label": "Order4 Cell center sampling",
        #     "color": "green",
        #     "linestyle": "-"
        # },        
        "/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_64_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv1P3/": {
            "step": 10001,
            "label": "Order3 64^3",
            "color": "magenta",
            "linestyle": "-"
        },        
        "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/": {
            "step": 10001,
            "label": "Order2",
            "color": "blue",
            "linestyle": "-"
        },
        "/p/lustre1/zendejas/TGV/mfem/Order3_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P3/": {
            "step": 10003,
            "label": "Order3",
            "color": "purple",
            "linestyle": "-"
        },
        "/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/": {
            "step": 10001,
            "label": "Order4",
            "color": "green",
            "linestyle": "-"
        },        
        "/p/lustre1/zendejas/TGV/mfem/sample_data/": {
            "step": 1000,
            "label": "Reference (256^3 FD, Order 4)",
            "color": "black",
            "linestyle": "-"
        },        
    }
 
    dir_styles = {
       # "/p/lustre1/zendejas/TGV/mfem/Order2_Re400/tgv_128_test_sampling/sample_npts_order_1": {
       #     "step": 9002,
       #     "label": "Re 400 Order 1 more sample points",
       #     "color": "blue",
       #     "linestyle": "-"
       # },  
       "/p/lustre1/zendejas/TGV/mfem/Order2_Re400/tgv_128_test_sampling/ElementCentersVelocity_Re400NumPtsPerDir32RefLv2P2/": {
           "step": 9001,
           "label": "Re 400 Order 2",
           "color": "magenta",
           "linestyle": "-"
       },  
       "/p/lustre1/zendejas/TGV/mfem/Order4_Re400/tgv_128_test_sampling/ElementCentersVelocity_Re400NumPtsPerDir32RefLv2P4/": {
           "step": 9001,
           "label": "Re 400 Order 4",
           "color": "blue",
           "linestyle": "-"
       },  
       # "/p/lustre1/zendejas/TGV/mfem/Order2_Re400/tgv_128_test_sampling/sample_npts_order_3": {
       #     "step": 9001,
       #     "label": "Re 400 Order 3 more sample points",
       #     "color": "red",
       #     "linestyle": "-"
       # },  
       # "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/": {
       #     "step": 8501,
       #     "label": "Order 2",
       #     "color": "blue",
       #     "linestyle": "-"
       # },  
       "/p/lustre1/zendejas/TGV/mfem/Order2_Re1600/tgv_128_test_sampling_higher/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P2/": {
           "step": 9002,
           "label": "Order 2",
           "color": "purple",
           "linestyle": "-"
       },  
       "/p/lustre1/zendejas/TGV/mfem/Order4_Re1600/tgv_128_test_sampling/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv2P4/": {
           "step": 9001,
           "label": "Order 4",
           "color": "green",
           "linestyle": "-"
       },  
       # "/p/lustre2/zendejas/TestCases/mfem/TGV/Order2_Re1600/tgv_512/ElementCentersVelocity_Re1600NumPtsPerDir32RefLv4P2": {
       #     "step": 11500,
       #     "label": "512^3",
       #     "color": "magenta",
       #     "linestyle": "-"
       # },          
       "/p/lustre1/zendejas/TGV/mfem/sample_data_2/": {
           "step": 9000,
           "label": "Reference (Spectral)",
           "color": "black",
           "linestyle": "-"
       },        
       # "/p/lustre1/zendejas/TGV/mfem/sample_data/": {
       #     "step": 1000,
       #     "label": "Reference (256^3 FD, Order 4)",
       #     "color": "red",
       #     "linestyle": "-"
       # },        
   }    
    

    plt.figure(figsize=(10, 8))
    all_k = []  # will collect all wavenumbers for plotting the -5/3 line

    for directory, style_info in dir_styles.items():
        step = style_info.get("step", 0)  # fallback if not present
        fname = f"energy_spectrum_step_{step}.txt"
        filepath = os.path.join(directory, fname)
        if not os.path.isfile(filepath):
            print(f"Warning: File '{filepath}' not found. Skipping.")
            continue

        step_num, time_val, k_vals, E_vals = parse_spectrum_file(filepath)
        if k_vals is None or E_vals is None:
            print(f"Warning: No valid data in '{filepath}'. Skipping.")
            continue
        
        # Build a legend label: start with style_info['label']
        legend_label = style_info.get("label", os.path.basename(directory.rstrip('/')))
        if step_num is not None:
            legend_label += f", Step {step_num}"
        if time_val is not None:
            legend_label += f", Time {time_val:.3e}"
        
        # copy the style_info so we don't overwrite step/label keys
        plot_kwargs = {k:v for k,v in style_info.items() if k not in ["step","label"]}
        # set the label in plot_kwargs
        plot_kwargs["label"] = legend_label
        
        plt.loglog(k_vals, E_vals, **plot_kwargs)
        all_k.append(k_vals)

    # Plot a -5/3 slope reference line if we found any wavenumbers
    if all_k:
        merged_k = np.concatenate(all_k)
        positive_k = merged_k[merged_k > 0]
        if positive_k.size > 0:
            kmin, kmax = positive_k.min(), positive_k.max()
            if kmax > kmin:
                ref_k = np.linspace(kmin, kmax, 200)
                E_ref = .1e1  # arbitrary amplitude
                slope_vals = E_ref * (ref_k / kmin)**(-5.0/3.0)
                plt.loglog(ref_k, slope_vals, 'r--', label='k^-5/3 slope')

    plt.xlabel('Wavenumber k')
    plt.ylabel('E(k)')  # or 'k E(k)' if you prefer
    plt.title('Energy Spectra Comparison, TGV Re=1600 t=10')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    #ymax = .11e1  
    #ymin = 1e-5
    #plt.ylim(ymin, ymax)
    #xmax = 256
    #xmin = 1
    #plt.xlim(xmin,xmax)
    plt.show()

if __name__ == "__main__":
    main()
