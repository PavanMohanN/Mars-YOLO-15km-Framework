"""
plot_csfd.py

Generates publication-grade Crater Size-Frequency Distribution (CSFD) 
log-log plots directly from the Refined Geomorphological Catalog.
Forces all outputs to 350 DPI with Font Size 25 for manuscript consistency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================================
# 1. THE PUBLICATION OVERRIDE
# =====================================================================
plt.rcParams.update({
    'font.size': 25,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'legend.title_fontsize': 25,
    'figure.titlesize': 25,
    'font.family': 'serif',
    'figure.figsize': (20, 16),      # Large canvas to accommodate Size 25 fonts
    'figure.autolayout': True        # Prevents label clipping
})

# =====================================================================
# 2. PATHS & REGIONAL CONSTANTS
# =====================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CATALOG_PATH = os.path.join(BASE_DIR, 'catalogs', 'Refined_Mars_Crater_Catalog.csv')

# Output directly to the validation_metrics folder for centralized manuscript figures
OUTPUT_DIR = os.path.join(BASE_DIR, 'validation_metrics')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Official Regional Areas (km^2) from Table 3 of the manuscript
AREAS = {
    'Elysium': 1371608,
    'Noachis': 6304280,
    'Arabia':  7733345,
    'Melas':   96482
}

def get_csfd_data(diams, area):
    """
    Calculates the cumulative frequency N(D >= d) normalized by area.
    """
    d_sorted = np.sort(diams)[::-1]
    n_cumulative = np.arange(1, len(d_sorted) + 1) / area
    return d_sorted, n_cumulative

def plot_catalog_csfd():
    print("[*] Initializing CSFD Plotting Engine...")
    
    if not os.path.exists(CATALOG_PATH):
        print(f"[-] Error: Refined Catalog not found at {CATALOG_PATH}")
        print("[*] Please ensure the catalog is generated and placed in the 'catalogs/' directory.")
        return

    print(f"[*] Reading geological data from: {os.path.basename(CATALOG_PATH)}")
    df = pd.read_csv(CATALOG_PATH)
    
    fig, axes = plt.subplots(2, 2)
    # Adjust spacing to ensure large axis labels do not collide
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    
    regions = ['Arabia', 'Elysium', 'Melas', 'Noachis']
    
    for idx, region in enumerate(regions):
        ax = axes.flatten()[idx]
        region_df = df[df['Province'] == region]
        
        if len(region_df) == 0: 
            print(f"[-] No data found for region: {region}. Skipping plot.")
            continue
            
        diams = region_df['Diameter_km'].values
        d_val, n_val = get_csfd_data(diams, AREAS[region])
        
        # Plot YOLOv8m Data
        ax.loglog(d_val, n_val, label=f'YOLOv8m', color='#D62728', linewidth=3.5)
        
        # Formatting
        ax.set_title(f"{region} CSFD", pad=20)
        ax.set_xlabel("Diameter D (km)", labelpad=15)
        ax.set_ylabel(r"$N(D \geq d) / km^2$", labelpad=15)
        ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=1.5)
        ax.legend()
        
    save_path = os.path.join(OUTPUT_DIR, "Refined_CSFD_Distributions.png")
    
    print("[*] Enforcing 350 DPI High-Resolution Export...")
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Success! Publication-grade CSFD plot saved to: {save_path}")

if __name__ == "__main__":
    print("==================================================")
    print(" Mars-YOLO-15km: CSFD Generation Tool ")
    print("==================================================\n")
    plot_catalog_csfd()