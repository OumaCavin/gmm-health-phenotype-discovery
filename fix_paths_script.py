#!/usr/bin/env python3
"""
Script to fix all paths in the notebook to use consistent path configurations
and remove redundant os.makedirs() calls.
"""

import json
import re

# Read the notebook
with open('GMM_Health_Phenotype_Discovery.ipynb', 'r') as f:
    notebook = json.load(f)

fixes_count = {
    'os.makedirs_removed': 0,
    'paths_fixed': 0,
    'print_statements_fixed': 0
}

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        original_source = source
        
        # Fix 1: Remove redundant os.makedirs calls (standalone ones that duplicate directory creation)
        # Pattern: os.makedirs('output_v2/.../', exist_ok=True) that are not using FIGURES_DIR/MODELS_DIR
        standalone_makedirs_pattern = r"os\.makedirs\(['\"]output_v2[^)]+['\"],\s*exist_ok=True\)"
        matches = re.findall(standalone_makedirs_pattern, source)
        for match in matches:
            source = source.replace(match, '')
            fixes_count['os.makedirs_removed'] += 1
        
        # Fix 2: Fix os.makedirs that should use FIGURES_DIR
        # os.makedirs(os.path.join(OUTPUT_DIR, 'figures', 'plots'), exist_ok=True)
        source = re.sub(
            r"os\.makedirs\(os\.path\.join\(OUTPUT_DIR,\s*'figures',\s*'plots'\),\s*exist_ok=True\)",
            "",
            source
        )
        
        # Fix 3: Fix print statements with hardcoded paths
        # Change print("[OK] ... output_v2/cluster_profiles/")
        # to print(f"[OK] ... {os.path.join(OUTPUT_DIR, 'cluster_profiles')}")
        
        source = re.sub(
            r"output_v2/cluster_profiles/",
            r"os.path.join(OUTPUT_DIR, 'cluster_profiles')",
            source
        )
        
        source = re.sub(
            r"output_v2/predictions/",
            r"os.path.join(OUTPUT_DIR, 'predictions')",
            source
        )
        
        source = re.sub(
            r"output_v2/metrics/",
            r"os.path.join(OUTPUT_DIR, 'metrics')",
            source
        )
        
        # Fix 4: Fix hardcoded file paths in save/plt.savefig calls
        source = re.sub(
            r"'output_v2/figures/plots/([^']+)'",
            r"os.path.join(FIGURES_DIR, 'plots', '\1')",
            source
        )
        
        source = re.sub(
            r"'output_v2/figures/plots/",
            r"os.path.join(FIGURES_DIR, 'plots', ",
            source
        )
        
        source = re.sub(
            r"'output_v2/metrics/([^']+)'",
            r"os.path.join(OUTPUT_DIR, 'metrics', '\1')",
            source
        )
        
        source = re.sub(
            r"'output_v2/metrics/",
            r"os.path.join(OUTPUT_DIR, 'metrics', ",
            source
        )
        
        # Fix 5: Fix plt.savefig with hardcoded paths
        source = re.sub(
            r"plt\.savefig\('output_v2/figures/plots/([^']+)'",
            r"plt.savefig(os.path.join(FIGURES_DIR, 'plots', '\1')",
            source
        )
        
        source = re.sub(
            r"plt\.savefig\('output_v2/metrics/([^']+)'",
            r"plt.savefig(os.path.join(OUTPUT_DIR, 'metrics', '\1')",
            source
        )
        
        # Fix 6: Fix print statements that show file paths
        source = re.sub(
            r"output_v2/figures/plots/06_bic_aic_analysis\.png",
            r"os.path.join(FIGURES_DIR, 'plots', '06_bic_aic_analysis.png')",
            source
        )
        
        source = re.sub(
            r"output_v2/metrics/model_selection_results\.csv",
            r"os.path.join(OUTPUT_DIR, 'metrics', 'model_selection_results.csv')",
            source
        )
        
        if source != original_source:
            fixes_count['paths_fixed'] += 1
            cell['source'] = [source]

print("=" * 60)
print("FIXES APPLIED")
print("=" * 60)
for key, value in fixes_count.items():
    print(f"  {key}: {value}")
print("=" * 60)

# Write the updated notebook
with open('GMM_Health_Phenotype_Discovery.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\nNotebook updated successfully!")
