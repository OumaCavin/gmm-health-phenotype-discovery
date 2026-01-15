#!/usr/bin/env python3
"""Fix the f-string lines in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# The problematic lines 828-833 need to be fixed
# They should be regular strings (not f-strings) with proper string concatenation
# because we can't nest f-strings inside the f""" started on line 827

# Replace the broken section
old_section = '''    report_content += f"""\\n",
f"Cluster {c}: {len(cluster_subset)} individuals ({100*len(cluster_subset)/len(data):.1f}%)\\n",
f"  - Mean Age: {cluster_subset['age'].mean():.1f} years\\n",
f"  - Mean BMI: {cluster_subset['bmi'].mean():.1f}\\n",
f"  - Mean Systolic BP: {cluster_subset['systolic_bp_mmHg'].mean():.1f} mmHg\\n",
f"  - Mean Glucose: {cluster_subset['fasting_glucose_mg_dL'].mean():.1f} mg/dL\\n",
f"  - Mean PHQ-9: {cluster_subset['phq9_total_score'].mean():.1f}\\n",
"""'''

new_section = '''    report_content += f"""\\n",
            "Cluster {c}: \\n",
            f"  {len(cluster_subset)} individuals ({100*len(cluster_subset)/len(data):.1f}%)\\n",
            "  - Mean Age: \\n",
            f"  {cluster_subset['age'].mean():.1f} years\\n",
            "  - Mean BMI: \\n",
            f"  {cluster_subset['bmi'].mean():.1f}\\n",
            "  - Mean Systolic BP: \\n",
            f"  {cluster_subset['systolic_bp_mmHg'].mean():.1f} mmHg\\n",
            "  - Mean Glucose: \\n",
            f"  {cluster_subset['fasting_glucose_mg_dL'].mean():.1f} mg/dL\\n",
            "  - Mean PHQ-9: \\n",
            f"  {cluster_subset['phq9_total_score'].mean():.1f}\\n",
            """'''

if old_section in content:
    content = content.replace(old_section, new_section)
    print("SUCCESS: Fixed the section")
else:
    print("ERROR: Could not find the section")
    # Try alternative approach - fix line by line
    lines = content.split('\n')
    for i in range(len(lines)):
        if 'f"Cluster {c}:' in lines[i] and 'individuals' in lines[i]:
            lines[i] = '            "    report_content += f"""\\n",'
            lines[i+1] = '            f"Cluster {{c}}: {{len(cluster_subset)}} individuals ({{100*len(cluster_subset)/len(data):.1f}}%)\\n",'
            lines[i+2] = '            f"  - Mean Age: {{cluster_subset[\\'age\\'].mean():.1f}} years\\n",'
            lines[i+3] = '            f"  - Mean BMI: {{cluster_subset[\\'bmi\\'].mean():.1f}}\\n",'
            lines[i+4] = '            f"  - Mean Systolic BP: {{cluster_subset[\\'systolic_bp_mmHg\\'].mean():.1f}} mmHg\\n",'
            lines[i+5] = '            f"  - Mean Glucose: {{cluster_subset[\\'fasting_glucose_mg_dL\\'].mean():.1f}} mg/dL\\n",'
            lines[i+6] = '            f"  - Mean PHQ-9: {{cluster_subset[\\'phq9_total_score\\'].mean():.1f}}\\n",'
            content = '\n'.join(lines)
            print("Applied alternative fix")
            break

# Write back
with open('/workspace/add_phases_13_20.py', 'w') as f:
    f.write(content)

# Verify syntax
import subprocess
result = subprocess.run(['python3', '-m', 'py_compile', '/workspace/add_phases_13_20.py'],
                       capture_output=True, text=True)
if result.returncode == 0:
    print("SUCCESS: Python syntax is valid!")
else:
    print(f"ERROR: {result.stderr}")
