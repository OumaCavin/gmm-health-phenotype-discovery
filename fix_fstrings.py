#!/usr/bin/env python3
"""Fix the f-string lines in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# The problematic lines have f" without proper string delimiters
# They should be: "            f"content\n",

# Replace the broken lines with properly formatted ones
# Using a line-by-line approach

lines = content.split('\n')
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if this is one of the problematic f-string lines
    if 'f"Cluster {c}:' in line and 'individuals' in line:
        # This line is broken - fix it
        fixed_lines.append('            "    report_content += f"""\\n",')
        fixed_lines.append('            "Cluster {c}: " + str(len(cluster_subset)) + " individuals (" + str(round(100*len(cluster_subset)/len(data), 1)) + "%)\\n",')
        i += 1
        continue
    elif 'f"  - Mean Age:' in line:
        fixed_lines.append('            "  - Mean Age: " + str(round(cluster_subset["age"].mean(), 1)) + " years\\n",')
        i += 1
        continue
    elif 'f"  - Mean BMI:' in line:
        fixed_lines.append('            "  - Mean BMI: " + str(round(cluster_subset["bmi"].mean(), 1)) + "\\n",')
        i += 1
        continue
    elif 'f"  - Mean Systolic BP:' in line:
        fixed_lines.append('            "  - Mean Systolic BP: " + str(round(cluster_subset["systolic_bp_mmHg"].mean(), 1)) + " mmHg\\n",')
        i += 1
        continue
    elif 'f"  - Mean Glucose:' in line:
        fixed_lines.append('            "  - Mean Glucose: " + str(round(cluster_subset["fasting_glucose_mg_dL"].mean(), 1)) + " mg/dL\\n",')
        i += 1
        continue
    elif 'f"  - Mean PHQ-9:' in line:
        fixed_lines.append('            "  - Mean PHQ-9: " + str(round(cluster_subset["phq9_total_score"].mean(), 1)) + "\\n",')
        i += 1
        continue
    else:
        fixed_lines.append(line)
        i += 1

content = '\n'.join(fixed_lines)

# Write back
with open('/workspace/add_phases_13_20.py', 'w') as f:
    f.write(content)

print("Applied fixes")

# Verify syntax
import subprocess
result = subprocess.run(['python3', '-m', 'py_compile', '/workspace/add_phases_13_20.py'],
                       capture_output=True, text=True)
if result.returncode == 0:
    print("SUCCESS: Python syntax is valid!")
else:
    print(f"ERROR: {result.stderr}")
