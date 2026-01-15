#!/usr/bin/env python3
"""Fix line 326 in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix line 326 - it has malformed string
old_326 = '"    ",'
new_326 = '"    \\n",'

if old_326 in content:
    content = content.replace(old_326, new_326, 1)  # Only replace first occurrence
    print("Fixed line 326")
else:
    print("Line 326 not in expected format")
    # Check what we have
    lines = content.split('\n')
    for i, line in enumerate(lines[320:330], start=321):
        if 'N = len' in line:
            print(f"Line {i}: {repr(line)}")

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
