#!/usr/bin/env python3
"""Fix the remaining syntax errors in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix the broken line 989 - it lost its opening quote
broken_989 = 'notebook["cells"].insert(insert_index, cell)",'
correct_989 = '"        notebook["cells"].insert(insert_index, cell),'

if broken_989 in content:
    content = content.replace(broken_989, correct_989)
    print("Fixed line 989")
else:
    print("Line 989 not in expected format")
    # Let's check what we actually have
    idx = content.find('notebook["cells"]')
    if idx != -1:
        print(f"Found at: {repr(content[idx:idx+60])}")

# Also fix line 990 if it has issues
broken_990 = 'insert_index += 1\\n    ",'
correct_990 = '        insert_index += 1\\n",'

if broken_990 in content:
    content = content.replace(broken_990, correct_990)
    print("Fixed line 990")
else:
    print("Line 990 not in expected format")

# Fix line 991
broken_991 = '\\n",'
# This might be correct already, leave it for now

# Write back
with open('/workspace/add_phases_13_20.py', 'w') as f:
    f.write(content)

# Verify syntax
import subprocess
result = subprocess.run(['python3', '-m', 'py_compile', '/workspace/add_phases_13_20.py'],
                       capture_output=True, text=True)
if result.returncode == 0:
    print("\nSUCCESS: Python syntax is valid!")
else:
    print(f"\nERROR: {result.stderr}")
