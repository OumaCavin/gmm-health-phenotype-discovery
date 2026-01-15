#!/usr/bin/env python3
"""Fix indentation error on line 1022 in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix line 1022 - wrong indentation
old_1022 = '            "}\\n",'
new_1022 = '    "}\\n",'

if old_1022 in content:
    content = content.replace(old_1022, new_1022)
    print("Fixed line 1022")
else:
    print("Line 1022 not in expected format")
    # Let's check what we have
    idx = content.find('"}\\n",')
    if idx != -1:
        print(f"Context: {repr(content[idx-30:idx+30])}")

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
