#!/usr/bin/env python3
"""Fix line 39 in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix line 39 - malformed
old_39 = '",    '
new_39 = '"\\n",'

if old_39 in content:
    content = content.replace(old_39, new_39)
    print("Fixed line 39")
else:
    print("Line 39 not in expected format")
    # Check what we have
    idx = content.find(',    ')
    if idx != -1:
        print(f"Context: {repr(content[idx-20:idx+20])}")

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
