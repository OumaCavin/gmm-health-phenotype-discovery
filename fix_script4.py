#!/usr/bin/env python3
"""Fix remaining syntax errors in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix line 995
old_995 = '    \\n",'
new_995 = '    '

if old_995 in content:
    content = content.replace(old_995, new_995)
    print("Fixed line 995")
else:
    print("Line 995 not in expected format")

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
