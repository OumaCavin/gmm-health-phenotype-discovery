#!/usr/bin/env python3
"""Comprehensive fix for add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix line 1036 - should be a string
old_1036 = '        ]\\n",'
new_1036 = '"        ]\\n",'

if old_1036 in content:
    content = content.replace(old_1036, new_1036)
    print("Fixed line 1036")
else:
    # Maybe it's already been partially fixed
    if '        ]\n' in content:
        content = content.replace('        ]\n', '"        ]\n",')
        print("Fixed line 1036 (alternative)")

# Fix line 1037 - should be a string
old_1037 = '    })\\n",'
new_1037 = '"    })\\n",'

if old_1037 in content:
    content = content.replace(old_1037, new_1037)
    print("Fixed line 1037")
else:
    if '    })\n' in content:
        content = content.replace('    })\n', '"    })\n",')
        print("Fixed line 1037 (alternative)")

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
