#!/usr/bin/env python3
"""Fix line 37 and check for other similar issues in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix line 37 - missing closing
old_37 = '"        ]'
new_37 = '"        ]\\n",'

if old_37 in content:
    # Only replace the first occurrence (the broken one)
    content = content.replace(old_37, new_37, 1)
    print("Fixed line 37")

# Also fix line 39 which should be "    })"
old_39 = '","    })'
new_39 = '"    })\\n",'

if old_39 in content:
    content = content.replace(old_39, new_39)
    print("Fixed line 39")
else:
    # Try alternative
    if '","    })' in content:
        content = content.replace('","    })', '"    })\\n",')
        print("Fixed line 39 (alternative)")

# Fix line 40 which should be ""
old_40 = '"",'
new_40 = '"\\n",'

if old_40 in content:
    content = content.replace(old_40, new_40)
    print("Fixed line 40")

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
