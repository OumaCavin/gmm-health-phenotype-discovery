#!/usr/bin/env python3
"""Fix the syntax errors in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix the escaped sequences in the for loop section
content = content.replace('for cell in new_cells:\\n",', 'for cell in new_cells:')
content = content.replace("notebook['cells'].insert(insert_index, cell)\\n", '    notebook["cells"].insert(insert_index, cell)')
content = content.replace('insert_index += 1\\n",', '    insert_index += 1')

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
