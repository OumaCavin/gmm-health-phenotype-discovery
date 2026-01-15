#!/usr/bin/env python3
"""Fix the quoting issues in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# The problematic section has incorrect quoting
# Line 988 should start with a quote
# Line 989 has unescaped inner quotes

# Find the exact broken section
old_section = '''    # Insert all new cells before References
    for cell in new_cells:
            "        notebook["cells"].insert(insert_index, cell),
                insert_index += 1\\n",
    \\n",
    # Write updated notebook\\n",
    "with open(output_path, 'w', encoding='utf-8') as f:\\n",
        "    json.dump(notebook, f, indent=1)\\n    ",'''

new_section = '''    # Insert all new cells before References
    for cell in new_cells:
        notebook["cells"].insert(insert_index, cell)
        insert_index += 1
    
    # Write updated notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)'''

if old_section in content:
    content = content.replace(old_section, new_section)
    print("SUCCESS: Fixed the section with proper Python code")
else:
    print("ERROR: Could not find the section to fix")
    # Let's check what we have
    idx = content.find('# Insert all new cells')
    if idx != -1:
        print(f"Context: {repr(content[idx:idx+500])}")

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
