#!/usr/bin/env python3
"""Fix all malformed lines in add_phases_13_20.py"""

with open('/workspace/add_phases_13_20.py', 'r') as f:
    content = f.read()

# Fix lines that have just "    " (incomplete strings)
# These should be "    \n",

fixes = 0
# Pattern: line with just whitespace inside quotes followed by newline without closing
# Match patterns like: "    $" (where $ is end of line in the cat -A output)
# In actual file content, this looks like: "    \n"

# Let's find and fix all occurrences of malformed lines
# The pattern is: "    " followed immediately by newline character in the actual file

# Replace all instances where we have "    ", followed by ", (end of line string)
# This needs to become "    \n",

# Actually, let's look at the actual bytes
lines = content.split('\n')
for i, line in enumerate(lines):
    # Check if this line is a malformed string (just spaces with no content)
    if line.strip() == '"' or (line.startswith('"') and line.endswith('"') and len(line) <= 10):
        print(f"Line {i+1}: {repr(line)}")
        # This is a malformed line
        if line.strip() == '"':
            lines[i] = '"    \\n",'
            fixes += 1
            print(f"  Fixed to: {repr(lines[i])}")

if fixes > 0:
    content = '\n'.join(lines)
    with open('/workspace/add_phases_13_20.py', 'w') as f:
        f.write(content)
    print(f"\nApplied {fixes} fixes")
else:
    print("No malformed lines found")

# Verify syntax
import subprocess
result = subprocess.run(['python3', '-m', 'py_compile', '/workspace/add_phases_13_20.py'],
                       capture_output=True, text=True)
if result.returncode == 0:
    print("SUCCESS: Python syntax is valid!")
else:
    print(f"ERROR: {result.stderr}")
