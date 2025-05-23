#!/usr/bin/env python
import sys
import subprocess
from packaging import version

vfile = "primpy/__version__.py"
README = "README.rst"


def run(*args):
    """Run a bash command and return the output in Python."""
    return subprocess.run(args, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


current_version = run("cat", vfile)
current_version = current_version.split("=")[-1].strip().strip("'")
escaped_version = current_version.replace(".", r"\.")
current_version = version.parse(current_version)

if len(sys.argv) > 1:
    update_type = sys.argv[1]
else:
    update_type = "micro"

major = current_version.major
minor = current_version.minor
micro = current_version.micro

if update_type == "micro":
    micro += 1
elif update_type == "minor":
    minor += 1
    micro = 0
elif update_type == "major":
    major += 1
    minor = 0
    micro = 0

new_version = version.parse(f"{major}.{minor}.{micro}")

for f in [vfile, README]:
    if sys.platform == "darwin":  # macOS sed requires empty string for backup
        run("sed", "-i", "", f"s/{escaped_version}/{new_version}/g", f)
    else:
        run("sed", "-i", f"s/{escaped_version}/{new_version}/g", f)

#run("git", "add", vfile, README)
#run("git", "commit", "-m", f"bump version to {new_version}")
