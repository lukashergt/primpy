#!/usr/bin/env python
import sys
import subprocess
from packaging import version


def run(*args):
    """Run a bash command and return the output in Python."""
    return subprocess.run(args, 
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def unit_incremented(a, b):
    """Check if a is one version larger than b."""
    a = version.parse(a)
    b = version.parse(b)
    if a.pre is not None and b.pre is not None:
        if a.pre[0] == b.pre[0]:
            return a.pre[1] == b.pre[1]+1 and a.base_version == b.base_version
        else:
            return (a.pre[1] == 0 and a.pre[0] > b.pre[0]
                    and a.base_version == b.base_version)
    elif a.pre is not None:
        return a.base_version > b.base_version and a.pre[1] == 0
    elif b.pre is not None:
        return a.base_version == b.base_version
    else:
        return (a.micro == b.micro+1 and
                a.minor == b.minor and
                a.major == b.major or
                a.micro == 0 and
                a.minor == b.minor+1 and
                a.major == b.major or
                a.micro == 0 and
                a.minor == 0 and
                a.major == b.major+1)


README = "README.rst"
vfile = "primpy/__version__.py"
current_version = run("cat", vfile)
current_version = current_version.split("=")[-1].strip().strip("'")

run("git", "fetch", "origin", "master")
previous_version = run("git", "show", "remotes/origin/master:" + vfile)
previous_version = previous_version.split("=")[-1].strip().strip("'")

readme_version = run("grep", ":Version:", README)
readme_version = readme_version.split(":")[-1].strip()

if version.parse(current_version) != version.parse(readme_version):
    sys.stderr.write("Version mismatch: {} != {}".format(vfile, README))
    sys.exit(1)
elif not unit_incremented(current_version, previous_version):
    sys.stderr.write(("Version must be incremented by one:\n"
                      "HEAD:   {},\n"
                      "master: {}.\n"
                      ).format(current_version, previous_version))
    sys.exit(1)

