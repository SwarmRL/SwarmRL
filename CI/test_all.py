# espresso does not allow multiple instances of a system. therefore we need to spawn the tests one by one manually

import pathlib
import os

test_root_dir = pathlib.Path(__file__).resolve().parent
test_files = sorted(test_root_dir.glob("./**/test_*.py"))
# remove self to prevent infinite recursion
test_files.remove(pathlib.Path(__file__).resolve())

failed = []
total = len(test_files)
for test_file in test_files:
    print(f"running {test_file}")
    exit_cmd = os.system(f"python3 {test_file}")
    if exit_cmd != 0:
        failed.append(test_file)

if len(failed) == 0:
    print("all tests successful")
    exit(0)
else:
    print(f"{len(failed)} tests falied:")
    for f in failed:
        print(f)
    exit(1)
