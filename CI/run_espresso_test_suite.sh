#!/bin/sh
ROOT_DIR=$(dirname "$(realpath "$0")")
ctest --output-on-failure --test-dir "${ROOT_DIR}/espresso_tests" --timeout 300 ${*}
