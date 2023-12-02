#!/bin/bash
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# Script to search for missing init FILES.
set -euxo pipefail

FILES=$( find ./skchange -type d '!' -exec test -e "{}/__init__.py" ";" -not -path "**/__pycache__" -not -path "**/datasets/data*" -not -path "**/contrib/*" -print )

if [[ -n "$FILES" ]]
then
    echo "Missing __init__.py files detected in the following modules:"
    echo "$FILES"
    exit 1
fi
