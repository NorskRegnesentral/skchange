#!/bin/bash

# Helper script to download and install skchange from test PyPI to check wheel
# and upload prior to new release

set -e

# Version to test, passed as input argument to script
VERSION=$1

# Make temporary directory
echo "Making test directory ..."
mkdir "$HOME"/testdir
cd "$HOME"/testdir

# Create test environment
echo "Creating test environment ..."

# shellcheck disable=SC1091
python -m venv .venv
# shellcheck disable=SC1091
source .venv/Scripts/activate

# Install from test PyPI
echo "Installing skchange from Test PyPI ..."
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple skchange=="$VERSION"
echo "Successfully installed skchange from Test PyPI."

# Clean up test directory and environment
echo "Cleaning up ..."
deactivate
rm -rf "$HOME"/testdir

echo "Done."
