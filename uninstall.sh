#! /bin/bash

# Deactivate Conda environment
conda deactivate

# Remove Conda environments
echo "Removing Conda environments py2k and py3k..."
conda env remove -n py2k
conda env remove -n py3k

# Remove REAPER directory
echo "Removing REAPER directory..."
rm -rf REAPER

# Manual intervention needed for PATH removal and uninstallation of git and cmake
echo "To fully uninstall, please manually remove the REAPER path from your PATH environment variable."
echo "Also, if you want to uninstall git and cmake, please do so carefully as it may affect other system dependencies."
