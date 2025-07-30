#!/bin/bash

# ==============================================================================
#  Asset Packaging Script for SAR3D
#  This script finds all generated asset folders in the ./eval directory
#  and packages them into a single timestamped ZIP file.
# ==============================================================================

# --- Configuration ---
# Directory where the generated assets are saved.
OUTPUT_DIR="./eval"
# Base name for the final ZIP file.
ZIP_BASENAME="generated_3d_assets"

# --- Script Logic ---

echo "Starting asset packaging process..."

# 1. Check if the output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' not found."
    echo "Please run the generation script (test.py) first to create the assets."
    exit 1
fi

# 2. Navigate into the output directory to get clean paths in the zip
cd "$OUTPUT_DIR" || exit

# 3. Check if there are any folders to package
# `ls -d */` lists all directories. We check if the output is empty.
if [ -z "$(ls -d */ 2>/dev/null)" ]; then
    echo "Warning: No asset folders found in '$OUTPUT_DIR' to package."
    cd ..
    exit 0
fi

# 4. Create a unique, timestamped filename for the ZIP archive
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_FILENAME="../${ZIP_BASENAME}_${TIMESTAMP}.zip"

# 5. List the folders that will be included for user confirmation
echo "--------------------------------------------------"
echo "The following asset folders will be packaged:"
ls -d */
echo "--------------------------------------------------"

# 6. Create the zip archive
echo "Creating ZIP file: ${ZIP_FILENAME}"
# The `*/` glob expands to all directories in the current folder.
# We use `zip -r` for recursive zipping.
zip -r "${ZIP_FILENAME}" */

# 7. Navigate back to the project root
cd ..

# 8. Final confirmation message
echo ""
echo "Packaging complete!"
echo "Your assets have been successfully packaged into:"
echo "${ZIP_FILENAME}"
echo "--------------------------------------------------"

