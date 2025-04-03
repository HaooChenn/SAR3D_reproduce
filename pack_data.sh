#!/bin/bash

DATA_ROOT="/mnt/ywchen/data/sar3d-dataset"
OUTPUT_DIR="/mnt/ywchen/data/sar3d-dataset-tar"

for i in {1..10}; do
    folder="${DATA_ROOT}/${i}"
    output="${OUTPUT_DIR}/${i}.tar.gz"
    if [ -d "$folder" ]; then
        echo "Packing folder $folder -> $output"
        tar -czvf "$output" -C "$DATA_ROOT" "$i"
    else
        echo "Warning: Folder $folder does not exist, skipping."
    fi
done

echo "✅ All folders 1-10 have been processed."
