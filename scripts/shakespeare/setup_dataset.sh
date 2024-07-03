#!/bin/bash

# Function to execute preprocess.sh and copy data
execute_iid_preprocess() {
    echo "Executing iid preprocess..."
    cd leaf/data/shakespeare || exit 1
    ./preprocess.sh -s iid --iu 0.089 --sf 1.0 -k 2000 -t sample -tf 0.8
    cp -r data/* "$1/iid"
    cd - > /dev/null
}

execute_niid_preprocess() {
    echo "Executing niid preprocess..."
    cd leaf/data/shakespeare || exit 1
    ./preprocess.sh -s niid --sf 1.0 -k 2000 -t sample -tf 0.8
    cp -r data/* "$1/niid"
    cd - > /dev/null
}

# Check if path argument is provided
if [ $# -eq 0 ]; then
    echo "No path provided. Using current directory as the path."
    target_path=$(pwd)
else
    target_path="$1"
fi

# Create iid and niid directories if they do not exist
mkdir -p "$target_path/iid"
mkdir -p "$target_path/niid"

# Download and process dataset for iid and niid
if [ ! -d "leaf" ]; then
    git clone https://github.com/TalwalkarLab/leaf.git
fi

# Execute preprocess for iid
execute_iid_preprocess "$target_path"

# Execute preprocess for niid
execute_niid_preprocess "$target_path"

echo "Dataset setup completed."