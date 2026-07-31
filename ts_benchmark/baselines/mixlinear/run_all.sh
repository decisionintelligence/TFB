#!/bin/bash

# Run all MixLinear test scripts in parallel.

SCRIPT_DIR="/root/ljk/benchmark/TFB/ts_benchmark/baselines/mixlinear/test_scripts"

if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: directory $SCRIPT_DIR does not exist"
    exit 1
fi

for script in "$SCRIPT_DIR"/*.sh; do
    if [ -f "$script" ]; then
        echo "Running: $script"
        bash "$script" &
    fi
done

wait
echo "All MixLinear scripts finished."
