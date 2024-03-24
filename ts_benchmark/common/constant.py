# -*- coding: utf-8 -*-
import os

# Get the root path where the code file is located
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# Path to build metadata file
META_FORECAST_DATA_PATH = os.path.join(
    ROOT_PATH, "dataset", "forecasting", "FORECAST_META.csv"
)

META_DETECTION_DATA_PATH = os.path.join(
    ROOT_PATH, "dataset", "anomaly_detect", "DETECT_META.csv"
)

# Build the path to the dataset folder
FORECASTING_DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "forecasting")

ANOMALY_DETECT_DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "anomaly_detect")


# Profile Path
CONFIG_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "config")
