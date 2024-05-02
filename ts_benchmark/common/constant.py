# -*- coding: utf-8 -*-
import os

# Get the root path where the code file is located
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# Build the path to the dataset folder
FORECASTING_DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "forecasting")

# Profile Path
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

# third-party library path
THIRD_PARTY_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "third_party")
