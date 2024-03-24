# -*- coding: utf-8 -*-
import concurrent.futures
import os
from scipy.signal import argrelextrema
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stl._stl import STL
import time

import warnings

from ts_benchmark.utils.data_processing import read_data

warnings.filterwarnings("ignore")

default_periods = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]


def adjust_period(period_value):
    if abs(period_value - 4) <= 1:
        period_value = 4
    if abs(period_value - 7) <= 1:
        period_value = 7
    if abs(period_value - 12) <= 2:
        period_value = 12
    if abs(period_value - 24) <= 3:
        period_value = 24
    if abs(period_value - 48) <= 1 or (
        (48 - period_value) <= 4 and (48 - period_value) >= 0
    ):
        period_value = 48
    if abs(period_value - 52) <= 2:
        period_value = 52
    if abs(period_value - 96) <= 10:
        period_value = 96
    if abs(period_value - 144) <= 10:
        period_value = 144
    if abs(period_value - 168) <= 10:
        period_value = 168
    if abs(period_value - 336) <= 50:
        period_value = 336
    if abs(period_value - 672) <= 20:
        period_value = 672
    if abs(period_value - 720) <= 20:
        period_value = 720
    if abs(period_value - 1008) <= 100:
        period_value = 1008
    if abs(period_value - 1440) <= 200:
        period_value = 1440
    if abs(period_value - 8766) <= 500:
        period_value = 8766
    if abs(period_value - 10080) <= 500:
        period_value = 10080
    if abs(period_value - 21600) <= 2000:
        period_value = 21600
    if abs(period_value - 43200) <= 2000:
        period_value = 43200
    return period_value


def fftTransfer(timeseries, fmin=0.2):
    yf = abs(np.fft.fft(timeseries))
    yfnormlize = yf / len(timeseries)
    yfhalf = yfnormlize[: len(timeseries) // 2] * 2

    fwbest = yfhalf[argrelextrema(yfhalf, np.greater)]
    xwbest = argrelextrema(yfhalf, np.greater)

    fwbest = fwbest[fwbest >= fmin].copy()

    return len(timeseries) / xwbest[0][: len(fwbest)], fwbest


def count_inversions(series):
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, inversions_left = merge_sort(arr[:mid])
        right, inversions_right = merge_sort(arr[mid:])

        merged = []
        inversions = inversions_left + inversions_right

        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inversions += len(left) - i

        merged.extend(left[i:])
        merged.extend(right[j:])

        return merged, inversions

    series_values = series.tolist()
    _, inversions_count = merge_sort(series_values)

    return inversions_count


def count_peaks_and_valleys(sequence):
    peaks = 0
    valleys = 0

    for i in range(1, len(sequence) - 1):
        if sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
            peaks += 1
        elif sequence[i] < sequence[i - 1] and sequence[i] < sequence[i + 1]:
            valleys += 1

    return peaks + valleys


def count_series(sequence, threshold):
    if len(sequence) == 0:
        return 0

    positive_series = 0
    negative_series = 0

    current_class = None

    for value in sequence:
        if value > threshold:
            if current_class == "negative":
                negative_series += 1
            current_class = "positive"
        else:
            if current_class == "positive":
                positive_series += 1
            current_class = "negative"

    if current_class == "positive":
        positive_series += 1
    elif current_class == "negative":
        negative_series += 1

    return positive_series + negative_series


def extract_other_features(series_value):
    skewness = skew(series_value)

    kurt = kurtosis(series_value)

    rsd = abs((np.std(series_value) / np.mean(series_value)) * 100)

    std_of_first_derivative = np.std(np.diff(series_value))

    inversions = count_inversions(series_value) / len(series_value)

    turning_points = count_peaks_and_valleys(series_value) / len(series_value)

    series_in_series = count_series(series_value, np.median(series_value)) / len(
        series_value
    )

    return [
        skewness,
        kurt,
        rsd,
        std_of_first_derivative,
        inversions,
        turning_points,
        series_in_series,
    ]


def feature_extract(path):
    index_columns = [
        "file_name",
        "length",
        "period_value1",
        "seasonal_strength1",
        "trend_strength1",
        "period_value2",
        "seasonal_strength2",
        "trend_strength2",
        "period_value3",
        "seasonal_strength3",
        "trend_strength3",
        "if_season",
        "if_trend",
        "ADF:p-value",
        "KPSS:p-value",
        "stability",
        "skewness",
        "kurt",
        "rsd",
        "std_of_first_derivative",
        "inversions",
        "turning_points",
        "series_in_series",
    ]
    result_frame = pd.DataFrame(columns=index_columns)

    file_name = path.split("/")[-1]
    file_name = [file_name]

    original_df = read_data(path)
    limited_length_df = original_df

    series_length = [original_df.shape[0]]
    try:
        ADF_P_value = [adfuller(limited_length_df.iloc[:, 0].values, autolag="AIC")[1]]

        KPSS_P_value = [kpss(limited_length_df.iloc[:, 0].values, regression="c")[1]]

        stability = [ADF_P_value[0] <= 0.05 or KPSS_P_value[0] >= 0.05]

    except:
        ADF_P_value = [None]
        KPSS_P_value = [None]
        stability = [None]

    series_value = limited_length_df.iloc[:, 0]
    origin_series_value = original_df.iloc[:, 0]
    series_value = pd.Series(series_value).astype("float")
    origin_series_value = pd.Series(origin_series_value).astype("float")
    other_features = extract_other_features(origin_series_value)
    periods, amplitude = fftTransfer(series_value, fmin=0)

    periods_list = []
    for index_j in range(len(amplitude)):
        periods_list.append(
            round(
                periods[
                    amplitude.tolist().index(sorted(amplitude, reverse=True)[index_j])
                ]
            )
        )

    final_periods1 = []
    for l1 in periods_list:
        l1 = adjust_period(l1)
        if l1 not in final_periods1 and l1 >= 4:
            final_periods1.append(l1)
    periods_num = min(len(final_periods1), 3)
    new_final_periods = final_periods1[:periods_num]
    new_final_periods = new_final_periods + default_periods

    final_periods = []
    for l1 in new_final_periods:
        if l1 not in final_periods and l1 >= 4:
            final_periods.append(l1)

    yuzhi = int(series_length[0] / 3)
    if yuzhi <= 12:
        yuzhi = 12

    season_dict = {}
    for index_period in range(max(13, len(final_periods))):
        period_value = final_periods[index_period]

        if period_value < yuzhi:
            res = STL(limited_length_df.iloc[:, 0], period=period_value).fit()
            limited_length_df["trend"] = res.trend
            limited_length_df["seasonal"] = res.seasonal
            limited_length_df["resid"] = res.resid
            limited_length_df["detrend"] = (
                limited_length_df.iloc[:, 0] - limited_length_df.trend
            )
            limited_length_df["deseasonal"] = (
                limited_length_df.iloc[:, 0] - limited_length_df.seasonal
            )
            trend_strength = max(
                0,
                1 - limited_length_df.resid.var() / limited_length_df.deseasonal.var(),
            )
            seasonal_strength = max(
                0, 1 - limited_length_df.resid.var() / limited_length_df.detrend.var()
            )
            season_dict[seasonal_strength] = [
                period_value,
                seasonal_strength,
                trend_strength,
            ]

    if len(season_dict) < 3:
        for i in range(3 - len(season_dict)):
            season_dict[0.1 * (i + 1)] = [0, -1, -1]

    season_dict = sorted(season_dict.items(), key=lambda x: x[0], reverse=True)

    result_list = []

    for num, (key, value) in enumerate(season_dict):
        if num == 0:
            max_seasonal_strength = value[1]
            max_trend_strength = value[2]
        if num <= 2:
            result_list = result_list + value

    if_seasonal = [max_seasonal_strength >= 0.9]
    if_trend = [max_trend_strength >= 0.85]
    result_list = (
        file_name
        + series_length
        + result_list
        + if_seasonal
        + if_trend
        + ADF_P_value
        + KPSS_P_value
        + stability
        + other_features
    )

    result_frame.loc[len(result_frame.index)] = result_list
    print(result_list)
    return result_frame


dir_path = os.path.join(
    r"/Volumes/UDisk/datasets/collected_datasets/徐学姐/GAIA_dataset/data1"
)
file_paths_list = []
for filename in os.listdir(dir_path):
    if filename.startswith("."):
        continue
    path = os.path.join(dir_path, filename)
    file_paths_list.append(path)

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(feature_extract, path) for path in file_paths_list]

concurrent.futures.wait(futures)

completed_results = [future.result() for future in futures]

combined_features = pd.concat(completed_results, ignore_index=True)
end_time = time.time()

execution_time = end_time - start_time
print(f"执行时间为: {execution_time} 秒")

print(combined_features)
combined_features.to_csv(
    "/Volumes/UDisk/datasets/collected_datasets/徐学姐/GAIA_dataset/saved/GAIA_self_features.csv",
    index=False,
)
