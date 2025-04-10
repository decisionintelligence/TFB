# Standard library imports
import os
import warnings
from typing import List, Tuple, Optional

# Third-party imports
import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import FloatVector, pandas2ri
from scipy.signal import argrelextrema
from scipy.stats import skew, kurtosis, norm, entropy
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stl._stl import STL

# Constants
DEFAULT_PERIODS = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]
TSFEATURE_NAMES = [
    "max_kl_shift", "max_level_shift", "max_var_shift", "acf_features",
    "arch_stat", "crossing_points", "entropy", "flat_spots", "holt_parameters",
    "hurst", "lumpiness", "nonlinearity", "pacf_features", "stability",
    "unitroot_kpss", "unitroot_pp", "firstmin_ac", "firstzero_ac",
    "trev_num", "walker_propcross", "std1st_der", "histogram_mode",
    "heterogeneity"
]

# Suppress warnings
warnings.filterwarnings("ignore")
pandas2ri.activate()


class TimeSeriesFeatureExtractor:
    def __init__(self):
        self._setup_r_environment()

    def read_data(self, path: str, nrows=None) -> pd.DataFrame:
        """
        Read the data file and return DataFrame.
        According to the provided file path, read the data file and return the corresponding DataFrame.
        :param path: The path to the data file.
        :return: The DataFrame of the content of the data file.
        """
        data = pd.read_csv(path)
        label_exists = "label" in data["cols"].values

        all_points = data.shape[0]
        columns = data.columns

        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()

        is_univariate = n_points == all_points
        n_cols = all_points // n_points
        df = pd.DataFrame()
        cols_name = data["cols"].unique()

        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        else:
            df[cols_name[0]] = data.iloc[:, 0]

        if label_exists:
            last_col_name = df.columns[-1]
            df.rename(columns={last_col_name: "label"}, inplace=True)
            df = df.drop(columns='label')

        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]

        return df

    def _setup_r_environment(self):
        """
        Initialize R environment and load required functions.
        Sets up the R environment with necessary libraries and functions for time series analysis.
        """
        r_script = '''
        library(tidyverse)
        library(Rcatch22)
        library(forecast)
        library(tsfeatures)

        calculate_features <- function(dir_name, data) {
            TSFEATURE_NAMES <- c("max_kl_shift",
                               "max_level_shift",
                               "max_var_shift",
                               "acf_features",
                               "arch_stat",
                               "crossing_points",
                               "entropy",
                               "flat_spots",
                               "holt_parameters",
                               "hurst",
                               "lumpiness",
                               "nonlinearity",
                               "pacf_features",
                               "stability",
                               "unitroot_kpss",
                               "unitroot_pp",
                               "firstmin_ac",
                               "firstzero_ac",
                               "trev_num",
                               "walker_propcross",
                               "std1st_der",
                               "histogram_mode",
                               "heterogeneity")

            all_features = tibble()
            catch_all_features = tibble()
            features = tibble()
            feature_type = "tsfeatures"

            i = 1
            seasonal.period = 1

            tryCatch(
                expr = {
                    ts <- forecast:::msts(data, seasonal.periods = seasonal.period)

                    if(feature_type == "tsfeatures"){ 
                        feature <- tibble(file_name = dir_name)
                        features <- tsfeatures:::tsfeatures(ts, c("mean","var"), scale = FALSE, na.rm = TRUE)
                        features <- bind_cols(feature, features)
                        for(f in TSFEATURE_NAMES){
                            calculated_features <- tsfeatures:::tsfeatures(ts, features = f)

                            if(sum(is.na(calculated_features)) > 0){ 
                                calculated_features <- tsfeatures:::tsfeatures(ts(ts, frequency = 1), features = f)

                                if(sum(is.na(calculated_features)) > 0){ 
                                    if(f == "max_kl_shift" | f == "max_level_shift" | f == "max_var_shift")
                                        calculated_features <- tsfeatures:::tsfeatures(ts, features = f, width = 1)
                                    else{
                                        if(f == "arch_stat")
                                            calculated_features <- tsfeatures:::tsfeatures(ts, features = f, lag = 1)
                                    }
                                }
                            }
                            features <- bind_cols(features, calculated_features)
                        }

                        tryCatch( 
                            {
                                seasonal_features <- tsfeatures:::tsfeatures(ts, "stl_features", s.window = 'periodic', robust = TRUE)
                            },
                            error = function(e) {
                                tryCatch({
                                    seasonal_features <<- tsfeatures:::tsfeatures(ts, "stl_features")
                                }, error = function(e) {
                                    seasonal_features <<- tsfeatures:::tsfeatures(ts(ts, frequency = 1), "stl_features")
                                })
                            }
                        )

                        catch_features <- catch22_all(ts)

                        if(i == 1){
                            catch_all_features <- as_tibble(matrix(NA, nrow = 1, ncol = 22))
                            colnames(catch_all_features) <- catch_features$names
                        }

                        catch_all_features[i,] <- as_tibble(t(catch_features$values))
                        i = i + 1

                        features <- bind_cols(features, seasonal_features)
                        lambdas <- tibble(lambdas = forecast:::BoxCox.lambda(ts))
                        features <- bind_cols(features, lambdas)
                    }
                    all_features <- bind_rows(all_features, features)

                },
                error = function(err) {
                    if (inherits(err, "error")) {
                        cat("An error occurred: ", conditionMessage(err), "\n")
                        cat("error file name:", dir_name, "\n")
                    }
                }
            )
            catch_all_features <- na.omit(catch_all_features)
            all_features <- bind_cols(all_features, catch_all_features)

            return(all_features)
        }
        '''
        robjects.r(r_script)
        self.calculate_features = robjects.globalenv['calculate_features']

    def adjust_period(self, period_value: int) -> int:
        """
        Adjust period value to nearest standard period.
        Maps the input period to the closest standard period value.
        :param period_value: The input period value to adjust
        :return: The adjusted standard period value
        """
        if abs(period_value - 4) <= 1:
            return 4
        if abs(period_value - 7) <= 1:
            return 7
        if abs(period_value - 12) <= 2:
            return 12
        if abs(period_value - 24) <= 3:
            return 24
        if abs(period_value - 48) <= 1 or ((48 - period_value) <= 4 and (48 - period_value) >= 0):
            return 48
        if abs(period_value - 52) <= 2:
            return 52
        if abs(period_value - 96) <= 10:
            return 96
        if abs(period_value - 144) <= 10:
            return 144
        if abs(period_value - 168) <= 10:
            return 168
        if abs(period_value - 336) <= 50:
            return 336
        if abs(period_value - 672) <= 20:
            return 672
        if abs(period_value - 720) <= 20:
            return 720
        if abs(period_value - 1008) <= 100:
            return 1008
        if abs(period_value - 1440) <= 200:
            return 1440
        if abs(period_value - 8766) <= 500:
            return 8766
        if abs(period_value - 10080) <= 500:
            return 10080
        if abs(period_value - 21600) <= 2000:
            return 21600
        if abs(period_value - 43200) <= 2000:
            return 43200
        return period_value

    def fft_transfer(self, timeseries: np.ndarray, fmin: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Fast Fourier Transform on time series.
        :param timeseries: Input time series data
        :param fmin: Minimum frequency threshold
        :return: Tuple of periods and amplitudes
        """
        yf = abs(np.fft.fft(timeseries))
        yfnormlize = yf / len(timeseries)
        yfhalf = yfnormlize[: len(timeseries) // 2] * 2

        fwbest = yfhalf[argrelextrema(yfhalf, np.greater)]
        xwbest = argrelextrema(yfhalf, np.greater)

        fwbest = fwbest[fwbest >= fmin].copy()

        return len(timeseries) / xwbest[0][: len(fwbest)], fwbest

    def count_inversions(self, series: np.ndarray) -> int:
        """
        Count inversions in a series using merge sort.
        :param series: Input time series data
        :return: Number of inversions in the series
        """
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

    def count_peaks_and_valleys(self, sequence: np.ndarray) -> int:
        """
        Count peaks and valleys in a sequence.
        :param sequence: Input sequence data
        :return: Total count of peaks and valleys
        """
        peaks = 0
        valleys = 0

        for i in range(1, len(sequence) - 1):
            if sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
                peaks += 1
            elif sequence[i] < sequence[i - 1] and sequence[i] < sequence[i + 1]:
                valleys += 1

        return peaks + valleys

    def count_series(self, sequence: np.ndarray, threshold: float) -> int:
        """
        Count number of series above/below threshold.
        :param sequence: Input sequence data
        :param threshold: Threshold value for counting
        :return: Total count of series crossing the threshold
        """
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

    def extract_other_features(self, series_value: np.ndarray) -> List[float]:
        """
        Extract additional statistical features from time series.
        :param series_value: Input time series data
        :return: List of extracted features
        """
        skewness = skew(series_value)
        kurt = kurtosis(series_value)
        rsd = abs((np.std(series_value) / np.mean(series_value)) * 100)
        std_of_first_derivative = np.std(np.diff(series_value))
        inversions = self.count_inversions(series_value) / len(series_value)
        turning_points = self.count_peaks_and_valleys(series_value) / len(series_value)
        series_in_series = self.count_series(series_value, np.median(series_value)) / len(series_value)

        return [
            skewness,
            kurt,
            rsd,
            std_of_first_derivative,
            inversions,
            turning_points,
            series_in_series,
        ]

    def feature_extract(self, path: str) -> pd.DataFrame:
        """
        Extract features from time series data.
        :param path: Path to the input data file
        :return: DataFrame containing extracted features
        """
        index_columns = [
            "length", "period_value1", "seasonal_strength1",
            "trend_strength1", "period_value2", "seasonal_strength2",
            "trend_strength2", "period_value3", "seasonal_strength3",
            "trend_strength3", "if_season", "if_trend", "ADF:p-value",
            "KPSS:p-value", "stability", "skewness", "kurt", "rsd",
            "std_of_first_derivative", "inversions", "turning_points",
            "series_in_series",
        ]
        result_frame = pd.DataFrame(columns=index_columns)

        original_df = self.read_data(path)
        limited_length_df = original_df
        series_length = [original_df.shape[0]]

        for i, col in enumerate(limited_length_df.columns):
            try:
                ADF_P_value = [adfuller(limited_length_df[col].values, autolag="AIC")[1]]
                KPSS_P_value = [kpss(limited_length_df[col].values, regression="c")[1]]
                stability = [ADF_P_value[0] <= 0.05 or KPSS_P_value[0] >= 0.05]
            except:
                ADF_P_value = [None]
                KPSS_P_value = [None]
                stability = [None]

            series_value = limited_length_df[col]
            origin_series_value = original_df[col]
            series_value = pd.Series(series_value).astype("float")
            origin_series_value = pd.Series(origin_series_value).astype("float")

            other_features = self.extract_other_features(origin_series_value)
            periods, amplitude = self.fft_transfer(series_value, fmin=0)

            periods_list = []
            for index_j in range(len(amplitude)):
                periods_list.append(
                    round(periods[amplitude.tolist().index(sorted(amplitude, reverse=True)[index_j])])
                )

            final_periods1 = []
            for l1 in periods_list:
                l1 = self.adjust_period(l1)
                if l1 not in final_periods1 and l1 >= 4:
                    final_periods1.append(l1)

            periods_num = min(len(final_periods1), 3)
            new_final_periods = final_periods1[:periods_num] + DEFAULT_PERIODS

            final_periods = []
            for l1 in new_final_periods:
                if l1 not in final_periods and l1 >= 4:
                    final_periods.append(l1)

            yuzhi = max(int(series_length[0] / 3), 12)

            season_dict = {}
            for index_period in range(len(final_periods)):
                period_value = final_periods[index_period]

                if period_value < yuzhi:
                    res = STL(limited_length_df[col], period=period_value).fit()
                    temp_df = pd.DataFrame()
                    temp_df["original"] = limited_length_df[col]
                    temp_df["trend"] = res.trend
                    temp_df["seasonal"] = res.seasonal
                    temp_df["resid"] = res.resid
                    temp_df["detrend"] = temp_df["original"] - temp_df["trend"]
                    temp_df["deseasonal"] = temp_df["original"] - temp_df["seasonal"]

                    trend_strength = 0 if temp_df["deseasonal"].var() == 0 else \
                        max(0, 1 - temp_df["resid"].var() / temp_df["deseasonal"].var())
                    seasonal_strength = 0 if temp_df["detrend"].var() == 0 else \
                        max(0, 1 - temp_df["resid"].var() / temp_df["detrend"].var())

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

            col_name = [str(i)]

            result_list = (
                    series_length
                    + result_list
                    + if_seasonal
                    + if_trend
                    + ADF_P_value
                    + KPSS_P_value
                    + stability
                    + other_features
            )

            result_frame.loc[len(result_frame.index)] = result_list

        return result_frame


class StatisticalCalculator:
    @staticmethod
    def compute_correlation(data):
        """
        Compute correlation from the last 22 columns of the input data.
        :param data: Input DataFrame or numpy array with at least 22 columns
        :return: Computed correlation value
        """
        n_samples = data.shape[0]
        if n_samples == 1:
            return None

        data = data.iloc[:, -22:]
        data.columns = [str(col) for col in data.columns]

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        corr_list = []

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                corr = abs(np.corrcoef(data_scaled[i], data_scaled[j])[0, 1])
                corr_list.append(corr)

        series_mean = np.mean(corr_list)
        series_var = np.var(corr_list)

        correlation = 2 * (series_mean + 1 / (series_var + 2)) / 3

        return correlation

    @staticmethod
    def calculate_jsd_for_window(data: np.ndarray, window_size: int) -> float:
        """
        Calculate Jensen-Shannon divergence for a given window.
        :param data: Input time series data
        :param window_size: Size of the window for calculation
        :return: Mean JSD value for the window
        """
        jsd_list = []
        num_windows = len(data) // window_size

        for i in range(num_windows):
            window_data = data[i * window_size: (i + 1) * window_size]
            hist, bin_edges = np.histogram(window_data, bins='stone', density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            mu = np.mean(window_data)
            sigma = np.std(window_data)

            if sigma == 0:
                jsd_list.append(0)
                continue

            pdf = norm.pdf(bin_centers, mu, sigma)
            jsd = StatisticalCalculator.js_divergence(hist, pdf)
            jsd_list.append(jsd)

        return np.mean(jsd_list)

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.
        :param p: First distribution
        :param q: Second distribution
        :return: JSD value between the distributions
        """
        m = 0.5 * (p + q)
        kl_p_m = entropy(p, m)
        kl_q_m = entropy(q, m)
        return 0.5 * (kl_p_m + kl_q_m)

    @staticmethod
    def calculate_jsd_multivariate(df: pd.DataFrame, window_size: int) -> List[float]:
        """
        Calculate Jensen-Shannon divergence for multivariate data.
        :param df: Input DataFrame containing multivariate data
        :param window_size: Size of the window for calculation
        :return: List of JSD values for each variable
        """
        # jsd_list = []
        # unique_cols = df['cols'].unique()

        # for col in df.columns:
        #     var_data = df[df['cols'] == col]['data'].values
        #     avg_jsd = StatisticalCalculator.calculate_jsd_for_window(var_data, window_size)
        #     jsd_list.append(avg_jsd)

        # return jsd_list
        return [
            StatisticalCalculator.calculate_jsd_for_window(df[col].values, window_size)
            for col in df.columns
        ]

    @staticmethod
    def calculate_jsd(filename: str) -> pd.DataFrame:
        """
        Calculate JSD for both short and long term windows.
        :param filename: Path to the input data file
        :return: DataFrame containing JSD values
        """
        feature_extractor = TimeSeriesFeatureExtractor()
        df = feature_extractor.read_data(path=filename)

        short_term_window_size = 30
        long_term_window_size = 336

        short_term_jsd = StatisticalCalculator.calculate_jsd_multivariate(
            df, short_term_window_size
        )
        long_term_jsd = StatisticalCalculator.calculate_jsd_multivariate(
            df, long_term_window_size
        )

        return pd.DataFrame(
            data={
                "short_term_jsd": short_term_jsd,
                "long_term_jsd": long_term_jsd,
            }
        )


class TimeSeriesProcessor:
    def __init__(self, output_dir: str = "characteristics"):
        """
        Initialize TimeSeriesProcessor.
        :param output_dir: Directory for saving output files
        """
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.stat_calculator = StatisticalCalculator()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a single file and extract features.
        :param file_path: Path to the input file
        :return: DataFrame containing extracted features
        """
        data = self.feature_extractor.read_data(file_path)
        num_cols = data.shape[1]

        file_results = []
        for i in range(num_cols):
            col_data = data.iloc[:, i].dropna().tolist()
            r_data = FloatVector(col_data)
            features = self.feature_extractor.calculate_features(str(i), r_data)
            df_features = pandas2ri.rpy2py(features)
            numeric_columns = df_features.select_dtypes(include=[np.number]).columns
            df_features[numeric_columns] = df_features[numeric_columns]
            file_results.append(df_features)

        return pd.concat(file_results, ignore_index=True) if file_results else None

    def process_path(self, file_path: str) -> None:
        """
        Process time series data from file or directory.
        :param file_path: Path to the input file or directory
        """
        if os.path.isfile(file_path) and file_path.lower().endswith(".csv"):
            self._process_single_file(file_path)
        elif os.path.isdir(file_path):
            self._process_directory(file_path)
        else:
            print(f"Invalid path: {file_path}")

    def _process_single_file(self, file_path: str) -> None:
        """
        Process a single CSV file and save results.
        :param file_path: Path to the input CSV file
        """
        print(f"Processing file: {file_path}")

        file_basename = os.path.splitext(os.path.basename(file_path))[0]

        data = self.feature_extractor.read_data(file_path)
        col_names = data.columns.tolist()
        num_cols = data.shape[1]

        if "label" in col_names:
            num_cols -= 1

        is_univariate = num_cols == 1

        feature_extract_result = self.feature_extractor.feature_extract(file_path)
        process_result = self.process_file(file_path)
        jsd_result = self.stat_calculator.calculate_jsd(file_path)

        result = pd.concat([
            jsd_result,
            feature_extract_result.loc[:, ["seasonal_strength1", "trend_strength1", "ADF:p-value"]],
            process_result
        ], axis=1)

        if not is_univariate:
            mean_results = self._calculate_mean_results(result)
            self._save_mean_results(mean_results, file_basename)

        result["DN_OutlierInclude_p_001_mdrmd"] = result["DN_OutlierInclude_p_001_mdrmd"].abs()
        self._save_basic_results(result, file_basename)

    def _save_basic_results(self, result: pd.DataFrame, file_prefix: str) -> None:
        """
        Save basic feature results to output directory.
        :param result: DataFrame containing feature results
        :param file_prefix: Prefix for output files
        """
        all_features_filename = os.path.join(self.output_dir, f"All_characteristics_{file_prefix}.csv")
        features_filename = os.path.join(self.output_dir, f"TFB_characteristics_{file_prefix}.csv")

        result.to_csv(all_features_filename, index=False)
        result["correlation"] = None
        result = result[['correlation'] + [col for col in result.columns if col != 'correlation']]
        dropandrename_dataframe(result).to_csv(features_filename, index=False)

        print(f"Saved results to: {all_features_filename} and {features_filename}")

    def _calculate_mean_results(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean features and correlation.
        :param result: Input DataFrame containing features
        :return: DataFrame containing mean features
        """
        mean_result = result.copy()
        correlation = self.stat_calculator.compute_correlation(result)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        non_numeric_cols = result.select_dtypes(exclude=[np.number]).columns
        result["DN_OutlierInclude_p_001_mdrmd"] = result["DN_OutlierInclude_p_001_mdrmd"].abs()
        mean_result = result[numeric_cols].mean().to_frame().transpose()
        mean_result["correlation"] = correlation

        return mean_result

    def _save_mean_results(self, mean_result: pd.DataFrame, file_prefix: str) -> None:
        """
        Save mean feature results to output directory.
        :param mean_result: DataFrame containing mean features
        :param file_prefix: Prefix for output files
        """
        mean_all_features_filename = os.path.join(self.output_dir, f"mean_All_characteristics_{file_prefix}.csv")
        mean_features_filename = os.path.join(self.output_dir, f"mean_TFB_characteristics_{file_prefix}.csv")

        mean_result.to_csv(mean_all_features_filename, index=False)
        dropandrename_dataframe(mean_result).to_csv(mean_features_filename, index=False)

        print(f"Saved mean results to: {mean_all_features_filename} and {mean_features_filename}")

    def _process_directory(self, dir_path: str) -> None:
        """
        Process all CSV files in a directory.
        :param dir_path: Path to the directory containing CSV files
        """
        print(f"Processing directory: {dir_path}")
        file_names = [f for f in os.listdir(dir_path)
                      if os.path.isfile(os.path.join(dir_path, f))
                      and f.lower().endswith(".csv")]

        if not file_names:
            print(f"No CSV files found in: {dir_path}")
            return

        for file_name in file_names:
            full_path = os.path.join(dir_path, file_name)
            self._process_single_file(full_path)


def dropandrename_dataframe(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns and rename remaining ones.
    :param result_df: Input DataFrame
    :return: Processed DataFrame with renamed columns
    """
    result = result_df.loc[
             :,
             [
                 "correlation",
                 "SB_TransitionMatrix_3ac_sumdiagcov",
                 "DN_OutlierInclude_p_001_mdrmd",
                 "seasonal_strength1",
                 "trend_strength1",
                 "ADF:p-value",
                 "short_term_jsd",
                 "long_term_jsd",
             ],
             ]

    result.columns = [
        "Correlation",
        "Transition",
        "Shifting",
        "Seasonality",
        "Trend",
        "Stationarity",
        "Short_term_jsd",
        "Long_term_jsd",
    ]

    return result


if __name__ == "__main__":
    processor = TimeSeriesProcessor(output_dir="characteristics")
    file_path = r"./DemoDatasets/Exchange.csv"  # supports input a single file
    # file_path = r"./DemoDatasets" # supports input a folder
    processor.process_path(file_path)
    print("Processing completed")
