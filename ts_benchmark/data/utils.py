import os

import pandas as pd

FREQ_MAP = {
    "Y": "yearly",
    "A": "yearly",
    "A-DEC": "yearly",
    "A-JAN": "yearly",
    "A-FEB": "yearly",
    "A-MAR": "yearly",
    "A-APR": "yearly",
    "A-MAY": "yearly",
    "A-JUN": "yearly",
    "A-JUL": "yearly",
    "A-AUG": "yearly",
    "A-SEP": "yearly",
    "A-OCT": "yearly",
    "A-NOV": "yearly",
    "AS-DEC": "yearly",
    "AS-JAN": "yearly",
    "AS-FEB": "yearly",
    "AS-MAR": "yearly",
    "AS-APR": "yearly",
    "AS-MAY": "yearly",
    "AS-JUN": "yearly",
    "AS-JUL": "yearly",
    "AS-AUG": "yearly",
    "AS-SEP": "yearly",
    "AS-OCT": "yearly",
    "AS-NOV": "yearly",
    "BA-DEC": "yearly",
    "BA-JAN": "yearly",
    "BA-FEB": "yearly",
    "BA-MAR": "yearly",
    "BA-APR": "yearly",
    "BA-MAY": "yearly",
    "BA-JUN": "yearly",
    "BA-JUL": "yearly",
    "BA-AUG": "yearly",
    "BA-SEP": "yearly",
    "BA-OCT": "yearly",
    "BA-NOV": "yearly",
    "BAS-DEC": "yearly",
    "BAS-JAN": "yearly",
    "BAS-FEB": "yearly",
    "BAS-MAR": "yearly",
    "BAS-APR": "yearly",
    "BAS-MAY": "yearly",
    "BAS-JUN": "yearly",
    "BAS-JUL": "yearly",
    "BAS-AUG": "yearly",
    "BAS-SEP": "yearly",
    "BAS-OCT": "yearly",
    "BAS-NOV": "yearly",
    "Q": "quarterly",
    "Q-DEC": "quarterly",
    "Q-JAN": "quarterly",
    "Q-FEB": "quarterly",
    "Q-MAR": "quarterly",
    "Q-APR": "quarterly",
    "Q-MAY": "quarterly",
    "Q-JUN": "quarterly",
    "Q-JUL": "quarterly",
    "Q-AUG": "quarterly",
    "Q-SEP": "quarterly",
    "Q-OCT": "quarterly",
    "Q-NOV": "quarterly",
    "QS-DEC": "quarterly",
    "QS-JAN": "quarterly",
    "QS-FEB": "quarterly",
    "QS-MAR": "quarterly",
    "QS-APR": "quarterly",
    "QS-MAY": "quarterly",
    "QS-JUN": "quarterly",
    "QS-JUL": "quarterly",
    "QS-AUG": "quarterly",
    "QS-SEP": "quarterly",
    "QS-OCT": "quarterly",
    "QS-NOV": "quarterly",
    "BQ-DEC": "quarterly",
    "BQ-JAN": "quarterly",
    "BQ-FEB": "quarterly",
    "BQ-MAR": "quarterly",
    "BQ-APR": "quarterly",
    "BQ-MAY": "quarterly",
    "BQ-JUN": "quarterly",
    "BQ-JUL": "quarterly",
    "BQ-AUG": "quarterly",
    "BQ-SEP": "quarterly",
    "BQ-OCT": "quarterly",
    "BQ-NOV": "quarterly",
    "BQS-DEC": "quarterly",
    "BQS-JAN": "quarterly",
    "BQS-FEB": "quarterly",
    "BQS-MAR": "quarterly",
    "BQS-APR": "quarterly",
    "BQS-MAY": "quarterly",
    "BQS-JUN": "quarterly",
    "BQS-JUL": "quarterly",
    "BQS-AUG": "quarterly",
    "BQS-SEP": "quarterly",
    "BQS-OCT": "quarterly",
    "BQS-NOV": "quarterly",
    "M": "monthly",
    "BM": "monthly",
    "CBM": "monthly",
    "MS": "monthly",
    "BMS": "monthly",
    "CBMS": "monthly",
    "W": "weekly",
    "W-SUN": "weekly",
    "W-MON": "weekly",
    "W-TUE": "weekly",
    "W-WED": "weekly",
    "W-THU": "weekly",
    "W-FRI": "weekly",
    "W-SAT": "weekly",
    "D": "daily",
    "B": "daily",
    "C": "daily",
    "H": "hourly",
    "UNKNOWN": "other",
}


def read_data(path: str, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.
    According to the provided file path, read the data file and return the corresponding DataFrame.
    :param path: The path to the data file.
    :return:  The DataFrame of the content of the data file.
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
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def load_series_info(file_path: str) -> dict:
    """
    get series info
    :param file_path: series file path
    :return: series info
    :rtype: dict
    """
    data = read_data(file_path)
    file_name = os.path.basename(file_path)
    freq = pd.infer_freq(data.index)
    freq = FREQ_MAP.get(freq, "other")
    if_univariate = data.shape[1] == 1
    return {
        "file_name": file_name,
        "freq": freq,
        "if_univariate": if_univariate,
        "size": "user",
        "length": data.shape[0],
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
    }
