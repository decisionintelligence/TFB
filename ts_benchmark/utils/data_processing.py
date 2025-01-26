# -*- coding: utf-8 -*-


from collections import OrderedDict
from typing import List, Optional, Tuple
import pandas as pd


def _parse_target_channel(
    target_channel: Optional[List], num_columns: int
) -> List[int]:
    """
    Parses the target_channel configuration to determine target column indices.
    """
    if target_channel is None:
        return list(range(num_columns))  # Select all columns

    target_columns = []
    for item in target_channel:
        if isinstance(item, int):
            # Handle single integer index (supports negative indices)
            actual_index = item if item >= 0 else num_columns + item
            if 0 <= actual_index < num_columns:
                target_columns.append(actual_index)
            else:
                raise IndexError(
                    f"target_channel configuration error: Column index {item} is out of range (total columns: {num_columns})."
                )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Handle slice represented as a list or tuple, e.g., [2, 4] or (2, 4) selects columns 2 and 3
            start, end = item
            start = start if start >= 0 else num_columns + start
            end = end if end >= 0 else num_columns + end

            if not (0 <= start < num_columns):
                raise IndexError(
                    f"target_channel configuration error: Slice start index {item[0]} is out of range (total columns: {num_columns})."
                )
            if not (0 <= end <= num_columns):
                raise IndexError(
                    f"target_channel configuration error: Slice end index {item[1]} is out of range (total columns: {num_columns})."
                )
            if start > end:
                raise ValueError(
                    f"target_channel configuration error: Slice start index {start} is greater than end index {end}."
                )

            # Add the range of indices to target_columns
            slice_indices = list(range(start, end))
            target_columns.extend(slice_indices)
        else:
            raise ValueError(
                f"target_channel configuration error: Invalid configuration item {item}."
            )

    # Remove duplicates while preserving order (using OrderedDict for compatibility with older Python versions)
    target_columns_unique = list(OrderedDict.fromkeys(target_columns))
    return target_columns_unique


def split_channel(
    df: pd.DataFrame, target_channel: Optional[List] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into target and exog parts based on the target_channel configuration.

    :param df: The input DataFrame to be split.
    :param target_channel: Configuration for selecting target columns.
        It can include:
        - Integers (positive or negative) representing single column indices.
        - Lists or tuples of two integers representing slices, e.g., [2, 4] or (2, 4) selects columns 2 and 3.
        - If set to None, all columns are selected as target columns, and the exog DataFrame is empty.

    Example 1:
        target_channel = [1, 3]
        - Selects columns 1 and 3.

    Example 2:
        target_channel = [(1, 4)]
        - Selects columns 1, 2, and 3 (range from column 1 to column 4 exclusive).

    Example 3:
        target_channel = None
        - Selects all columns as target columns, and the exog DataFrame is empty.

    :return: A tuple containing the target DataFrame and the exog DataFrame.
    """
    num_columns = df.shape[1]  # Total number of columns in the DataFrame

    # Parse target_channel to get target column indices
    target_columns = _parse_target_channel(target_channel, num_columns)

    if target_channel is not None:
        # Determine exog columns by excluding target columns
        all_columns = set(range(num_columns))
        exog_columns = sorted(all_columns - set(target_columns))
    else:
        # If target_channel is None, exog_columns is empty
        exog_columns = []

    # Split the DataFrame into target and exog parts
    target_df = df.iloc[:, target_columns]
    exog_df = df.iloc[
        :, exog_columns
    ]  # This works even if exog_columns is an empty list

    return target_df, exog_df


def split_time(data: pd.DataFrame, index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
    :param index: Split index position.
    :return: Split the first and second half of the data.
    """
    return data.iloc[:index, :], data.iloc[index:, :]
