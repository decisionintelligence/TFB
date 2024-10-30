import pandas as pd
from typing import List, Optional, Tuple

def split_dataframe(df: pd.DataFrame, target_channel: Optional[List] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into target and remaining parts based on the target_channel configuration.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be split.
    - target_channel (Optional[List]): Configuration for selecting target columns.
      It can include integers (positive or negative) and lists of two integers representing slices.
      If set to None, all columns are selected as target columns, and the remaining DataFrame is empty.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the target DataFrame and the remaining DataFrame.

    Raises:
    - IndexError: If any specified column index is out of range.
    - ValueError: If the target_channel configuration contains invalid items.
    """
    num_columns = df.shape[1]  # Total number of columns in the DataFrame

    def parse_target_channel(target_channel: Optional[List], num_columns: int) -> List[int]:
        """
        Parses the target_channel configuration to determine target column indices.

        Parameters:
        - target_channel (Optional[List]): Configuration for selecting target columns.
        - num_columns (int): Total number of columns in the DataFrame.

        Returns:
        - List[int]: A list of unique target column indices.

        Raises:
        - IndexError: If any specified column index is out of range.
        - ValueError: If the target_channel contains invalid configurations.
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
            elif isinstance(item, list) and len(item) == 2:
                # Handle slice represented as a list, e.g., [2, 4] selects columns 2 and 3
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

        # Remove duplicates while preserving order
        target_columns_unique = list(dict.fromkeys(target_columns))
        return target_columns_unique

    # Parse target_channel to get target column indices
    target_columns = parse_target_channel(target_channel, num_columns)

    if target_channel is not None:
        # Determine remaining columns by excluding target columns
        all_columns = set(range(num_columns))
        remaining_columns = sorted(all_columns - set(target_columns))
    else:
        # If target_channel is None, remaining_columns is empty
        remaining_columns = []

    # Split the DataFrame into target and remaining parts
    target_df = df.iloc[:, target_columns]
    if remaining_columns:
        remaining_df = df.iloc[:, remaining_columns]
    else:
        # Create an empty DataFrame with the same index as df and zero columns
        remaining_df = pd.DataFrame(index=df.index)

    return target_df, remaining_df
