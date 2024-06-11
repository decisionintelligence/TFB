# -*- coding: utf-8 -*-

from __future__ import absolute_import

import io
import itertools
import logging
import os
import os.path
from io import StringIO
from typing import List, Optional

import pandas as pd
from pandas.errors import ParserError

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.utils.compress import (
    get_compress_method_from_ext,
    decompress,
    compress,
    get_compress_file_ext,
)
from ts_benchmark.utils.get_file_name import get_unique_file_suffix

logger = logging.getLogger(__name__)


def read_record_file(fn: str) -> pd.DataFrame:
    """
    Reads a single record file.

    The format of the file is currently determined by the extension name.

    :param fn: Path to the record file.
    :return: Benchmarking records in DataFrame format.
    """
    ext = os.path.splitext(fn)[1]
    compress_method = get_compress_method_from_ext(ext)
    if compress_method is None:
        return pd.read_csv(fn)
    else:
        with open(fn, "rb") as fh:
            data = fh.read()
        data = decompress(data, method=compress_method)
        ret = []
        for k, v in data.items():
            ret.append(pd.read_csv(StringIO(v.decode("utf8"))))
        return pd.concat(ret, axis=0)


def write_record_file(
    result_df: pd.DataFrame,
    file_path: str,
    compress_method: Optional[str] = None,
) -> str:
    """
    Write to a single record file.

    :param result_df: Benchmarking records in DataFrame format.
    :param file_path: Path to the record file to save.
    :param compress_method: The format used to compress the record file, if None is given,
        no compression is applied.
    :return: Path to the record file written.
    """
    if compress_method is not None:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        write_data = compress(
            {os.path.basename(file_path): buf.getvalue()}, method=compress_method
        )
        file_path = f"{file_path}.{get_compress_file_ext(compress_method)}"

        with open(file_path, "wb") as fh:
            fh.write(write_data)
    else:
        result_df.to_csv(file_path, index=False)

    return file_path


def load_record_data(
    record_files: List[str], drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Loads benchmarking records from multiple record files.

    :param record_files: The list of paths to the record files. Each item in the list can either
        be the path to a directory or a file. If it is a path to a directory, then all record files
        in the directory are loaded; Otherwise, the file specified by the path is loaded.
    :param drop_columns: The columns to drop during loading.
        This parameter is mainly used to save memory.
    :return: The loaded benchmarking records in DataFrame format.
    """
    record_files = itertools.chain.from_iterable(
        [
            [fn] if not os.path.isdir(fn) else find_record_files(fn)
            for fn in record_files
        ]
    )

    ret = []
    for fn in record_files:
        logger.info("loading log file %s", fn)
        try:
            cur_record = read_record_file(fn)
            if drop_columns:
                cur_record = cur_record.drop(columns=drop_columns)
            ret.append(cur_record)
        except (FileNotFoundError, PermissionError, KeyError, ParserError):
            # TODO: it is ugly to identify log files by artifact columns...
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)


def find_record_files(directory: str) -> List[str]:
    """
    Finds records files in a directory.

    :param directory: The path to the directory.
    :return: The list of file paths to the record files that are found in the give directory.
    """
    record_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # TODO: this is a temporary solution, any good methods to identify a log file?
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                record_files.append(os.path.join(root, file))
    return record_files


def save_log(
    result_df: pd.DataFrame, save_path, file_prefix: str, compress_method: str = "gz"
) -> str:
    """
    Save log data.

    Save the evaluation results, model hyperparameters, model evaluation configuration, and model name to a log file.

    :param result_df: Benchmarking records in DataFrame format.
    :param save_path: Path to the directory where the records are saved.
    :param file_prefix: Prefix of the file name to save the records.
    :param compress_method: The compression method for the output file.
    :return: The path to the output file.
    """
    if result_df["log_info"].any():
        error_itr = filter(None, result_df["log_info"])
        for error in itertools.islice(error_itr, 3):
            logger.info(error)
        if any(error_itr):
            logger.info(
                "-------------More error messages can be found in the record files!-------------"
            )

    if save_path is not None:
        result_path = (
            os.path.join(ROOT_PATH, "result", save_path)
            if not os.path.isabs(save_path)
            else save_path
        )
    else:
        result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)

    record_filename = file_prefix + get_unique_file_suffix()
    file_path = os.path.join(result_path, record_filename)

    return write_record_file(result_df, file_path, compress_method)
