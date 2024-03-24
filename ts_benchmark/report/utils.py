# -*- coding: utf-8 -*-

from __future__ import absolute_import

import io
import os

import os.path
from io import StringIO
from typing import NoReturn

import pandas as pd

from ts_benchmark.utils.compress import get_compress_method_from_ext, decompress, compress, get_compress_file_ext


def read_log_file(fn: str) -> pd.DataFrame:
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


def write_log_file(result_df: pd.DataFrame, file_path: str, compress_method: str) -> str:
    if compress_method is not None:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        write_data = compress({os.path.basename(file_path): buf.getvalue()}, method=compress_method)
        file_path = f"{file_path}.{get_compress_file_ext(compress_method)}"

        with open(file_path, "wb") as fh:
            fh.write(write_data)
    else:
        result_df.to_csv(file_path, index=False)

    return file_path
