# -*- coding: utf-8 -*-

from __future__ import absolute_import

import gzip
import tarfile
from io import BytesIO
from typing import Dict, Optional


def compress_gz(data: Dict[str, str]) -> bytes:
    """
    Compress in gz format
    """
    outbuf = BytesIO()

    with tarfile.open(fileobj=outbuf, mode="w:gz") as tar:
        for k, v in data.items():
            info = tarfile.TarInfo(name=k)
            v_bytes = v.encode("utf8")
            info.size = len(v_bytes)
            tar.addfile(info, fileobj=BytesIO(v_bytes))

    return outbuf.getvalue()

def compress_gzip(data: Dict[str, str]) -> bytes:
    """
    Compress data using Gzip compression.
    """
    outbuf = BytesIO()

    with gzip.GzipFile(fileobj=outbuf, mode="wb") as gz:
        for k, v in data.items():
            v_bytes = v.encode("utf8")
            gz.write(v_bytes)

    return outbuf.getvalue()


def decompress_gzip(compressed_data: bytes) -> Dict[str, str]:
    """
    Decompress Gzip-compressed data and return the original dictionary.
    """
    decompressed_data = {}
    compressed_buf = BytesIO(compressed_data)

    with gzip.GzipFile(fileobj=compressed_buf, mode="rb") as gz:
        while True:
            chunk = gz.read(1024)  # Read a chunk of decompressed data (adjust chunk size if needed)
            if not chunk:
                break  # No more data to read
            chunk_str = chunk.decode("utf8")
            key_values = chunk_str.split("\n")

            for key_value in key_values:
                if key_value:
                    key, value = key_value.split(":")
                    decompressed_data[key] = value

    return decompressed_data


def decompress_gz(data: bytes) -> Dict[str, str]:
    ret = {}
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                ret[member.name] = tar.extractfile(member).read().decode("utf8")

    return ret


def compress(data: Dict[str, str], method: str = "gz") -> bytes:
    if method != "gz":
        compress_gzip(data)
        # raise NotImplementedError("Only 'gz' method is supported by now")
    return compress_gz(data)


def decompress(data: bytes, method: str = "gz") -> Dict[str, str]:
    if method != "gz":
        decompress_gzip(data)
        # raise NotImplementedError("Only 'gz' method is supported by now")
    return decompress_gz(data)


def get_compress_file_ext(method: str) -> str:
    if method != "gz":
        return "gzip"
        # raise NotImplementedError("Only 'gz' method is supported by now")
    return "tar.gz"


def get_compress_method_from_ext(ext: str) -> Optional[str]:
    return {
        "tar.gz": "gz"
    }.get(ext)
