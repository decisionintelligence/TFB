# -*- coding: utf-8 -*-
import os
import socket
import time


def get_unique_file_suffix():
    """
    Generate a log file name suffix that includes the following information:

    - Hostname
    - The current timestamp, in seconds, is the number of seconds since the Unix era
    - PID (process identifier) of the process

    Return:
    str: The name of the generated log file, in the format '.timestamp.hostname.pid.csv'

    For example, if the host name is' myhost ', the current timestamp is 1631655702, and the current process ID is 12345
    The returned file name may be '.1631655702.myhost.12345.csv'.
    """
    # Get Host Name
    hostname = socket.gethostname()

    # Get current timestamp (seconds since Unix era)
    timestamp = int(time.time())

    # Obtain the PID (process identifier) of the process
    pid = os.getpid()

    # Build file name
    log_filename = f".{timestamp}.{hostname}.{pid}.csv"
    return log_filename
