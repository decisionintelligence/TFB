# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
from typing import Union, List, Dict, NoReturn

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html
from flask import Flask, redirect

from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.recording import load_record_data
from ts_benchmark.report.report_dash.memory import READONLY_MEMORY
from ts_benchmark.report.utils.leaderboard import get_leaderboard

# currently we do not support showing or processing artifact columns
# these columns are dropped as soon as data is loaded in order to save memory
ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
    FieldNames.LOG_INFO,
]

logger = logging.getLogger(__name__)


def report(report_config: Dict) -> NoReturn:
    log_files: Union[List[str], pd.DataFrame] = report_config.get("log_files_list")
    if not log_files:
        raise ValueError("No log files to report")

    log_data = (
        log_files
        if isinstance(log_files, pd.DataFrame)
        else load_record_data(log_files, drop_columns=ARTIFACT_COLUMNS)
    )

    log_data = log_data.drop(columns=ARTIFACT_COLUMNS, errors="ignore")

    leaderboard_df = get_leaderboard(
        log_data,
        report_config["report_metrics"],
        report_config.get("aggregate_type", "mean"),
        report_config.get("fill_type", "mean_value"),
        report_config.get("null_value_threshold", 0.3),
    )

    READONLY_MEMORY["raw_data"] = log_data
    READONLY_MEMORY["leaderboard_df"] = leaderboard_df

    server = Flask(__name__)

    @server.route("/")
    def index_redirect():
        return redirect("/leaderboard")

    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        use_pages=True,
    )

    # Define the layout
    app.layout = html.Div([dash.page_container])

    app.run(
        host=report_config.get("host", "0.0.0.0"),
        port=report_config.get("port", "12345"),
        debug=report_config.get("debug", False),
    )
