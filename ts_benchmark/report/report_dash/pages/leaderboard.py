# -*- coding: utf-8 -*-

from __future__ import absolute_import

import dash
from dash import dash_table

from ts_benchmark.report.report_dash.layout import generate_layout, format_data_columns
from ts_benchmark.report.report_dash.memory import READONLY_MEMORY

dash.register_page(__name__)

leaderboard_df = READONLY_MEMORY["leaderboard_df"]

table = dash_table.DataTable(
    id="leaderboard",
    columns=format_data_columns(leaderboard_df),
    data=leaderboard_df.to_dict("records"),
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    column_selectable="single",
    row_selectable="multi",
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current=0,
    page_size=100,
    style_data={
        "whiteSpace": "normal",
        "height": "auto",
        "lineHeight": "15px",
        "maxWidth": "200px",
        "wordWrap": "break-word",
    },
)

layout = generate_layout([table])
