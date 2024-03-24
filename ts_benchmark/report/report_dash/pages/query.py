# -*- coding: utf-8 -*-

from __future__ import absolute_import

import dash
import pandas as pd
from dash import html, dcc, callback, Output, Input, State, dash_table

from ts_benchmark.report.report_dash.layout import generate_layout, format_data_columns
from ts_benchmark.report.report_dash.memory import READONLY_MEMORY

dash.register_page(__name__)

raw_data = READONLY_MEMORY["raw_data"]

content = html.Div(
    [
        dcc.Input(
            id="sql-input", placeholder="Enter SQL statement...", type="text", value=""
        ),
        html.Button("Execute", id="execute-button", n_clicks=0),
        html.Hr(),
        html.Div(id="error-message"),
        dash_table.DataTable(
            id="query-table",
            columns=[],
            data=[],
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
        ),
    ]
)


def _query_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if "{}" in query:
        # we do not apply security checks on the query
        # DO NOT serve this program to untrusted users
        return eval(query.format("df"))
    else:
        return df.query(query)


@callback(
    Output("query-table", "columns"),
    Output("query-table", "data"),
    Output("error-message", "children"),
    Input("execute-button", "n_clicks"),
    State("sql-input", "value"),
)
def execute_sql(n_clicks, sql_statement):
    if n_clicks > 0:
        try:
            result = _query_df(raw_data, sql_statement) if sql_statement else raw_data
            if result.index.name is not None:
                result = result.reset_index()
            columns = format_data_columns(result)
            data = result.to_dict("records")
            return columns, data, None
        except Exception as e:
            error_message = html.Div(children=[html.H4("Error"), html.P(str(e))])
            return [], [], error_message
    else:
        return [], [], None


layout = generate_layout([content])
