# -*- coding: utf-8 -*-

from __future__ import absolute_import

from typing import Optional, List

import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc
from dash.dash_table.Format import Format, Scheme
from dash.development.base_component import Component
from pandas.api.types import is_numeric_dtype

CONTENT_STYLE = {
    "margin-left": "14rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


def generate_layout(page_content: Optional[List] = None) -> Component:
    sidebar = html.Div(
        [
            html.H2("Menu", className="display-4"),
            html.Hr(),
            html.P("Navigation", className="lead"),
            dbc.Nav(
                [
                    dbc.NavLink("Leaderboard", href="/leaderboard", active="exact"),
                    dbc.NavLink("Query", href="/query", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    content = html.Div(children=page_content, id="page-content", style=CONTENT_STYLE)

    return html.Div(
        [dcc.Store(id="raw-data"), dcc.Location(id="url"), sidebar, content]
    )


def format_data_columns(df: pd.DataFrame) -> List:
    ret = []

    for name, dtype in df.dtypes.items():
        if is_numeric_dtype(dtype):
            ret.append(
                {
                    "name": name,
                    "id": name,
                    "type": "numeric",
                    "format": Format(precision=4, scheme=Scheme.decimal_or_exponent),
                }
            )
        else:
            ret.append({"name": name, "id": name})
    return ret
