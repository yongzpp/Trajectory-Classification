from datetime import datetime as dt
from datetime import date, timedelta
import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_table

from components import Header
from database import database_functions
from . import figures
from . import config as cfg
import os

# Can be passed as argument to application, automatically determined, etc.
#Here we assume that the appropriate topoJson files (world_110m.json, world_50m.json, etc.)
#normally hosted on https://cdn.plot.ly/ are available under ./assets
#NOTE: folder name MUST be assets


tab_style = {
    
# =============================================================================
#     'borderBottom': '1px solid #d6d6d6',
# # =============================================================================
#     'padding': '6px',
#     'borderBottom':'2px solid #1C2818'    
}

tab_selected_style = {
#     'borderTop': '1.5px solid #1C2818',
#     'borderLeft': '2px solid #1C2818',
#     'borderRight': '2px solid #1C2818',
# # =============================================================================
# #     'borderBottom': '3px solid white',
# # =============================================================================
#     'backgroundColor': 'white',
#     'color': '#1C2818',
#     'padding': '0px',
#     'fontWeight':'bold'
}

main_tab_style = {'display':'inline-block','width':'95%','padding':'15px','border':'2px'}
layout_main = html.Div([
    Header(),
    dcc.Tabs(id='main_tabs',children=[
        dcc.Tab(label = 'Real Data', value='tab_real',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Real Data + Predictions',value='tab_viz',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Upload / Delete',value='tab_upload',style=tab_style, selected_style=tab_selected_style),
        ],
        value='tab_viz',
        style= {'width': '40%'}
        ),
    html.Div(id='main_tab_content')
    ]
    ,style=main_tab_style, className="main-page")
######################## 404 Page ########################
noPage = html.Div([ 
    # CC Header
    Header(),
    html.P([" Oh no...404 Page not found"])
    ], className="no-page")
