#%%
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.figure_factory as ff
from .app import app,db
import plotly.graph_objs as go
from plotly import tools
import pandas as pd
from datetime import datetime as dt
from datetime import date, timedelta
from datetime import datetime
import numpy as np
import os 
import dash_bootstrap_components as dbc

from database import database_functions
from . import figures
from . import functions
from . import config as cfg

def get_summary_page():
    df = database_functions.read_labels()['label'].value_counts(dropna=False).reset_index()
    df.rename(columns={'index': 'Label','label': 'Counts'}, inplace=True)
    page = [
        html.H3("Annotation Summary"),
        dcc.Graph(id='label_table',figure=figures.generate_table_figure(df),style={'display':'inline-block','height':'00px'}),
            #html.Button('Show Selected Track on Map',id='show_map_button'),
        ]
    return page

def get_real_page():
    
    plotlyConfig = {'topojsonURL': './assets/'} if cfg.OFFLINE else {} 
    dates = database_functions.read_dates()['date(datetime)'].astype(str).unique()
    page = html.Div([
        html.Div([
            html.H2('Filters'),
            html.Div([
                html.Label('Date Filter'),
                dcc.Dropdown(
                    id='date_filter',
                    clearable = False,
                    placeholder='Select date',
                    options=[{'label':'All Dates','value':'all'}]+[{'label':t,'value':t} for t in dates],
                    value='all'
                    )
                ],style={'display':'inline-block','width':'50%'}),
            ],style={'display':'inline-block','width':'50%','padding':'1px','vertical-align': 'top'}),

        html.Div(
            id = 'show_track',
            children = [
            html.H2('Labels List'),
            dcc.Dropdown(id='tracks', multi = True),
            ]
            ),

        html.Div(
                id='show_track', 
                children=[
                html.H2('Select Track Id'),
                dcc.Dropdown(id='tracksid', multi = True)
                ]),

        html.Div(
            id='map_div',
            children=[html.Div([
                dcc.Checklist(
                    id='map_options'),
                dcc.Graph(
                    id='map',
                    config=plotlyConfig,
                    style={'display':'inline-block','width':'100%','height':'900px'})
                ],style={'display':'inline-block','width':'70%','vertical-align': 'top'}),##this is map width
            html.Div([
                html.Div([
                    html.H4('Altitude/Speed',style={'padding-left': '10px'}),
                    dcc.Graph(id='altitude',style={'display':'inline-block','width':'100%','height':'300px'}),
                    html.H4('Latitude',style={'padding-left': '10px'}),
                    dcc.Graph(id='latitude',style={'display':'inline-block','width':'100%','height':'300px'}),
                    html.H4('Longitude',style={'padding-left': '10px'}),
                    dcc.Graph(id='longitude',style={'display':'inline-block','width':'100%','height':'300px'}),
                    #dcc.Graph(id='track_info_table',style={'display':'inline-block','width':'90%','height':'100px'}),
                    ]),
        ],style={'display':'inline-block','width':'30%','vertical-align': 'top'}),
        ])
    ])
    return page

def get_viz_page():
    
    plotlyConfig = {'topojsonURL': './assets/'} if cfg.OFFLINE else {} 
    dates = database_functions.read_dates()['date(datetime)'].astype(str).unique()

    page = html.Div([
    
        html.Div([
            html.H2('Filters'),
            html.Div([
                html.Label('Date Filter'),
                dcc.Dropdown(
                    id='date_filterpred',
                    clearable = False,
                    placeholder='Select date',
                    options=[{'label':'All Dates','value':'all'}]+[{'label':t,'value':t} for t in dates],
                    value='all'
                    )
                ],style={'display':'inline-block','width':'50%'}),
            ],style={'display':'inline-block','width':'50%','padding':'1px','vertical-align': 'top'}),
    
        html.Div(
                id='show_track', 
                children=[
                html.H2('Pattern List'),
                dcc.Dropdown(id='trackspred',)
                ]),

        html.Div(
                id='show_track', 
                children=[
                html.H2('Select Track Id'),
                dcc.Dropdown(id='tracksidpred', multi = True)
                ]),

        html.Div(
            id='map_div',
            children=[html.Div([
                
                dcc.Checklist(
                    id='map_optionspred'),
                
                dcc.Graph(
                    id='mappred',
                    config=plotlyConfig,
                    style={'display':'inline-block','width':'100%','height':'900px'})
                ],style={'display':'inline-block','width':'70%','vertical-align': 'top'}),##this is map width
            html.Div([
                html.Div([
                    html.H4('Altitude/Speed',style={'padding-left': '10px'}),
                    dcc.Graph(id='altitudepred',style={'display':'inline-block','width':'100%','height':'300px'}),
                    html.H4('Latitude',style={'padding-left': '10px'}),
                    dcc.Graph(id='latitudepred',style={'display':'inline-block','width':'100%','height':'300px'}),
                    html.H4('Longitude',style={'padding-left': '10px'}),
                    dcc.Graph(id='longitudepred',style={'display':'inline-block','width':'100%','height':'300px'}),
                    #dcc.Graph(id='track_info_table',style={'display':'inline-block','width':'90%','height':'100px'}),
                    ]),
        ],style={'display':'inline-block','width':'30%','vertical-align': 'top'}),
        ])
    ])
    return page
def get_upload_page():
    files_for_processing = os.listdir('../csv/')
    page = html.Div([
        html.Div([
            dcc.Dropdown(id='filelist_dropdown',
            placeholder='Select file for upload',
            options=[{'label':f,'value':f} for f in files_for_processing if '.csv' in f]
            ),
        ],style={'display':'inline-block','width':'30%'}),
        html.Div([
            html.Button('upload', id='upload_button'),
            ],style={'display':'inline-block','width':'71%'}),
        html.Div([
            html.Div(id='hidden_div',style={'display':None}),
            dcc.Interval(
                id='log-update',
                interval=1 * 1000  # in milliseconds
            ),
            html.Div(id='log')
            ])
        ])
    return page

ops_assets = {
    'Unknown': 'unknown',
    'F-16' :'f16a',
    'Hawk': 'hawk',
    'Sukhoi' : 'sukhoi',
    }
ops_fobs = {
    'Unknown': 'unknown',
    'Hang Nadim' : 'hang_nadim',
    'Raja Haji Fisabilillah': 'raja_haji',
    'Soewondo' : 'soewondo',
    'Roesmin Nurjadin': 'roesmin',
    'Raden Sadjad': 'raden_sadjad',
    'Halim' : 'halim',
    'Dhomber': 'dhomber',
    'Sri Mulyono Herlambang':'sri_mulyono',
    'Supadio':'supadio'
    }
ops_binary = {
    'Unknown' : 'unknown',
    'Yes': 'yes',
    'No': 'no'
    }
ops_hhq = {
    'Unknown' : 'unknown',
    'Indonesia Air Defence': 'indonesia_air_defence',
    'PANGLIMA': 'panglima',
    'No' : 'no',
    }
ops_service_cord = {
    'Unknown' : 'unknown',
    'TNI AI': 'tni_ai',
    'TNI AD': 'tni_ad'
    }
def get_ops_options_subpage():
    page = html.Div([
        html.Div([
            html.Div([
                html.Label('Asset Involved'),
                dcc.Dropdown(
                    id='asset',
                    clearable = False,
                    placeholder='Select Asset Involved',
                    options=[{'label':k,'value':v} for k,v in ops_assets.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'45%','padding':'5px'}),
            html.Div([
                html.Label('FOB'),
                dcc.Dropdown(
                    id='fob',
                    clearable = False,
                    placeholder='Select FOB',
                    options=[{'label':k,'value':v} for k,v in ops_fobs.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'45%','padding':'5px'}),
        ]),
        html.Div([
            html.Div([
                html.Label('Flight Plan'),
                dcc.Dropdown(
                    id='flight_plan',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
            html.Div([
                html.Label('Flight Response'),
                dcc.Dropdown(
                    id='flight_response',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
            html.Div([
                html.Label('KRI Coordination'),
                dcc.Dropdown(
                    id='kri_cord',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
            html.Div([
                html.Label('Vessel Reporting'),
                dcc.Dropdown(
                    id='vessel_report',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
        ]),
        html.Div([
            html.Div([
                html.Label('Transponder Off'),
                dcc.Dropdown(
                    id='transponder_off',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
            html.Div([
                html.Label('Diversion'),
                dcc.Dropdown(
                    id='diversion',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
            html.Div([
                html.Label('SG Airspace Infringement'),
                dcc.Dropdown(
                    id='airspace_infringe',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
            html.Div([
                html.Label('SAF Asset Interfence'),
                dcc.Dropdown(
                    id='saf_interfere',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_binary.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'22%','padding':'3px'}),
        ]),
        html.Div([
            html.Div([
                html.Label('HHQ Calling'),
                dcc.Dropdown(
                    id='hhq_call',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_hhq.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'45%','padding':'5px'}),
            html.Div([
                html.Label('Service Coordination'),
                dcc.Dropdown(
                    id='service_cord',
                    clearable = False,
                    options=[{'label':k,'value':v} for k,v in ops_hhq.items()],
                    value='unknown'),
                
                ],style={'display':'inline-block','width':'45%','padding':'5px'}),
        ]),
    ],style = {'border': '2px solid #696969',})
    return page