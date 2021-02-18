import logging

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# see https://community.plot.ly/t/nolayoutexception-on-deployment-of-multi-page-dash-app-example-code/12463/2?u=dcomfort
from annotationapp.app import server
from annotationapp.app import app
from annotationapp.layouts import layout_main, noPage
import annotationapp.callbacks
import annotationapp.config as cfg

# see https://dash.plot.ly/external-resources to alter header, footer and favicon

# Update page
# # # # # # # # #
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/main/':
        return layout_main
    # elif pathname == '/ldnmap/':
    #     return layout_ldn_map
    
    # elif pathname == '/trend/':
    #     return layout_trend
    else:
        return noPage



if __name__ == '__main__':
    app.index_string = ''' 
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Trajectory Annotator</title>
            {%favicon%}
            {%css%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
        ])
    app.css.config.serve_locally = cfg.OFFLINE
    app.scripts.config.serve_locally =  cfg.OFFLINE
    app.run_server(host='0.0.0.0', debug=True)


