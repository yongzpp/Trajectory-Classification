
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import dash
import mysql.connector
import dash_bootstrap_components as dbc
server = Flask(__name__)
server.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:root@db:3306/intel_sample?use_pure=True'
app = dash.Dash(__name__,server=server,url_base_pathname='/main/',external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

db = SQLAlchemy(server)
