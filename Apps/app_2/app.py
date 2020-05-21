import dash
import dash_bootstrap_components as dbc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State
# from navbar import Navbar

from demo import create_layout, demo_callbacks






BS = "assets/custom.css"
app = dash.Dash(external_stylesheets=[BS])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])




server = app.server
app.layout = create_layout(app)
demo_callbacks(app)


# Running server
if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
