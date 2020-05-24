import base64
import io
import pathlib
import time

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from PIL import Image
from io import BytesIO
import json
import utils

import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance

import warnings

import dash_bootstrap_components as dbc

import datetime
import math
from PIL import Image
import json

import functions as demo_f



## dataFrame:
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
df = pd.read_csv(DATA_PATH.joinpath('data.csv'), encoding='ISO-8859-1').iloc[:,:]


## Load Images in Store:
df_all = pd.read_csv(DATA_PATH.joinpath('data.csv'), encoding='ISO-8859-1')
images_names = list(np.arange(df_all.shape[0]))
l_images_data = []
for i in range(len(images_names)):
  img = Image.open("data/images/" + str(images_names[i]) + ".png")
  l_images_data.append(img)
  img.load()
df_all = None


def create_layout(app):


  l_header = html.Div(
    [
      html.H3(
        "Plankton Explorer",
        className="header_title",
        id="app-title",
        style={'fontSize': 35}

      ),
      html.H6(
        "Orientadora: Profa. Dra. Nina Hirata",
        className="header_title",
        id="app-professora",
        # style={'fontSize': 16, 'margin-bottom': '2px', 'margin-top': '-4px'}
      ),
      html.H6(
        "Aluno: Renan C. Jacomassi",
        className="header_title",
        id="app-aluno",
        # style={'fontSize': 16}
      )
    ],
    style={'text-align': 'right', }
  )

  l_header_logo = html.Div(
    [
        html.Img(
            src=app.get_asset_url("ime2.png"),
            className="logo",
            id="plotly-ime",
            # style={"max-width": "12%", 'margin-left': '15px', 'margin-right': '16px'},
            style={"max-width": "30%"},
            # opacity= 0,

            # style={'float':'right', 'horizontal-align':'left', 'vertical-align':'left'},
        )
    ],
    # className="three columns header_img",
  )

  l_filtros_1 = html.Div(
    [
        dcc.Input(id="input_label", type="number", placeholder=None, min=0, max=9, step=1,),
        html.Button(id="setar-label", children=["Setar Label"]),
    ],
  )

  l_grafico_1 = html.Div(
    [
        dcc.Graph(id="g0", style={"height": "50vh"}

        )
    ],
  )

  l_grafico_2 = html.Div(
    [
        # dcc.Graph(id="g2", style={"height": "50vh"})
        dcc.Graph(id="g1", style=dict(height='50vh',overflow='scroll')) #auto



    ], #style={"width": "100%", "border": "1px"}
  )

  l_grafico_3 = html.Div(
    [
        # dcc.Graph(id="g2", style={"height": "50vh"})
        dcc.Graph(id="g3", style=dict(height='50vh',overflow='scroll')) #auto



    ], #style={"width": "100%", "border": "1px"}
  )

  l_grafico_4 = html.Div(
    [
        # dcc.Graph(id="g2", style={"height": "50vh"})
        dcc.Graph(id="g4", style=dict(height='50vh',overflow='scroll')) #auto



    ], #style={"width": "100%", "border": "1px"}
  )

  l_stores = html.Div(
    [
      dcc.Store(id='memory0'),
      dcc.Store(id='memory1'),
      dcc.Store(id='memory3'),
      dcc.Store(id='memory4'),
      dcc.Store(id='store_df', storage_type='session'),
      dcc.Store(id='n_clicks_memory'),
      dcc.Store(id='images_list')
    ],
  )

  ### Final Layout
  row = html.Div(
      [

          #Memorys Invisible
          dbc.Row(
              [
                  dbc.Col(l_stores),
              ],
              # align="start",
          ),


          #HEADER
          dbc.Row(
              [
                  dbc.Col(l_header_logo, width={"size": 3}),
                  dbc.Col(l_header, width={"size": 8, "offset":1})
              ],
              # align="start",
          ),

          #Space
          dbc.Row([dbc.Col(html.Div("_________________________")),],),

          #GRAFICOS
          dbc.Row(
              [
                  dbc.Col(html.Div(l_grafico_1)),
                  dbc.Col(html.Div(l_grafico_2)),
              ],
              # align="center",
          ),


          dbc.Row(
              [
                  dbc.Col(html.Div(l_grafico_3)),
                  dbc.Col(html.Div(l_grafico_4)),
              ],
              # align="center",
          ),
          #Space
          dbc.Row([dbc.Col(html.Div("_________________________")),],),

          #Botao Setar Label
          dbc.Row(
              [
                  dbc.Col(html.Div(l_filtros_1))
              ],
              # align="center",
          ),
          dbc.Row(
              [
                  dbc.Col(html.Div("Outros graficos")),

              ],
              # align="end",
          ),

      ]
  )

  return html.Div([row])



def demo_callbacks(app):


  @app.callback(
      [Output('g0', 'figure'),
      Output('g1', 'figure'),
      Output('g3', 'figure'),
      Output('g4', 'figure'),
      Output('memory0', 'data'),
      Output('memory1', 'data'),
      Output('memory3', 'data'),
      Output('memory4', 'data'),
      Output('n_clicks_memory', 'data'),
      Output('store_df', "data")
      ],
      [
      Input('g0', 'selectedData'),
      Input('g1', 'selectedData'),
      Input('g3', 'selectedData'),
      Input('g4', 'selectedData'),
      Input("setar-label", "n_clicks")
      ],
      [
      State('memory0', 'data'),
      State('memory1', 'data'),
      State('memory3', 'data'),
      State('memory4', 'data'),
      State('g0', 'figure'),
      State('g1', 'figure'),
      State('g3', 'figure'),
      State('g4', 'figure'),
      State("input_label", "value"),
      State("n_clicks_memory", "data"),
      State('store_df', "data")
      ]
  )
  def callback(
    # Inputs:
    selection0,
    selection1,
    selection3,
    selection4,
    n_clicks,
    # States:
    memory0,
    memory1,
    memory3,
    memory4,
    figure0,
    figure1,
    figure3,
    figure4,
    input_value,
    n_clicks_memory,
    store_df
    ):


    if n_clicks_memory is None:
      n_clicks_memory = 0
    else:
      n_clicks_memory = n_clicks_memory


    if n_clicks is None:
      n_clicks = 0
    else:
      None


    # (New Charts here):
    _list_selection = [selection0, selection1, selection3, selection4]
    _list_memory = [memory0, memory1, memory3, memory4]
    _list_figure = [None,figure1, figure3, figure4]



    #---------------------------------------------------------------------------------------------------------------------

    if n_clicks == 0:
      #Simplesmente carregou o DataFrame para o Store (sotore_df)
      #A
      # ----
      ####################################
      if store_df is None:

        df_json = df.to_json()

        df['x'] = round(df['x'], 7)
        df['y'] = round(df['y'], 7)

        df['x3'] = round(df['x3'], 7)
        df['y3'] = round(df['y3'], 7)

        df['x4'] = round(df['x4'], 7)
        df['y4'] = round(df['y4'], 7)



        _x_selection, _y_selection = demo_f.f_selection_check(_df=df,
                                    _list_selection=_list_selection,
                                    _list_memory=_list_memory,
                                    _list_figure=_list_figure)

        print("_x_selection", _x_selection)
        print("_y_selection", _y_selection)


        figure_0 = demo_f.f_figure_0(_df = df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)



        figure_1 = demo_f.f_figure_1(_df = df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              images_data=l_images_data)


        figure_3 = demo_f.f_figure_3(_df = df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)


        figure_4 = demo_f.f_figure_4(_df = df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)

        return [figure_0,
                figure_1,
                figure_3,
                figure_4,
                selection0,
                selection1,
                selection3,
                selection4,
                n_clicks,
                df_json]

      #B
      # ----
      ####################################
      else:
        # Dataframe atualizado carregado:
        _df = pd.read_json(store_df)

        df['x'] = round(df['x'], 7)
        df['y'] = round(df['y'], 7)

        df['x3'] = round(df['x3'], 7)
        df['y3'] = round(df['y3'], 7)

        df['x4'] = round(df['x4'], 7)
        df['y4'] = round(df['y4'], 7)





        _x_selection, _y_selection = demo_f.f_selection_check(_df=_df,
                                    _list_selection=_list_selection,
                                    _list_memory=_list_memory,
                                    _list_figure=_list_figure)




        figure_0 = demo_f.f_figure_0(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection=_y_selection)


        print("_x_selection", _x_selection)
        print("_y_selection", _y_selection)

        figure_1 = demo_f.f_figure_1(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              images_data=l_images_data)
        df_store_updated = _df.to_json()


        figure_3 = demo_f.f_figure_3(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)


        figure_4 = demo_f.f_figure_4(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)


        return [figure_0,
                figure_1,
                figure_3,
                figure_4,
                selection0,
                selection1,
                selection3,
                selection4,
                n_clicks_memory,
                df_store_updated]


    else:
      #C   Set Labels
      # ----
      ####################################
      if n_clicks > n_clicks_memory:


        # Dataframe atualizado carregado:
        _df = pd.read_json(store_df)
        df['x'] = round(df['x'], 7)
        df['y'] = round(df['y'], 7)

        df['x3'] = round(df['x3'], 7)
        df['y3'] = round(df['y3'], 7)

        df['x4'] = round(df['x4'], 7)
        df['y4'] = round(df['y4'], 7)

        if selection0 == None and selection1 == None:
          df_store_updated = _df.to_json()
          n_clicks_memory = n_clicks_memory + 1

          return [figure_0,
                figure_1,
                figure_3,
                figure_4,
                selection0,
                selection1,
                selection3,
                selection4,
                n_clicks_memory,
                df_store_updated]
        else:
          None


        if input_value is None:
          input_value = '-'
        else:
          input_value = input_value

        _df['manual_label'][_df.index.isin(figure1['data'][0]['customdata'])] = input_value

        # df_store_updated
        df_store_updated = _df.to_json()
        n_clicks_memory = n_clicks_memory + 1


        selection0 = None
        selection1 = None

        _x_selection = _df['x'].values.tolist()
        _y_selection = _df['y'].values.tolist()


        figure_0 = demo_f.f_figure_0(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection=_y_selection)


        figure_1 = demo_f.f_figure_1(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              images_data=l_images_data)


        figure_3 = demo_f.f_figure_0(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)


        figure_4 = demo_f.f_figure_0(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)



        return [figure_0,
                figure_1,
                figure_3,
                figure_4,
                selection0,
                selection1,
                selection3,
                selection4,
                n_clicks_memory,
                df_store_updated]
      #D
      # ----
      ####################################
      else:


        # Dataframe atualizado carregado:
        # print ("Aqui")
        _df = pd.read_json(store_df)
        df['x'] = round(df['x'], 7)
        df['y'] = round(df['y'], 7)

        df['x3'] = round(df['x3'], 7)
        df['y3'] = round(df['y3'], 7)

        df['x4'] = round(df['x4'], 7)
        df['y4'] = round(df['y4'], 7)


        _x_selection, _y_selection = demo_f.f_selection_check(_df=_df,
                                    _list_selection=_list_selection,
                                    _list_memory=_list_memory,
                                    _list_figure=_list_figure)



        figure_0 = demo_f.f_figure_0(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)



        figure_1 = demo_f.f_figure_1(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              images_data=l_images_data)
        df_store_updated = _df.to_json()


        figure_3 = demo_f.f_figure_0(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)


        figure_4 = demo_f.f_figure_0(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)



        return [figure_0,
                figure_1,
                figure_3,
                figure_4,
                selection0,
                selection1,
                selection3,
                selection4,
                n_clicks_memory,
                df_store_updated]







