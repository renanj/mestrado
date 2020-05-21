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



# make a sample data frame with 6 columns
np.random.seed(0)
df = pd.DataFrame({"Col " + str(i+1): np.random.rand(5) for i in range(6)})
hidden_inputs = html.Div(id="hidden-inputs", style={"display": "none"}, children=[])


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
# data_dict = {
#     'mnist_3000': pd.read_csv(DATA_PATH.joinpath('mnist_3000_input.csv')).loc[:,:],
# }

df = pd.read_csv(DATA_PATH.joinpath('data.csv'), encoding='ISO-8859-1').iloc[:400,:]

# print ("df", data_dict['mnist_3000'].head())




def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64

def generate_thumbnail(image_b64):
    return html.Div([
        html.A([
            html.Img(
                src="data:image/png;base64, " + image_b64,
                style = {
                    'height': '10%',
                    'width': '10%',
                    'float': 'left',
                    'max-width': '235px',
                    'position': 'relative',
                    'padding-top': 5,
                    'padding-right': 5
                }
            )
        ],),
    ])







def create_images_layout(qtd_pontos=0, _qtd_cols_figure_2=6, selectedpoints=[1], images_names=None):

  list_parameters = [
    'x',
    'y',
    'sizex',
    'sizey',
    'source',
    'opacity',
    'xanchor',
    'yanchor',
    'xref',
    'yref'
  ]
  list_values = [
    1,
    0,
    0.5,
    0.5,
    # Image.open(BytesIO(base64.b64decode(numpy_to_b64(data_dict['mnist_3000'].iloc[0,:].values.reshape(28, 28).astype(np.float64))))),
    # Image.open("plancton_" + str(1) + ".png"),
    Image.open("data/" + str(1) + ".png"),
    1,
    'center',
    'middle',
    'x',
    'y'
  ]



  _list_of_dicts = []
  for i in range(qtd_pontos):
      list_values[0] = i - ((math.ceil((i + 1) / _qtd_cols_figure_2) -1) * _qtd_cols_figure_2) + 1
      list_values[1] = math.ceil((i + 1) / _qtd_cols_figure_2) - 1
      _num = str(images_names[i])
      # list_values[4] = Image.open(BytesIO(base64.b64decode(numpy_to_b64(data_dict['mnist_3000'].iloc[0,:].values.reshape(28, 28).astype(np.float64)))))
      # list_values[4] = Image.open("plancton_" + _num + ".png")
      list_values[4] = Image.open("data/" + _num + ".png")
      _dict = {list_parameters[i]: list_values[i] for i in range(len(list_parameters))}
      _list_of_dicts.append(_dict)
      list_values[4].load()

  return _list_of_dicts

def f_selection_check(_df, _selection_1, _selection_2, _memory_1, _memory_2, _figure2, _qtd_cols_figure_2=6):

  _x_selection = None
  _y_selection = None

  # _df['x'] = round(_df['x'], 7)
  # _df['y'] = round(_df['y'], 7)

  # Selection_1 vs Memory 1
  if _selection_1 != _memory_1:
    try:
      temp_df = pd.DataFrame(_selection_1["points"])
      _x_selection = temp_df['x'].values.tolist()
      _y_selection = temp_df['y'].values.tolist()
    except:
      temp_df = None
  else:
    None


  # Selection_2 vs Memory 2
  if _selection_2 != _memory_2:
    try:
      _figure2 = _figure2['data']
      _text = []
      _selected_points = []
      _custom_points = []
      i_count = 0
      for i in range(len(_figure2)):
        _selected_points.append(_figure2[i]['selectedpoints'])
        _custom_points.append(_figure2[i]['customdata'][(_qtd_cols_figure_2 * i_count):(_qtd_cols_figure_2*(i_count+1))])
        i_count = i_count +1
      _final_selected_points = []

      for i in range(len(_selected_points)):
        for ii in range(len(_selected_points[i])):
          _final_selected_points.append(_custom_points[i][_selected_points[i][ii]])

      _x_selection = _df['x'][_df.index.isin(_final_selected_points)].values.tolist()
      _y_selection = _df['y'][_df.index.isin(_final_selected_points)].values.tolist()
    except:
      temp_df = None
  else:
    None



  if _x_selection == None:
    # print("entrou")
    _x_selection = _df['x'].values.tolist()
  else:
    None

  if _y_selection == None:
    _y_selection = _df['y'].values.tolist()
  else:
    None

  return _x_selection, _y_selection

def f_figure_1(_df, _x_selection, _y_selection):


  l_data = []
  groups = _df.groupby('manual_label')

  for idx, val in groups:
    ## Selected Points
    _selectedpoints = _df['manual_label'][
      (_df['manual_label'] == idx) &
      (_df['x'].isin(_x_selection)) &
      (_df['y'].isin(_y_selection))
    ].index.values
    _temp = []
    for i in range(len(_selectedpoints)):
          _temp.append(val.index.get_loc(_selectedpoints[i]))
    _selectedpoints = _temp

    scatter = go.Scatter(
        name=idx,
        x=val["x"],
        y=val["y"],
        text=val['manual_label'],
        selectedpoints=_selectedpoints,
        textposition="top center",
        mode="markers",
        marker=dict(size=8, symbol="circle")
    )
    l_data.append(scatter)

  layout = go.Layout(
      xaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      yaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      # xaxis={'gridcolor': 'rgba(0,0,0,0)'},
      # yaxis={'gridcolor': 'rgba(0,0,0,0)'},

      # barmode='stack',
      margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
      # hovermode='x'
      dragmode='lasso',
      paper_bgcolor='rgba(0,0,0,0)',
      # plot_bgcolor='rgba(53,63,68,100)',
      plot_bgcolor='rgba(39,43,48,100)',
      # fig_bgcolor = 'rgba(0,0,0,0)',
      # zerolinecolor= 'rgba(0,0,0,0)'
  )

  figure_1 = go.Figure(data=l_data, layout=layout)

  return figure_1

def f_figure_2(_df, _x_selection, _y_selection, _qtd_cols_figure_2=6):

  #time_check
  global_start_time = time.time()
  start_time = time.time()




  qtd_selected_points = _x_selection

  _ini = 1
  _fin = len(qtd_selected_points)
  _mul = 1
  _len = (_fin-_ini)*_mul+1

  _x_columns = _qtd_cols_figure_2
  _x_qtd = math.ceil(_fin / _x_columns)
  _list_x = []


  for i in range(_x_qtd):
    if (i + 1) == _x_qtd:
      _list_x.append(np.linspace(1,(_fin - (i * _x_columns)), num=(_fin - (i * _x_columns))))
    else:
      _list_x.append(np.linspace(1,_x_columns, num=_x_columns))
  try:
    _list_x = np.concatenate(_list_x)
  except:
    _list_x =[]
  _temp = _fin - ((_x_qtd-1) * _x_columns)



  #time_check
  # print("--- %s seconds ---" % (time.time() - start_time))


  #time_check
  start_time = time.time()


  _y_columns = _qtd_cols_figure_2
  _y_qtd = math.ceil(_fin / _y_columns)
  _list_y = []
  for i in range(_y_qtd):
    if (i+1) == _y_qtd:
      _list_y.append(np.full(
          shape=_fin - (i * _y_columns),
          fill_value=i,
          dtype=np.int))
    else:
      _list_y.append(np.full(
          shape=_y_columns,
          fill_value=i,
          dtype=np.int))
  try:
    _list_y = np.concatenate(_list_y)
  except:
    _list_y = np.full(
          shape=_fin,
          fill_value=0,
          dtype=np.int)


  #time_check
  # print("--- %s seconds ---" % (time.time() - start_time))

  _data = {
      'x_2': _list_x,
      'y_2': _list_y,
      'label': _list_y
    }
  _df_graph2 = pd.DataFrame(_data)

  groups = _df_graph2.groupby('label')
  l_data = []
  i_count = 0

  # print("_df_graph2-----------------------", _df_graph2.head(20))


  #time_check
  start_time = time.time()


  ## Selected Points
  _selectedpoints = _df['manual_label'][
    (_df['x'].isin(_x_selection)) &
    (_df['y'].isin(_y_selection))
  ].index.values
  _temp = []
  _images_names = _selectedpoints.copy()
  for i in range(len(_selectedpoints)):
        _temp.append(_df.index.get_loc(_selectedpoints[i]))
  _selectedpoints = _temp
  _search = _df.iloc[_selectedpoints].index.values


  #time_check
  # print("--- %s seconds ---" % (time.time() - start_time))


  #time_check
  start_time = time.time()

  for idx, val in groups:
        scatter = go.Scatter(
            name=idx,
            x=val['x_2'],
            y=val['y_2'],
            text=val['y_2'],
            selectedpoints=_selectedpoints,
            customdata=_df[_df.index.isin(_search)].index.values.tolist(),
            textposition="top center",
            mode="markers",
            marker=dict(size=1, symbol="circle")
        )
        i_count = i_count+1
        l_data.append(scatter)

  #time_check
  # print("--- %s seconds ---" % (time.time() - start_time))



  #time_check
  # print("--- %s seconds Global---" % (time.time() - global_start_time))

  if len(qtd_selected_points) >= 100:
    _autosize = False
    _height = len(qtd_selected_points) * 20
  else:
    _autosize = True
    _height = None

  layout = go.Layout(
      autosize= _autosize,
      # bargap=0.15,
      # bargroupgap=0.1,
      # barmode='stack',
      # height=len(qtd_selected_points)*12,
      height=_height,
      # width=1200,
      # hovermode='x',
      # margin={'l': 300, 'r': 20, 'b': 75, 't': 125},
      xaxis={'gridcolor': 'rgba(0,0,0,0)', 'zerolinecolor': 'rgba(0,0,0,0)'},
      yaxis={'gridcolor': 'rgba(0,0,0,0)', 'zerolinecolor': 'rgba(0,0,0,0)'},
      dragmode = 'lasso',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      # rangeslider=dict(visible = True),
      images=create_images_layout(len(_selectedpoints), selectedpoints=_selectedpoints, images_names=_images_names)
  )

  figure_2 = go.Figure(data=l_data, layout=layout)

  print("len groups", len(groups))
  print("AQUI!!!!!", layout['height'])
  print("qtd points!!!!!", len(qtd_selected_points))
  return figure_2




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
        dcc.Graph(id="g1", style={"height": "50vh"}

        )
    ],
  )

  l_grafico_2 = html.Div(
    [
        # dcc.Graph(id="g2", style={"height": "50vh"})
        dcc.Graph(id="g2", style=dict(height='50vh',overflow='scroll')) #auto



    ], #style={"width": "100%", "border": "1px"}
  )

  l_stores = html.Div(
    [
      dcc.Store(id='memory1'),
      dcc.Store(id='memory2'),
      dcc.Store(id='store_df', storage_type='session'),
      dcc.Store(id='n_clicks_memory')
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
      [Output('g1', 'figure'),
      Output('g2', 'figure'),
      Output('memory1', 'data'),
      Output('memory2', 'data'),
      Output('n_clicks_memory', 'data'),
      Output('store_df', "data")
      ],
      [
      Input('g1', 'selectedData'),
      Input('g2', 'selectedData'),
      Input("setar-label", "n_clicks")
      ],
      [
      State('memory1', 'data'),
      State('memory2', 'data'),
      State('g1', 'figure'),
      State('g2', 'figure'),
      State("input_label", "value"),
      State("n_clicks_memory", "data"),
      State('store_df', "data")
      ]
  )
  def callback(
    # Inputs:
    selection1,
    selection2,
    n_clicks,
    # States:
    memory1,
    memory2,
    figure1,
    figure2,
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


    print("n_clicks", n_clicks)
    print("n_clicks_memory", n_clicks_memory)
    print("------------------")


    #---------------------------------------------------------------------------------------------------------------------

    if n_clicks == 0:
      #Simplesmente carregou o DataFrame para o Store (sotore_df)
      #A    ####################################
      if store_df is None:


        #time_check
        global_start_time = time.time()


        df_json = df.to_json()



        df['x'] = round(df['x'], 7)
        df['y'] = round(df['y'], 7)
        selectedData1, selectedData2 = [[], []]


        #time_check
        start_time = time.time()


        _x_selection, _y_selection = f_selection_check(_df = df,
                                    _selection_1 = selection1,
                                    _selection_2 = selection2,
                                    _memory_1 = memory1,
                                    _memory_2 = memory2,
                                    _figure2 = figure2,
                                    _qtd_cols_figure_2=6)

        #time_check
        print("--- %s seconds ---Callback f_selection_check" % (time.time() - start_time))


        #time_check
        start_time = time.time()

        figure_1 = f_figure_1(_df = df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)



        #time_check
        print("--- %s seconds ---Callback figure1" % (time.time() - start_time))



        #time_check
        start_time = time.time()



        figure_2 = f_figure_2(_df = df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              _qtd_cols_figure_2=6)


        #time_check
        print("--- %s seconds ---Callback figure2" % (time.time() - start_time))



        #time_check
        print("--- %s seconds ---Callback (A)" % (time.time() - global_start_time))

        return [figure_1,
                figure_2,
                selection1,
                selection2,
                n_clicks,
                df_json]

      #B    ####################################
      else:
        # Dataframe atualizado carregado:
        # print ("Aqui")


        #time_check
        global_start_time = time.time()


        _df = pd.read_json(store_df)

        _df['x'] = round(_df['x'], 7)
        _df['y'] = round(_df['y'], 7)


        #time_check
        start_time = time.time()



        _x_selection, _y_selection = f_selection_check(_df = _df,
                                    _selection_1 = selection1,
                                    _selection_2 = selection2,
                                    _memory_1 = memory1,
                                    _memory_2 = memory2,
                                    _figure2 = figure2,
                                    _qtd_cols_figure_2=6)

        #time_check
        print("--- %s seconds ---Callback f_selection_check" % (time.time() - start_time))


        #time_check
        start_time = time.time()




        figure_1 = f_figure_1(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection=_y_selection)

        #time_check
        print("--- %s seconds ---Callback figure1" % (time.time() - start_time))



        #time_check
        start_time = time.time()



        figure_2 = f_figure_2(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              _qtd_cols_figure_2=6)
        df_store_updated = _df.to_json()


        #time_check
        print("--- %s seconds ---Callback figure2" % (time.time() - start_time))




        #time_check
        print("--- %s seconds ---Callback (B)" % (time.time() - global_start_time))

        return [figure_1,
                figure_2,
                selection1,
                selection2,
                n_clicks_memory,
                df_store_updated]


    else:
      #C    #################################### Set Labels
      if n_clicks > n_clicks_memory:

        #time_check
        start_time = time.time()


        # Dataframe atualizado carregado:
        _df = pd.read_json(store_df)
        _df['x'] = round(_df['x'], 7)
        _df['y'] = round(_df['y'], 7)

        if selection1 == None and selection2 == None:
          df_store_updated = _df.to_json()
          n_clicks_memory = n_clicks_memory + 1
          return [figure1,figure2,memory1,memory2,n_clicks_memory,df_store_updated]
        else:
          None


        if input_value is None:
          input_value = '-'
        else:
          input_value = input_value

        _df['manual_label'][_df.index.isin(figure2['data'][0]['customdata'])] = input_value

        # df_store_updated
        df_store_updated = _df.to_json()
        n_clicks_memory = n_clicks_memory + 1


        selection1 = None
        selection2 = None

        _x_selection = _df['x'].values.tolist()
        _y_selection = _df['y'].values.tolist()

        #time_check
        start_time = time.time()


        figure_1 = f_figure_1(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection=_y_selection)


        #time_check
        print("--- %s seconds ---Callback figure1" % (time.time() - start_time))

        #time_check
        start_time = time.time()



        figure_2 = f_figure_2(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              _qtd_cols_figure_2=6)

        #time_check
        print("--- %s seconds ---Callback figure2" % (time.time() - start_time))



        print("AQUI!")
        #time_check
        print("--- %s seconds ---Callback (C)" % (time.time() - start_time))

        return [figure_1,
                figure_2,
                selection1,
                selection2,
                n_clicks_memory,
                df_store_updated]

      #D    ####################################
      else:


        #time_check
        global_start_time = time.time()





        # Dataframe atualizado carregado:
        print ("Aqui")
        _df = pd.read_json(store_df)
        _df['x'] = round(_df['x'], 7)
        _df['y'] = round(_df['y'], 7)


        #time_check
        start_time = time.time()


        _x_selection, _y_selection = f_selection_check(_df = _df,
                                    _selection_1 = selection1,
                                    _selection_2 = selection2,
                                    _memory_1 = memory1,
                                    _memory_2 = memory2,
                                    _figure2 = figure2,
                                    _qtd_cols_figure_2=6)


        #time_check
        print("--- %s seconds ---Callback f_selection_check" % (time.time() - start_time))


        #time_check
        start_time = time.time()


        figure_1 = f_figure_1(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection = _y_selection)

        #time_check
        print("--- %s seconds ---Callback figure1" % (time.time() - start_time))


        #time_check
        start_time = time.time()



        figure_2 = f_figure_2(_df = _df,
                              _x_selection = _x_selection,
                              _y_selection= _y_selection,
                              _qtd_cols_figure_2=6)
        df_store_updated = _df.to_json()

        #time_check
        print("--- %s seconds ---Callback figure2" % (time.time() - start_time))



        #time_check
        print("--- %s seconds ---Callback (GLOBAL D)" % (time.time() - global_start_time))

        return [figure_1,
                figure_2,
                selection1,
                selection2,
                n_clicks_memory,
                df_store_updated]

