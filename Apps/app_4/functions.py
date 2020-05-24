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
# import utils

import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance

import warnings

import dash_bootstrap_components as dbc


import datetime
import math
from PIL import Image
import json







def create_images_layout(qtd_pontos=0, _qtd_cols_figure_2=6, selectedpoints=[1], images_names=None, _list_x=None, _list_y=None):

  _data = {
      'x': _list_x,
      'y': _list_y,
      'sizex': 0.5,
      'sizey': 0.5,
      'images': images_names,
      'source': images_names,
      'folder': 'data/',
      'png': '.png',
      'opacity': 1,
      'xanchor': 'center',
      'yanchor': 'middle',
      'xref': 'x',
      'yref': 'y'
  }



  _df_dict = pd.DataFrame(_data)
  _df_dict = _df_dict[['x', 'y', 'sizex', 'sizey', 'source', 'opacity', 'xanchor', 'yanchor', 'xref', 'yref']]
  _df_dict = _df_dict.to_dict('records')


  return _df_dict


def f_selection_check(_df,_list_selection, _list_memory, _list_figure):

  _x_selection = None
  _y_selection = None

  # Selection_0 vs Memory_0
  if _list_selection[0] != _list_memory[0]:
    try:

      temp_df = pd.DataFrame(_list_selection[0]["points"])

      _x_selection = temp_df['x'].values.tolist()
      _y_selection = temp_df['y'].values.tolist()
    except:
      temp_df = None
  else:
    None


  # Selection_1 vs Memory_1
  if _list_selection[1] != _list_memory[1]:
    try:
      _list_figure[1] = _list_figure[1]['data']
      _text = []
      _selected_points = []
      _custom_points = []
      i_count = 0
      for i in range(len(_list_figure[1])):
        _selected_points.append(_list_figure[1][i]['selectedpoints'])
        _custom_points.append(_list_figure[1][i]['customdata'])
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
    _x_selection = _df['x'].values.tolist()
  else:
    None

  if _y_selection == None:
    _y_selection = _df['y'].values.tolist()
  else:
    None

  return _x_selection, _y_selection

def f_figure_0(_df, _x_selection, _y_selection):

  l_data = []
  groups = _df.groupby('manual_label')

  for idx, val in groups:
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
        customdata=_selectedpoints,
        textposition="top center",
        mode="markers",
        marker=dict(size=20, symbol="circle")
    )
    l_data.append(scatter)

  layout = go.Layout(
      xaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      yaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
      dragmode='lasso',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(39,43,48,100)',
  )

  figure_0 = go.Figure(data=l_data, layout=layout)
  return figure_0

def f_figure_1(_df, _x_selection, _y_selection, images_data=None):

  qtd_selected_points = _x_selection


  _list_x = list(np.linspace(1, 6, 6))
  _list_x = _list_x * math.ceil(len(qtd_selected_points) / 6)
  _list_x = _list_x[0:len(qtd_selected_points)]


  _list_y = np.linspace(0,math.ceil(len(qtd_selected_points) / 6)-1,math.ceil(len(qtd_selected_points) / 6)).reshape(math.ceil(len(qtd_selected_points) / 6),1)
  _list_y = np.concatenate((_list_y,_list_y,_list_y,_list_y,_list_y,_list_y), axis=1)
  _list_y = list(_list_y.flat)
  _list_y = _list_y[0:len(qtd_selected_points)]



  _data = {'x_2': _list_x, 'y_2': _list_y, 'label': "-"}
  _df_graph2 = pd.DataFrame(_data)


  groups = _df_graph2.groupby('label')
  l_data = []
  i_count = 0



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


  if len(qtd_selected_points) >= 100:
    _autosize = False
    _height = len(qtd_selected_points) * 20
  else:
    _autosize = True
    _height = None


  # reduzir a lista de imagens para a seleção
  try:
    indices = _images_names
    sublist = []
    print("indices", indices)
    for i in indices:
        sublist.append(images_data[i])
  except:
    sublist = images_data


  print("AQUI!")
  print("_selectedpoints", len(_selectedpoints))
  print("sublist", len(sublist))
  print("_list_x", len(_list_x))
  print("_list_y", len(_list_y))
  layout = go.Layout(
      autosize= _autosize,
      height=_height,
      xaxis={'gridcolor': 'rgba(0,0,0,0)', 'zerolinecolor': 'rgba(0,0,0,0)'},
      yaxis={'gridcolor': 'rgba(0,0,0,0)', 'zerolinecolor': 'rgba(0,0,0,0)'},
      dragmode = 'lasso',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      images=create_images_layout(len(_selectedpoints), selectedpoints=_selectedpoints, images_names=sublist, _list_x=_list_x, _list_y=_list_y)
  )

  figure_1 = go.Figure(data=l_data, layout=layout)

  return figure_1




def f_figure_3(_df, _x_selection, _y_selection):

  l_data = []
  groups = _df.groupby('manual_label')

  for idx, val in groups:
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
        x=val["x3"],
        y=val["y3"],
        text=val['manual_label'],
        selectedpoints=_selectedpoints,
        customdata=_selectedpoints,
        textposition="top center",
        mode="markers",
        marker=dict(size=20, symbol="circle")
    )
    l_data.append(scatter)

  layout = go.Layout(
      xaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      yaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
      dragmode='lasso',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(39,43,48,100)',
  )

  figure_0 = go.Figure(data=l_data, layout=layout)
  return figure_0




def f_figure_4(_df, _x_selection, _y_selection):

  l_data = []
  groups = _df.groupby('manual_label')

  for idx, val in groups:
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
        x=val["x4"],
        y=val["y4"],
        text=val['manual_label'],
        selectedpoints=_selectedpoints,
        customdata=_selectedpoints,
        textposition="top center",
        mode="markers",
        marker=dict(size=20, symbol="circle")
    )
    l_data.append(scatter)

  layout = go.Layout(
      xaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      yaxis={'range': [-1, 1.1], 'autorange': True, 'gridcolor': 'rgba(0,0,0,0)','zeroline': False, 'showgrid': False},
      margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
      dragmode='lasso',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(39,43,48,100)',
  )

  figure_0 = go.Figure(data=l_data, layout=layout)
  return figure_0



