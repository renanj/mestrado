import base64
import io
import pathlib

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
warnings.filterwarnings('ignore')
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

data_dict = {
    "mnist_3000": pd.read_csv(DATA_PATH.joinpath("mnist_3000_input.csv")),
    # "wikipedia_3000": pd.read_csv(DATA_PATH.joinpath("wikipedia_3000.csv")),
    # "twitter_3000": pd.read_csv(
    #     DATA_PATH.joinpath("twitter_3000.csv"), encoding="ISO-8859-1"
    # ),
}

# Import datasets here for running the Local version
IMAGE_DATASETS = "mnist_3000"
# WORD_EMBEDDINGS = ("wikipedia_3000", "twitter_3000")


with open(PATH.joinpath("demo_intro.md"), "r") as file:
    demo_intro_md = file.read()
with open(PATH.joinpath("demo_description.md"), "r") as file:
    demo_description_md = file.read()


def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64

# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")

def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )

def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )




def create_layout(app):
    ## Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "230%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.H3(
                                "Plankton Explorer",
                                className="header_title",
                                id="app-title",
                                style={'fontSize': 35}
                                # style={'text-align': 'left'}

                            ),
                            html.H6(
                                "Orientadora: Profa. Dra. Nina Hirata",
                                className="header_title",
                                id="app-professora",
                                style={'fontSize': 16, 'margin-bottom': '-4px', 'margin-top': '-4px'}
                            ),
                            html.H6(
                                "Aluno: Renan C. Jacomassi",
                                className="header_title",
                                id="app-aluno",
                                style={'fontSize': 16}
                            )

                        ],
                        className="nine columns header_title_container",
                    ),
                    html.Div(
                        [
                            # html.Img(
                            #     src=app.get_asset_url("usp_logo.png"),
                            #     className="logo",
                            #     id="plotly-usp",
                            #     style={"max-width": "100%", 'margin-left': '15px', 'margin-right': '16px'},
                            #     # style={'float':'right', 'horizontal-align':'left', 'vertical-align':'left'},
                            # ),
                            html.Img(
                                src=app.get_asset_url("ime.png"),
                                className="logo",
                                id="plotly-ime",
                                style={"max-width": "200%", 'margin-left': '15px', 'margin-right': '16px'},
                                # style={'float':'right', 'horizontal-align':'left', 'vertical-align':'left'},
                            ),
                        ],
                        className="three columns header_img",
                    ),
                ],
            ),



            # Demo Description
            html.Div(
                className="row background",
                id="demo-explanation",
                style={"padding": "50px 45px"},
                children=[
                    html.Div(
                        id="description-text", children=dcc.Markdown(demo_intro_md)
                    ),
                    html.Div(
                        html.Button(id="learn-more-button", children=["Saiba Mais"]),
                    )
                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[

                            Card(
                                [
                                    dcc.Dropdown(
                                        id="dropdown-dataset",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "MNIST Digits",
                                                "value": "mnist_3000",
                                            },
                                        ],
                                        placeholder="Select a dataset",
                                        value="mnist_3000",
                                    ),
                                    NamedSlider(
                                        name="Número de Iterações",
                                        short="iterations",
                                        min=250,
                                        max=1000,
                                        step=None,
                                        val=500,
                                        marks={
                                            i: str(i) for i in [250, 500, 750, 1000]
                                        },
                                    ),
                                    NamedSlider(
                                        name="Perplexity",
                                        short="perplexity",
                                        min=3,
                                        max=100,
                                        step=None,
                                        val=30,
                                        marks={i: str(i) for i in [3, 10, 30, 50, 100]},
                                    ),
                                    NamedSlider(
                                        name="Initial PCA Dimensions",
                                        short="pca-dimension",
                                        min=25,
                                        max=100,
                                        step=None,
                                        val=50,
                                        marks={i: str(i) for i in [25, 50, 100]},
                                    ),
                                    NamedSlider(
                                        name="Learning Rate",
                                        short="learning-rate",
                                        min=10,
                                        max=200,
                                        step=None,
                                        val=100,
                                        marks={i: str(i) for i in [10, 50, 100, 200]},
                                    ),
                                    dcc.Dropdown(
                                        id="dropdown-labels",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "Labels Corretos",
                                                "value": "correct_labels",
                                            },
                                            {
                                                "label": "Labels Usuário",
                                                "value": "manual_labels",
                                            },
                                        ],
                                        placeholder="Labels Selection",
                                        value="manual_labels",
                                    ),
                                    # dcc.Store(id='store_df', storage_type='local'),
                                    dcc.Store(id='store_df', storage_type='session'),

                                    dcc.Input(id="input_label", type="number", placeholder=None, min=0, max=9, step=1,),
                                    html.Button(id="ativar-grafico", children=["Setar Label"]),
                                            dcc.Loading(html.Pre(id="click-data2", className="info__container", style={'display': 'none'}), type="dot",),

                                    html.Div(
                                        id="div-wordemb-controls",
                                        style={"display": "none"},
                                        children=[
                                            NamedInlineRadioItems(
                                                name="Display Mode",
                                                short="wordemb-display-mode",
                                                options=[
                                                    {
                                                        "label": " Regular",
                                                        "value": "regular",
                                                    },
                                                    {
                                                        "label": " Top-100 Neighbors",
                                                        "value": "neighbors",
                                                    },
                                                ],
                                                val="regular",
                                            ),
                                            dcc.Dropdown(
                                                id="dropdown-word-selected",
                                                placeholder="Select word to display its neighbors",
                                                style={"background-color": "#f2f3f4"},
                                            ),
                                        ],
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", style={"height": "50vh"})
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
                        children=[
                            Card(
                                style={"padding": "5px"},
                                children=[
                                    html.Div(
                                        id="div-plot-click-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Div(id="div-plot-click-image"),
                                    # html.Div(id="div-plot-click-wordemb"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

def demo_callbacks(app):
    def generate_figure_image(groups, layout, n_dim=3):
        data = []

        if n_dim == 3:
            for idx, val in groups:
                scatter = go.Scatter3d(
                    name=idx,
                    x=val["x"],
                    y=val["y"],
                    z=val["z"],
                    text=[idx for _ in range(val["x"].shape[0])],
                    textposition="top center",
                    mode="markers",
                    marker=dict(size=3, symbol="circle"),
                )
                data.append(scatter)

            figure = go.Figure(data=data, layout=layout)

            return figure
        else:
            for idx, val in groups:
                scatter = go.Scatter(
                    name=idx,
                    x=val["x"],
                    y=val["y"],
                    text=[idx for _ in range(val["x"].shape[0])],
                    textposition="top center",
                    mode="markers",
                    marker=dict(size=5, symbol="circle"),
                )
                data.append(scatter)

            figure = go.Figure(data=data, layout=layout)

            return figure


    # Scatter Plot of the t-SNE datasets
    def generate_figure_word_vec(
        embedding_df, layout, wordemb_display_mode, selected_word, dataset):

        try:
            # Regular displays the full scatter plot with only circles
            if wordemb_display_mode == "regular":
                plot_mode = "markers"

            # Nearest Neighbors displays only the 200 nearest neighbors of the selected_word, in text rather than circles
            elif wordemb_display_mode == "neighbors":
                if not selected_word:
                    return go.Figure()

                plot_mode = "text"

                # Get the nearest neighbors indices using Euclidean distance
                vector = data_dict[dataset].set_index("0")
                selected_vec = vector.loc[selected_word]

                def compare_pd(vector):
                    return spatial_distance.euclidean(vector, selected_vec)

                # vector.apply takes compare_pd function as the first argument
                distance_map = vector.apply(compare_pd, axis=1)
                neighbors_idx = distance_map.sort_values()[:100].index

                # Select those neighbors from the embedding_df
                embedding_df = embedding_df.loc[neighbors_idx]

            scatter = go.Scatter3d(
                name=str(embedding_df.index),
                x=embedding_df["x"],
                y=embedding_df["y"],
                z=embedding_df["z"],
                text=embedding_df.index,
                textposition="middle center",
                showlegend=False,
                mode=plot_mode,
                marker=dict(size=3, color="#3266c1", symbol="circle"),
            )

            figure = go.Figure(data=[scatter], layout=layout)

            return figure
        except KeyError as error:
            print(error)
            raise PreventUpdate

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






    # Callback function for the learn-more button
    @app.callback(
        [
            Output("description-text", "children"),
            Output("learn-more-button", "children"),
        ],
        [Input("learn-more-button", "n_clicks")],
        )
    def learn_more(n_clicks):
        # If clicked odd times, the instructions will show; else (even times), only the header will show
        if n_clicks == None:
            n_clicks = 0
        if (n_clicks % 2) == 1:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(demo_description_md)],
                ),
                "Close",
            )
        else:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(demo_intro_md)],
                ),
                "Saiba Mais",
            )


    @app.callback(
        [Output("store_df", "data"), Output("graph-3d-plot-tsne", "figure")],
        [
            # Input("ativar-grafico", "n_clicks")
            Input("dropdown-dataset", "value"),
            Input("slider-iterations", "value"),
            Input("slider-perplexity", "value"),
            Input("slider-pca-dimension", "value"),
            Input("slider-learning-rate", "value"),
            Input("dropdown-labels", "value"),
            Input("ativar-grafico", "n_clicks"),
            ],
        [State('store_df', 'data'), State("ativar-grafico", "n_clicks"),
            State("graph-3d-plot-tsne", "clickData"),
            State("graph-3d-plot-tsne", "selectedData"),
            State("input_label", "value"),
        ]
        )
    def save_store(
        dataset,
        iterations,
        perplexity,
        pca_dim,
        learning_rate,
        drop_labels,
        n_clicks,
        store_data,
        n_clicks_check,
        clickData,
        selectedData,
        input_value,
        ):



        # print("Store df______________________")
        # print("dataset = ", dataset)
        # print("type = ", type(dataset))
        if n_clicks is None:
            # print ("interations", iterations)
            if dataset:
                # print ("dataset", dataset)
                path = "demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}"
                # print ("aqui :",  path)
                try:

                    data_url = [
                        "demo_embeddings",
                        str(dataset),
                        "iterations_" + str(iterations),
                        "perplexity_" + str(perplexity),
                        "pca_" + str(pca_dim),
                        "learning_rate_" + str(learning_rate),
                        "data.csv",
                    ]
                    full_path = PATH.joinpath(*data_url)
                    # print ("FULL PATH = ", full_path)
                    embedding_df = pd.read_csv(
                        # full_path, index_col=0, encoding="ISO-8859-1"
                        full_path, encoding="ISO-8859-1"
                    )

                    embedding_df['x'] = round(embedding_df['x'], 7)
                    embedding_df['y'] = round(embedding_df['y'], 7)

                    # print ("XXXAAAAAAA :", embedding_df.iloc[179:180,:])
                    print ("x :", embedding_df.loc[179:180,"x"])
                    # print ("xxxx: ", embedding_df.head())
                    df_json = embedding_df.to_json()
                    # print ("ESSE :", df_json)
                    # print ("JSON: ", df_json[:1000])
                    # return df_json
                    # print (df_json)
                    # print (b['x'])
                    # print ('x new = ', b['x']['179'])
                    # CHART
                    # Plot layout
                    axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)


                    n_dim = 2
                    if n_dim == 3:
                        layout = go.Layout(
                            margin=dict(l=0, r=0, b=0, t=0),
                            scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
                        )
                    else:
                        layout = go.Layout(
                            margin=dict(l=0, r=0, b=0, t=0),
                            scene=dict(xaxis=axes, yaxis=axes),
                        )


                    # print("!!!!!! aahh")
                    if drop_labels == 'correct_labels':
                        # embedding_df["label"] = embedding_df.index
                        embedding_df["label"] = embedding_df['correct_label']
                    else:
                        embedding_df["label"] = embedding_df['mannual_label']
                        # print ('store_data     ===== ', store_data)
                        # embedding_df["label"] = store_data['mannual_label']



                    groups = embedding_df.groupby("label")
                    figure = generate_figure_image(groups, layout, n_dim=2)

                    return df_json, figure


                except FileNotFoundError as error:
                    print(
                        error,
                        "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                    )
                    return None, None

        else:
            embedding_df = pd.read_json(store_data)
            embedding_df['x'] = round(embedding_df['x'], 7)
            embedding_df['y'] = round(embedding_df['y'], 7)
            print ("x else :", embedding_df.loc[179:180,"x"])
            # print ("STOE_DATA", store_data)
            # print("AQUIII: ", count_button)
            # print("ctx.triggered", ctx.triggered[0]['value'])
            # print ("n_clicks" , n_clicks)
                # print ("entrou!")
                # print ("n_clicks :", n_clicks)
                # print ("n_clicks_check :", n_clicks_check)
                # if n_clicks >= n_clicks_check:

            array_points = []
            _selected_data = selectedData
            print ( "AQUIIII 2:", _selected_data)
            if _selected_data is None:
                click_point_np = np.array(
                    [clickData["points"][0][i] for i in ["x", "y"]]
                ).astype(np.float64)
                # print("click_point_np = ", click_point_np)
                array_points.append(click_point_np)

            else:
                temp_df = pd.DataFrame(_selected_data["points"])
                print ( "XELECETTTT  = ", temp_df)
                # _x = round(temp_df['x'],10).to_list()
                # _y = round(temp_df['y'],10).to_list()
                _x = temp_df['x'].values.tolist() #_x = temp_df['x'].to_list()
                _y = temp_df['y'].values.tolist() #_y = temp_df['y'].to_list()
                print ("_x 222", _x)
                print ("_y 222", _y)
                # print ( "AQUIIII:", _selected_data['points'])
                # print ("_x", _x)
                # for i in range(len(_selected_data['points'])):
                #     x = _selected_data['points'][i]['x']
                #     y = _selected_data['points'][i]['y']
                #     array_points.append([x, y])


                # print (embedding_df.head())

                embedding_df['mannual_label'][
                    (embedding_df['x'].isin(_x)) &
                    (embedding_df['y'].isin(_y))

                ] = input_value
                print ("x ---->", embedding_df['x'][embedding_df.index == 1953])
                print ("y ---->", embedding_df['y'][embedding_df.index == 1953])

                print ("______ VER: ", embedding_df[
                    (embedding_df['x'].isin(_x)) &
                    (embedding_df['y'].isin(_y))

                ])
                embedding_df.to_csv('AQUI.csv')

            # for i in range(len(array_points)):
            #     bool_mask_click = (
            #         embedding_df.loc[:, "x":"y"].eq(array_points[i]).all(axis=1)
            #     )

                # Retrieve the index of the point clicked, given it is present in the set
                # if bool_mask_click.any():
                #     clicked_idx = embedding_df[bool_mask_click].index[0]
                #     embedding_df['mannual_label'][embedding_df.index == clicked_idx] = input_value


            data_updated = embedding_df.to_json()




            # CHART
            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)


            n_dim = 2
            if n_dim == 3:
                layout = go.Layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
                )
            else:
                layout = go.Layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene=dict(xaxis=axes, yaxis=axes),
                )


            # print("!!!!!! aahh")
            if drop_labels == 'correct_labels':
                # embedding_df["label"] = embedding_df.index
                embedding_df["label"] = embedding_df['correct_label']
            else:
                embedding_df["label"] = embedding_df['mannual_label']
                # print ('store_data     ===== ', store_data)
                # embedding_df["label"] = store_data['mannual_label']



            groups = embedding_df.groupby("label")
            figure = generate_figure_image(groups, layout, n_dim=2)



            # if n_clicks > n_clicks_check:
            return data_updated, figure
            # else:
            #     return store_data, figure



    @app.callback(Output("click-data2", "children"),
                # [Input("graph-plot-tsne", "config")],
                [Input("graph-3d-plot-tsne", "selectedData")],

                #  [State("graph-plot-tsne", "loading_state")],
                )
    def display_click_data(data):
        return json.dumps(data, indent=4)



    @app.callback(
        Output("div-plot-click-image", "children"),
        [
            Input("graph-3d-plot-tsne", "clickData"),
            Input("graph-3d-plot-tsne", "selectedData"),
            Input("dropdown-dataset", "value"),
            # Input("slider-iterations", "value"),
            # Input("slider-perplexity", "value"),
            # Input("slider-pca-dimension", "value"),
            # Input("slider-learning-rate", "value"),
        ],
        [State('store_df', 'data')])
    def display_click_image(
        clickData, selectedData, dataset, store_data):


        if clickData or selectedData:
            # Load the same dataset as the one displayed

            try:
                embedding_df = pd.read_json(store_data)
                embedding_df['x'] = round(embedding_df['x'], 7)
                embedding_df['y'] = round(embedding_df['y'], 7)

            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                )
                return None
            # n_dim = 2
            # if n_dim == 3:
                # Convert the point clicked into float64 numpy array
                click_point_np = np.array(
                    [clickData["points"][0][i] for i in ["x", "y", "z"]]
                ).astype(np.float64)
                # Create a boolean mask of the point clicked, truth value exists at only one row
                bool_mask_click = (
                    embedding_df.loc[:, "x":"z"].eq(click_point_np).all(axis=1)
                )
            # else:
            # print ("selectedData XXXXX = ", selectedData)
            # _selected_data = json.loads(selectedData)
            _selected_data = selectedData
            # array_1 = [25.328316, -17.343037]
            # array_2 = [29.725433000000002, 16.650507]
            # array_points = [array_1, array_2]

            array_points = []
            if _selected_data is None:
                click_point_np = np.array(
                    [clickData["points"][0][i] for i in ["x", "y"]]
                ).astype(np.float64)
                # print("click_point_np = ", click_point_np)

                array_points.append(click_point_np)
            else:
                ###### Criar o array
                for i in range(len(_selected_data['points'])):
                    x = _selected_data['points'][i]['x']
                    y = _selected_data['points'][i]['y']
                    array_points.append([x, y])


            print ("FINAAAAL  = ", _selected_data)
            images_div = []
            for i in range(len(array_points)):

                bool_mask_click = (
                    embedding_df.loc[:, "x":"y"].eq(array_points[i]).all(axis=1)
                )
                # print ("bool_mask_click = ", bool_mask_click)



                # Retrieve the index of the point clicked, given it is present in the set
                if bool_mask_click.any():
                    clicked_idx = embedding_df[bool_mask_click].index[0]
                    # print ("clicked_idx = ", clicked_idx)

                    # Retrieve the image corresponding to the index
                    image_vector = data_dict[dataset].iloc[clicked_idx]
                    if dataset == "cifar_gray_3000":
                        image_np = image_vector.values.reshape(32, 32).astype(np.float64)
                    else:
                        image_np = image_vector.values.reshape(28, 28).astype(np.float64)

                    # Encode image into base 64
                    image_b64 = numpy_to_b64(image_np)
                    images_div.append(generate_thumbnail(image_b64))
                    # print ("images_div  === ", images_div)

            return images_div
        return None









    @app.callback(
        Output("div-plot-click-message", "children"),
        [Input("graph-3d-plot-tsne", "clickData")],
        # Input("dropdown-dataset", "value")],
)
    def display_click_message(clickData):
        dataset = 'mnist_3000'
        # Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        # if dataset in IMAGE_DATASETS:
        if clickData:
            return "Image Selected"
        else:
            return "Click a data point on the scatter plot to display its corresponding image."