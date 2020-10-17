import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from typing import Union, Tuple, Optional

import json

from utils import timeit

@timeit
def plot_3d(
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
    text: Union[None, pd.Series, pd.Index],
    color: [None, pd.Series],
) -> None:
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                text=x.index if text is None else text,
                mode="markers",
                marker=dict(
                    size=5,
                    color=z
                    if color is None
                    else color,  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                ),
            )
        ]
    )

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # proportional aspect ratio with data, alternatively cube or manual (see below)
    fig.update_layout(scene_aspectmode="data")
    # fig.update_layout(scene_aspectratio=dict(x=d_x/100, y=d_y/100, z=d_z/100))
    fig.show()


def plot_3d_json(
    file: str = "data/json/objects.example.json",
    objects: Optional[dict] = None,
) -> None:
    if objects is None:
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = objects

    cubes: list = []
    for layer_index, layer_value in enumerate(data["layers"]):
        cubes += [[]]
        for i, v in enumerate(layer_value["points"]):
            cubes[layer_index] += [(v["x"], v["y"], 0.1)]
            cubes[layer_index] += [(v["x"], v["y"], float(layer_value["height"]))]
            cubes[layer_index].sort(key=lambda _x: (_x[2], _x[1], _x[0]))

    # fixed list, do not change or cubes will look weird
    i = [0, 3, 4, 7, 0, 6, 1, 7, 0, 5, 2, 7]
    j = [1, 2, 5, 6, 2, 4, 3, 5, 4, 1, 6, 3]
    k = [3, 0, 7, 4, 6, 0, 7, 1, 5, 0, 7, 2]

    meshes: list = []
    for index, cube in enumerate(cubes):
        x = [i_[0] for i_ in cube]
        y = [i_[1] for i_ in cube]
        z = [i_[2] for i_ in cube]

        meshes += [
            go.Mesh3d(
                # 8 vertices of a cube
                x=x,
                y=y,
                z=z,
                # colorbar_title='z',
                colorscale=[[0, "gold"], [0.5, "mediumturquoise"], [1, "magenta"]],
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity=np.linspace(0, 1, 8, endpoint=True),
                # i, j and k give the vertices of triangles
                i=i,
                j=j,
                k=k,
                name="y",
                showscale=True,
            )
        ]

    fig = go.Figure(data=meshes)

    fig.show()
    pass
