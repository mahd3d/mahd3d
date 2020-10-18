import json
from typing import Union, Optional

import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from plotly import express as px

from src.utils import timeit


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
            cubes[layer_index] += [(v["x"], v["y"], 0.0)]
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
                # color="rgb(255,0,0)",
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

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # proportional aspect ratio with data, alternatively cube or manual (see below)
    fig.update_layout(scene_aspectmode="data")

    fig.show()


def plot_3d_Grouped_json(
    file: str = "data/json/objects.example.json",
    objects: Optional[dict] = None,
    cubeEdgeLength: Optional[int] = 1,
    global_minX: Optional[int] = 0,
    global_maxX: Optional[int] = 20,
    global_minY: Optional[int] = -20,
    global_maxY: Optional[int] = 50,
    minimumHeight: Optional[float] = 1.2

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

    ################################################################################
    # START COMBINE THE CUBES

    groupedCubes: list = []

    for xi in range (math.floor(global_minX / cubeEdgeLength), math.ceil(global_maxX / cubeEdgeLength)):       # math.floor(global_minX)
        for yi in range (math.floor(global_minY / cubeEdgeLength), math.ceil(global_maxY / cubeEdgeLength)):
            square: list = []

            # iterate over all cubes and group them vertically
            for cube in enumerate(cubes):

                xj = cube[1][0][0]
                yj = cube[1][0][1]

                if(((xj >= (xi * cubeEdgeLength)) & (xj < ((xi + 1) * cubeEdgeLength))) & ((yj >= (yi * cubeEdgeLength)) & (yj < ((yi + 1) * cubeEdgeLength)))):
                    square += [cube[1]]   # drop the index, just take the data

            if(len(square) != 0):
                # combine the vertical set of cubes into a single cube
                minZ = 1
                maxZ = 0

                for s in enumerate(square):
                    currentZ_lower = s[1][0][2]
                    currentZ_upper = s[1][4][2]

                    if currentZ_lower < minZ:
                        minZ = currentZ_lower

                    if currentZ_upper > maxZ:
                        maxZ = currentZ_upper

                newCube = list(s[1].copy())

                for k_ in range(4):
                    lst = list(newCube[k_])
                    lst[2] = minZ
                    newCube[k_] = tuple(lst)

                for k__ in range(4,8):

                    lst = list(newCube[k__])
                    lst[2] = maxZ
                    newCube[k__] = tuple(lst)

                groupedCubes.append(newCube)

    print("groupedCubes finished...")

    # END COMBINE THE CUBES
    ################################################################################
    # remove all vertical boxes that are not high enough (smaller than the minimal limit)

    minimumHeight = 1.2

    selectedTallCubes: list = []

    for tallCube in enumerate(groupedCubes):

        if(tallCube[1][4][2] >= minimumHeight):
            selectedTallCubes.append(tallCube[1])

    ################################################################################
    # START COMBINE THE BUILDINGS
    # for now just create rectangles (otherwise we could also go for polygons, but that would break the curreny flow on the website)

    #


    # END COMBINE THE BUILDINGS
    ################################################################################

    # fixed list, do not change or cubes will look weird
    i = [0, 3, 4, 7, 0, 6, 1, 7, 0, 5, 2, 7]
    j = [1, 2, 5, 6, 2, 4, 3, 5, 4, 1, 6, 3]
    k = [3, 0, 7, 4, 6, 0, 7, 1, 5, 0, 7, 2]

    meshes: list = []
    for index, cube in enumerate(selectedTallCubes):
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

    # proportional aspect ratio with data, alternatively cube or manual (see below)
    fig.update_layout(scene_aspectmode="data")

    fig.show()


def plot_histograms(
    points: pd.DataFrame,
) -> None:
    for col in points:
        if col not in ["x", "y", "z"]:
            continue
        fig = px.histogram(points, x=col)
        fig.show()
