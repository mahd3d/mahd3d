import uuid
import math

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pandas as pd
import itertools
import json
from DataRotation import correctlyRotateDataFrame
from LoadE57 import load_e57
from Plots import plot_3d, plot_3d_json, plot_3d_Grouped_json

from utils import timeit

global_minX = 0
global_maxX = 0
global_minY = 0
global_maxY = 0

@timeit
def get_points(
    data: dict,
    step: int = 1000,
) -> pd.DataFrame:
    x = data["cartesianX"][::step]
    y = data["cartesianY"][::step]
    z = data["cartesianZ"][::step]
    r = data["colorRed"][::step]
    g = data["colorGreen"][::step]
    b = data["colorBlue"][::step]
    i = data["intensity"][::step]

    points: pd.DataFrame = pd.DataFrame(
        data={"x": x, "y": y, "z": z, "r": r, "g": g, "b": b, "i": i}
    )

    points["rgba"] = points.apply(compute_rgba, axis=1)

    return points


@timeit
def get_points_with_computed(
    points: pd.DataFrame,
) -> pd.DataFrame:
    points["roof"] = False
    points.loc[points["z"] > 3.4, ["roof"]] = True
    points["floor"] = False
    points.loc[points["z"] < -1.5, ["floor"]] = True

    return points


def compute_rgba(
    row,
) -> str:
    if "a" not in row:
        row["a"] = 1.0
    return f'rgba({row["r"]}, {row["g"]}, {row["b"]}, {row["a"]:.2f})'


def get_squares(
    p: pd.DataFrame,
) -> pd.DataFrame:
    f = correctlyRotateDataFrame(p.copy())

    f["count"] = 0

    grid: int = 1
    counts = []
    for x, y in itertools.product(
        np.arange(min(f["x"]), max(f["x"]), grid),
        np.arange(min(f["y"]), max(f["y"]), grid),
    ):
        overscan = 0.0
        dots = f[
            (f["x"] >= x - overscan)
            & (f["x"] < x + 1 + overscan)
            & (f["y"] >= y - overscan)
            & (f["y"] < y + 1 + overscan)
        ]
        count = len(dots)
        counts += [count]
        f.loc[
            (f["x"] >= x) & (f["x"] < x + 1) & (f["y"] >= y) & (f["y"] < y + 1), "count"
        ] = count
        # print(f"{x}, {y}: {count}")

    f.loc[f["count"] > 20, "count"] = 20
    f = f[f["count"] > 2]

    return f


@timeit
def get_cubes(
    p: pd.DataFrame,
    grid: float = 1.0,
) -> pd.DataFrame:
    v = correctlyRotateDataFrame(p.copy())

    v["count"] = 0

    counts = pd.DataFrame(
        data={
            "x": [],
            "y": [],
            "z": [],
            "r": [],
            "g": [],
            "b": [],
            "c": [],
        }
    )

    global global_min
    global global_maxX
    global global_minY
    global global_maxY
    global_minX = min(v['x'])
    global_maxX = max(v['x'])
    global_minY = min(v['y'])
    global_maxY = max(v['y'])

    for x, y, z in itertools.product(
        np.arange(min(v["x"]), max(v["x"]), grid),
        np.arange(min(v["y"]), max(v["y"]), grid),
        np.arange(min(v["z"]), max(v["z"]), grid),
    ):
        overscan = 0.0
        dots = v[
            (v["x"] >= x - overscan)
            & (v["x"] < x + grid + overscan)
            & (v["y"] >= y - overscan)
            & (v["y"] < y + grid + overscan)
            & (v["z"] >= z - overscan)
            & (v["z"] < z + grid + overscan)
        ]
        count = len(dots)
        counts = counts.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "r": dots["r"].mean(),
                "g": dots["g"].mean(),
                "b": dots["b"].mean(),
                "c": count,
            },
            ignore_index=True,
        )
        #        counts += [count]
        v.loc[
            (v["x"] >= x)
            & (v["x"] < x + grid)
            & (v["y"] >= y)
            & (v["y"] < y + grid)
            & (v["z"] >= z)
            & (v["z"] < z + grid),
            "count",
        ] = count
        # print(f"{x}, {y}: {count}")

    v.loc[v["count"] > 20, "count"] = 20
    v = v[v["count"] > 5]

    return counts


def main() -> None:
    # Detail size, smaller is more detailed but slower
    # 1000 is recommended for displaying with Plotly, 300 is the minimum
    step: int = 3000
    filename: str = f"data/computed/points_{step}.v2.df.feather"
    minimumHeightOfObstacle: float = 1.2

    try:
        print(
            f"Trying to use existing points with step size of {step} from saved points file."
        )
        p = pd.read_feather(filename)
    except FileNotFoundError:
        print(f"Couldn't get points with step size of {step}, loading raw data.")
        data, header = load_e57()
        p: pd.DataFrame = get_points(data, step)
        p.to_feather(filename)
    finally:
        print("p loaded")

    # plot_histograms(p)

    p["0"] = 0
    p = get_points_with_computed(p)

    mid = p[~p["roof"] & ~p["floor"]]
    # mid = p[~p["roof"]]
    # points = points[(points["z"] >= -1.5) & (points["z"] <= -1.0)]
    # points["a"] = points.apply(compute_alpha, axis=1)

    plot_3d(
        x=mid["x"],
        y=mid["y"],
        z=mid["z"],
        text=mid.index,
        color=mid["rgba"],
    )
    #
    # f = get_squares(p[p["floor"]])
    #
    # plot_3d(
    #     x=f["x"],
    #     y=f["y"],
    #     z=f["0"],
    #     text=f["count"],
    #     color=f["count"],
    # )

    grid: float = 1.0
    Z_adjustment = 2.4995  # hardcoded floor level, not adjusted for tilt

    v = get_cubes(p[~p["roof"] & ~p["floor"]], grid)
    v = v[v["c"] > 5]

    layers = []
    all_shapes = []
    for index, row in v.iterrows():
        shape_id = str(uuid.uuid4())
        layer = {
            "points": [
                {"x": row["x"], "y": row["y"], "id": 1},
                {"x": row["x"] + grid, "y": row["y"], "id": 2},
                {"x": row["x"] + grid, "y": row["y"] + grid, "id": 3},
                {"x": row["x"], "y": row["y"] + grid, "id": 4},
            ],
            "height": row["z"] + Z_adjustment, 
            "shape_type": "obstacle",
            "shapeId": shape_id,
        }
        layers += [layer]
        shape = {
            "userInput":    round(row["z"]*10)/10 + Z_adjustment,
            "x":            row["x"]*20,
            "y":            row["y"]*20,
            "width":        grid*20,
            "height":       grid*20,
            "rotation":     0,
            "sId":          shape_id,
            "sType":        "Rect",
            "obstacleType": "obstacle",
            "stroke":       "red",
            "strokeWidth":  1,
            "cursor":       "pointer",
            "fill":         "#808080",
            "opacity":      0.6
        }
        all_shapes += [shape]

    ################################################################################
    # go over layes and shapes, remove all non-top layers and shapes

    cubeEdgeLength = 1
    layersRefined = []

    for xi in range (math.floor(global_minX / cubeEdgeLength), math.ceil(global_maxX / cubeEdgeLength)):       # math.floor(global_minX)
        for yi in range (math.floor(global_minY / cubeEdgeLength), math.ceil(global_maxY / cubeEdgeLength)):
            layersTemp : list = []

            for index, layer in enumerate(layers):

                xj = layer['points'][0]['x']
                yj = layer['points'][0]['y']

                if(((xj >= (xi * cubeEdgeLength)) & (xj < ((xi + 1) * cubeEdgeLength))) & ((yj >= (yi * cubeEdgeLength)) & (yj < ((yi + 1) * cubeEdgeLength)))):
                    layersTemp += [layer]  

            if(len(layersTemp) != 0):
                maxZ = 0     

                for index2, lay in enumerate(layersTemp):

                    currentZ = lay['height']

                    if currentZ> maxZ:
                        maxZ = currentZ      

                new_lay = lay.copy()
                new_lay['height'] = maxZ

                layersRefined.append(new_lay)

    #########
    # go over shapes, if you can find it's ID in the layersRefined, then keep the shape and change it's height 'Z', otherwise throw it away

    all_shapesRefined = []

    for index, shape in enumerate(all_shapes):

        for index2, lay in enumerate(layersRefined):

            if(shape['sId'] == lay['shapeId']):
                shapeTemp = shape.copy()

                h = lay['height'] - Z_adjustment

                shapeTemp['userInput'] = round(h * 10)/10 + Z_adjustment

                all_shapesRefined.append(shapeTemp)

        #print("")
        #print(shape)
        #print(all_shapesRefined[0])     
        #print("")   

    ################################################################################



    objects = {
        #"allShapes": all_shapes,
        #"layers": layers,
        "allShapes": all_shapesRefined,
        "layers": layersRefined,
        "scale": {
            "convertVal": 1.0,
            "unit":       "m"
        },
        "ratio": 1.0,
        "img": """data:image/gif;base64,R0lGODdhEAAQAMwAAPj7+FmhUYjNfGuxYYDJdYTIeanOpT+DOTuANXi/bGOrWj6CONzv2sPjv2CmV1unU4zPgISg6DJnJ3ImTh8Mtbs00aNP1CZSGy0YqLEn47RgXW8amasW7XWsmmvX2iuXiwAAAAAEAAQAAAFVyAgjmRpnihqGCkpDQPbGkNUOFk6DZqgHCNGg2T4QAQBoIiRSAwBE4VA4FACKgkB5NGReASFZEmxsQ0whPDi9BiACYQAInXhwOUtgCUQoORFCGt/g4QAIQA7"""
    }

    with open("data/json/objects2.example.json") as f:
        example = json.load(f)

    example = {**example, **objects}

    with open("data/result.json", "w") as f:
        json.dump(example, f, indent=2)

    # plot_3d(
    #     x=v["x"],
    #     y=v["y"],
    #     z=v["z"],
    #     text=v["count"],
    #     color=v["count"],
    # )

    plot_3d_json(objects=objects)

#    plot_3d_Grouped_json(objects=objects, global_maxX=global_maxX, global_minX=global_minX, global_maxY=global_maxY, global_minY=global_minY, minimumHeight=minimumHeightOfObstacle)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("Interrupted by user.")
