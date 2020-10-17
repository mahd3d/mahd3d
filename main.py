import itertools
import json
import uuid
from typing import Union, Tuple, Optional

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from typing import Union, Tuple, Optional
import itertools
import json
from DataRotation import correctlyRotateDataFrame, rotate
from LoadE57 import load_e57
from Plots import plot_3d, plot_3d_json

from utils import timeit


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


def plot_histograms(
    points: pd.DataFrame,
) -> None:
    for col in points:
        if col not in ["x", "y", "z"]:
            continue
        fig = px.histogram(points, x=col)
        fig.show()


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

    v = get_cubes(p[~p["roof"] & ~p["floor"]], grid)
    v = v[v["c"] > 5]

    layers = []
    for index, row in v.iterrows():
        layer = {
            "points": [
                {"x": row["x"], "y": row["y"], "id": 1},
                {"x": row["x"] + grid, "y": row["y"], "id": 2},
                {"x": row["x"] + grid, "y": row["y"] + grid, "id": 3},
                {"x": row["x"], "y": row["y"] + grid, "id": 4},
            ],
            "height": row["z"] + 2.4995,  # hardcoded floor level, not adjusted for tilt
            "shape_type": "obstacle",
            "shapeId": str(uuid.uuid4()),
        }
        layers += [layer]

    objects = {
        "unit": "m",
        "z_ceil": 4.9,
        "z_sat": 4.5,
        "z_marker": 1,
        "layers": layers,
        "optimize": True,
        "marker_grid": 1,
        "sat_grid": 10,
    }

    with open("data/result.json", "w") as f:
        json.dump(objects, f, indent=2)

    # plot_3d(
    #     x=v["x"],
    #     y=v["y"],
    #     z=v["z"],
    #     text=v["count"],
    #     color=v["count"],
    # )

    # TODO: unskew everything
    # TODO: build json

    # plot_3d_json()
    plot_3d_json(objects=objects)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("Interrupted by user.")
