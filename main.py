import numpy as np
import pye57
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from typing import Union, Tuple, Optional
import itertools
import json

from utils import timeit


@timeit
def load_e57(
    file: str = "data/raw/CustomerCenter1 1.e57",
) -> Tuple[dict, dict]:
    """Return a dictionary with the point types as keys."""
    print(f"Loading e57 file {file}.")
    e57 = pye57.E57(file)

    # Get and clean-up header
    raw_header = e57.get_header(0)
    header = {}
    for attr in dir(raw_header):
        if attr[0:1].startswith("_"):
            continue
        try:
            value = getattr(raw_header, attr)
        except pye57.libe57.E57Exception:
            continue
        else:
            header[attr] = value

    header["pos"] = e57.scan_position(0)

    data = e57.read_scan_raw(0)
    # for key, values in data.items():
    #     assert isinstance(values, np.ndarray)
    #     assert len(values) == 151157671
    #     print(f"len of {key}: {len(values)} ")

    return data, header


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

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # proportional aspect ratio with data, alternatively cube or manual (see below)
    fig.update_layout(scene_aspectmode="data")

    fig.show()
    pass


def get_squares(
    p: pd.DataFrame,
) -> pd.DataFrame:
    f = p.copy()

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
) -> pd.DataFrame:
    v = p.copy()

    v["count"] = 0

    grid: int = 1
    counts = pd.DataFrame(
        data={
            "x": [],
            "y": [],
            "z": [],
            # "r": [], "g": [], "b": [], "i": [],
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
            & (v["x"] < x + 1 + overscan)
            & (v["y"] >= y - overscan)
            & (v["y"] < y + 1 + overscan)
            & (v["z"] >= z - overscan)
            & (v["z"] < z + 1 + overscan)
        ]
        count = len(dots)
        counts = counts.append({"x": x, "y": y, "z": z, "c": count}, ignore_index=True)
        #        counts += [count]
        v.loc[
            (v["x"] >= x)
            & (v["x"] < x + 1)
            & (v["y"] >= y)
            & (v["y"] < y + 1)
            & (v["z"] >= z)
            & (v["z"] < z + 1),
            "count",
        ] = count
        # print(f"{x}, {y}: {count}")

    v.loc[v["count"] > 20, "count"] = 20
    v = v[v["count"] > 5]

    return v


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

    # mid = p[~p["roof"] & ~p["floor"]]
    # mid = p[~p["roof"]]
    # points = points[(points["z"] >= -1.5) & (points["z"] <= -1.0)]
    # points["a"] = points.apply(compute_alpha, axis=1)

    # plot_3d(
    #     x=mid["x"],
    #     y=mid["y"],
    #     z=mid["z"],
    #     text=mid.index,
    #     color=mid["rgba"],
    # )
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

    v = get_cubes(p[~p["roof"] & ~p["floor"]])
    v["x2"] = v["x"] + 1
    v["y2"] = v["y"] + 1

    import uuid

    layers = []
    for index, row in v.iterrows():
        layer = {
            "points": [
                {"x": row["x"], "y": row["y"], "id": 1},
                {"x": row["x2"], "y": row["y"], "id": 2},
                {"x": row["x2"], "y": row["y2"], "id": 3},
                {"x": row["x"], "y": row["y2"], "id": 4},
            ],
            "height": row["z"] + 1.5,  # hardcoded floor level
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
