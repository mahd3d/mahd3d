import numpy as np
import pye57
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from typing import Union, Tuple

from utils import timeit


class Line:
    p1 = (0, 0, 0)
    x1 = 0
    y1 = 0
    z1 = 0
    p2 = (1, 1, 1)
    x2 = 0
    y2 = 0
    z2 = 0
    m = (1, 1, 1)

    def equation(s):
        return f"= f{s.p1} + t {s.p2}"


class Sphere:
    p = (0.5, 0.5, 0.5)
    x = 0.5
    y = 0.5
    z = 0.5
    r = 0.5
    d = 1.0

    def equation(s):
        return f"(x-{s.x})²+(y-{s.y})²+(y-{s.y})² = {s.r**2}"


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
        except pye57.libe57.E57Exception as e:
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
    # points["roof"] = points.apply(compute_roof, axis=1)
    # points["floor"] = points.apply(compute_floor, axis=1)

    return points


def compute_rgba(
    row,
) -> str:
    if "a" not in row:
        row["a"] = 1.0
    return f'rgba({row["r"]}, {row["g"]}, {row["b"]}, {row["a"]:.2f})'


def compute_alpha(
    row,
) -> float:
    if row["roof"] or row["floor"]:
        return 0.3
    else:
        return 1.0


def plot_histograms(
    points: pd.DataFrame,
) -> bool:
    for col in points:
        fig = px.histogram(points, x=col)
        fig.show()
    return True


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


def plot_3d_json() -> None:
    import json
    with open("data/json/objects.example.json", "r") as f:
        data = json.load(f)

    cubes = []
    for layer_index, layer_value in enumerate(data["layers"]):
        cubes += [[]]
        for i, v in enumerate(layer_value["points"]):
            cubes[layer_index] += [(v["x"], v["y"], 0)]
            cubes[layer_index] += [(v["x"], v["y"], layer_value["height"])]

    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=[i[0] for i in cubes[0]],
            y=[i[1] for i in cubes[1]],
            z=[i[2] for i in cubes[2]],
            colorbar_title='z',
            colorscale=[[0, 'gold'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity=np.linspace(0, 1, 8, endpoint=True),
            # # i, j and k give the vertices of triangles
            # i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            # j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            # k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            name='y',
            showscale=True
        )
    ])

    fig.show()
    pass


def main() -> None:
    plot_3d_json()

    # Detail size, smaller is more detailed but slower
    # 1000 is recommended for displaying with Plotly, 300 is the minimum
    step: int = 3000
    filename = f"data/computed/points_{step}.v2.df.feather"

    try:
        print(
            f"Trying to use existing points with step size of {step} from saved points file."
        )
        p = pd.read_feather(filename)
    except FileNotFoundError as e:
        print(f"Couldn't get points with step size of {step}, loading raw data.")
        data, header = load_e57()
        p: pd.DataFrame = get_points(data, step)
        p.to_feather(filename)
    finally:
        print("p loaded")

    # plot_histograms(p)

    p["0"] = 0
    p = get_points_with_computed(p)

    # p = p[~p["roof"] & ~p["floor"]]
    p = p[~p["roof"]]
    # points = points[(points["z"] >= -1.5) & (points["z"] <= -1.0)]
    # points["a"] = points.apply(compute_alpha, axis=1)

    plot_3d(
        x=p["x"],
        y=p["y"],
        z=p["z"],
        text=p.index,
        color=p["rgba"],
    )

    f = p[p["floor"]].copy()
    f["count"] = 0
    import itertools

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

    plot_3d(
        x=f["x"],
        y=f["y"],
        z=f["0"],
        text=f["count"],
        color=f["count"],
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("Interrupted by user.")
