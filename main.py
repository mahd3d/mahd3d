import numpy as np
import pye57
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
# import pickle
from typing import Union, Optional
# from typecheck import typecheck as typed


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



def load_e57(file: str = "data/raw/CustomerCenter1 1.e57") -> dict:
    """Return a dictionary with the point types as keys."""
    print(f"Loading e57 file {file}.")
    e57 = pye57.E57(file)
    data = e57.read_scan_raw(0)

    for key, values in data.items():
        assert isinstance(values, np.ndarray)
        print(f"len of {key}: {len(values)} ")

    return data


def get_points(data: dict, step: int = 1000) -> pd.DataFrame:

    x = data["cartesianX"][::step]
    y = data["cartesianY"][::step]
    z = data["cartesianZ"][::step]
    r = data["colorRed"][::step]
    g = data["colorGreen"][::step]
    b = data["colorBlue"][::step]
    i = data["intensity"][::step]

    points: pd.DataFrame = pd.DataFrame(data={"x": x, "y": y, "z": z, "r": r, "g": g, "b": b, "i": i})

    return points


def get_points_with_computed(points: pd.DataFrame) -> pd.DataFrame:
    points["rgba"] = points.apply(compute_rgba, axis=1)
    points["roof"] = points.apply(compute_roof, axis=1)
    points["floor"] = points.apply(compute_floor, axis=1)

    return points


def compute_rgba(row) -> str:
    if "a" not in row:
        row["a"] = 1.0
    return f'rgba({row["r"]}, {row["g"]}, {row["b"]}, {row["a"]:.2f})'


def compute_rgba_from_count(row) -> str:
    if "a" not in row:
        row["a"] = 1.0
    return f'rgba({row["r"]}, {row["g"]}, {row["b"]}, {row["a"]:.2f})'


def compute_roof(row) -> bool:
    return row["z"] > 3.4
    # if row["z"] > 5:
    #     row["r"] = 255
    #     row["g"] = 255
    #     row["b"] = 255


def compute_floor(row) -> bool:
    return row["z"] < -1.5


def compute_alpha(row) -> float:
    if row["roof"] or row["floor"]:
        return 0.3
    else:
        return 1.0


def plot_histograms(points: pd.DataFrame) -> bool:
    for col in points:
        fig = px.histogram(points, x=col)
        fig.show()
    return True


def plot_3d(x: pd.Series, y: pd.Series, z: pd.Series, text: Union[None, pd.Series, pd.Index], color: [None, pd.Series],) -> None:
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        text=x.index if text is None else text,
        mode='markers',
        marker=dict(
            size=8,
            color=z if color is None else color,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8,
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # proportional aspect ratio with data, alternatively cube or manual (see below)
    fig.update_layout(scene_aspectmode="data")
    # fig.update_layout(scene_aspectratio=dict(x=d_x/100, y=d_y/100, z=d_z/100))
    fig.show()


def main() -> None:

    # Detail size, smaller is more detailed but slower
    # 1000 is recommended for displaying with Plotly, 300 is the minimum
    step: int = 3000

    try:
        print(f"Trying to use existing points with step size of {step} from saved points file.")
        # p = pd.read_pickle(f"data/computed/points_{step}.df.pickle")
        # p = pd.read_hdf(f"data/computed/points_{step}.df.hdf", "df")
        # p = pd.read_parquet(f"data/computed/points_{step}.df.parquet")
        p = pd.read_feather(f"data/computed/points_{step}.df.feather")
    except Exception as e:
        print(f"Couldn't get points with step size of {step}, loading raw data.")
        data: dict = load_e57()
        p: pd.DataFrame = get_points(data, step)
        # p.to_pickle(f"data/computed/points_{step}.df.pickle")
        # p.to_hdf(f"data/computed/points_{step}.df.hdf", "df")
        # p.to_parquet(f"data/computed/points_{step}.df.parquet")
        # p.to_feather(f"data/computed/points_{step}.df.feather")
    finally:
        print("p loaded")

    # plot_histograms(p)

    p["0"] = 0
    p = get_points_with_computed(p)

    p = p[~p["roof"] & ~p["floor"]]
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
        dots = f[(f["x"] >= x-overscan) & (f["x"] < x+1+overscan) & (f["y"] >= y-overscan) & (f["y"] < y+1+overscan)]
        count = len(dots)
        counts += [count]
        f.loc[(f["x"] >= x) & (f["x"] < x + 1) & (f["y"] >= y) & (f["y"] < y + 1), "count"] = count
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
