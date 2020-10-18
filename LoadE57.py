import pye57
from typing import Tuple
from src.utils import timeit

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
