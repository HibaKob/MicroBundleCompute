import glob
import numpy as np
from pathlib import Path
from skimage import io
from typing import List


def hello_microbundle_compute() -> str:
    "Given no input. Simple hello world as a test function."
    return "Hello World!"


def read_tiff(img_path: Path) -> np.ndarray:
    """Given a path to a tiff. Will return an array."""
    img = io.imread(img_path)
    return img


def image_folder_to_path_list(folder_path: Path) -> List:
    """Given a folder path. Will return the path to all files in that path in order."""
    name_list = glob.glob(str(folder_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list_path


def read_all_tiff(path_list: List) -> List:
    """Given a folder path. Will return a list of all tiffs as an array."""
    tiff_list = []
    for path in path_list:
        array = read_tiff(path)
        tiff_list.append(array)
    return tiff_list


# segment shoshana images


