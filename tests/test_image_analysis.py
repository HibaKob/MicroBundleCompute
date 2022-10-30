from microbundlecompute import image_analysis as ia
import numpy as np
import os
from pathlib import Path
from skimage import io


def test_hello_world():
    res = ia.hello_microbundle_compute()
    assert res == "Hello World!"


def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    example_path = data_path.joinpath(example_name).resolve()
    return example_path


def test_read_tiff():
    data_path = files_path()
    img_path = data_path.joinpath("segmentation_examples").resolve().joinpath("ex1_0000.TIF").resolve()
    known = io.imread(img_path)
    found = ia.read_tiff(img_path)
    assert np.allclose(known, found)


def test_image_folder_to_path_list():
    folder_path = example_path("segmentation_examples")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    assert len(name_list_path) == 5
    for kk in range(0, 5):
        assert os.path.isfile(name_list_path[kk])


def test_read_all_tiff():
    folder_path = example_path("segmentation_examples")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    assert len(tiff_list) == 5
    assert tiff_list[0].shape == (512, 512)


