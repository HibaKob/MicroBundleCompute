from microbundlecompute import optional_preprocessing as op
from microbundlecompute import image_analysis as ia
from pathlib import Path
import numpy as np
import cv2
import glob
import shutil
import os


def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    ex_path = data_path.joinpath(example_name).resolve()
    return ex_path


def movie_path(example_name):
    ex_path = example_path(example_name)
    mov_path = ex_path.joinpath("movie").resolve()
    return mov_path


def glob_movie(example_name):
    folder_path = example_path(example_name)
    movie_path = folder_path.joinpath("movie").resolve()
    name_list = glob.glob(str(movie_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list


def test_rename_folder():
    folder_path = example_path("io_testing_examples")
    new_folder_name = "test_create_folder_%i" % (np.random.random() * 1000000)
    new_folder_path = folder_path.joinpath(new_folder_name).resolve()
    folder_name = "old_name"
    _ = ia.create_folder(folder_path,folder_name)
    if os.path.exists(new_folder_path):
        shutil.rmtree(new_folder_path)
    new_folder_path = op.rename_folder(folder_path, folder_name, new_folder_name)
    assert new_folder_path.is_dir


def test_apply_image_kernel():
    array = np.random.random((10, 10))
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    known = cv2.filter2D(array, -1, kernel)
    found = op.apply_image_kernel(array, kernel)
    assert np.allclose(known, found)


def test_filter_all_images():
    folder_path = movie_path("example_image_filter")
    path_list = ia.image_folder_to_path_list(folder_path)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    filtered_img_list = op.filter_all_images(path_list, kernel)
    assert len(path_list) == len(filtered_img_list)


def test_run_image_filtering():
    folder_path = example_path("example_image_filter")
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    filtered_img_list = op.run_image_filtering(folder_path, kernel)
    raw_images = glob_movie(folder_path)
    assert len(filtered_img_list) == len(raw_images)


def test_adjust_first_valley():
    folder_path = example_path("example_adjust_valley")
    unadjusted_movie_path = folder_path.joinpath("unadjusted_movie").resolve()
    adjusted_movie_path = movie_path(folder_path)
    if os.path.exists(unadjusted_movie_path):
        shutil.rmtree(adjusted_movie_path)
        op.rename_folder(folder_path,"unadjusted_movie","movie")
    valley_image = 5
    adjusted_img_paths = op.adjust_first_valley(folder_path, valley_image)
    raw_movie_path = folder_path.joinpath("unadjusted_movie").resolve()
    raw_images = glob.glob(str(raw_movie_path) + '/*.TIF')
    assert len(adjusted_img_paths) == (len(raw_images) - valley_image)

     