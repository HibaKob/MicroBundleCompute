import matplotlib.pyplot as plt
from microbundlecompute import image_analysis as ia
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from skimage import io
from skimage import morphology


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


def test_apply_median_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = np.ones((10, 10))
    found = ia.apply_median_filter(array, filter_size)
    assert np.allclose(known, found)


def test_apply_gaussian_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = ndimage.gaussian_filter(array, filter_size)
    found = ia.apply_gaussian_filter(array, filter_size)
    assert np.allclose(known, found)


def test_compute_otsu_thresh():
    dim = 10
    known_lower = 10
    known_upper = 100
    std_lower = 2
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    found = ia.compute_otsu_thresh(x1)
    assert found > known_lower and found < (known_upper + known_lower)


def test_apply_otsu_thresh():
    dim = 10
    known_lower = 10
    known_upper = 10000
    std_lower = 0.1
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    known = x1 > np.mean(x1)
    found = ia.apply_otsu_thresh(x1)
    assert np.allclose(known, found)


def test_get_region_props():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    region_props = ia.get_region_props(array)
    assert region_props[0].area == np.sum(disk_1)
    assert region_props[1].area == np.sum(disk_2)


def test_get_domain_center():
    array = np.ones((10, 20))
    center_0, center_1 = ia.get_domain_center(array)
    assert center_0 == 5
    assert center_1 == 10


def test_region_to_coords():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 2
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    region_props = ia.get_region_props(array)
    coords_list = ia.region_to_coords(region_props)
    assert len(coords_list) == 3
    assert coords_list[0].shape[1] == 2


def test_coords_to_mask():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    coords = [region.coords]
    mask = ia.coords_to_mask(coords, disk_1)
    assert np.allclose(mask, disk_1)


def test_compute_distance():
    a0 = 0
    a1 = 0
    b0 = 10
    b1 = 0
    known = 10
    found = ia.compute_distance(a0, a1, b0, b1)
    assert known == found


def test_get_largest_regions():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 2
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    region_props = ia.get_region_props(array)
    num_regions = 2
    regions_list = ia.get_largest_regions(region_props, num_regions)
    assert len(regions_list) == 2
    assert regions_list[0].area == np.sum(disk_1)
    assert regions_list[1].area == np.sum(disk_2)


def test_invert_mask():
    array_half = np.zeros((10, 10))
    array_half[0:5, :] = 1
    array_invert = ia.invert_mask(array_half)
    assert np.allclose(array_invert + array_half, np.ones((10, 10)))


def test_close_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_missing = np.copy(array)
    array_missing[5, 5] = 0
    array_closed = ia.close_region(array_missing)
    assert np.allclose(array_closed, array)


def test_gabor_filter():
    array = np.zeros((10, 10))
    gabor_all = ia.gabor_filter(array)
    assert np.sum(gabor_all) == 0
    data_path = files_path()
    img_path = data_path.joinpath("segmentation_examples").resolve().joinpath("ex1_0000.TIF").resolve()
    array = io.imread(img_path)
    gabor_all = ia.gabor_filter(array)
    assert np.allclose(gabor_all, array) is False


def test_crop_boundaries():
    array = np.ones((100, 100))
    cropped_array = ia.crop_boundaries(array)
    assert np.sum(cropped_array) == (100 - 20 * 2) ** 2.0


def test_microbundle_mask_v1():
    data_path = files_path()
    img_path = data_path.joinpath("segmentation_examples").resolve().joinpath("ex1_0000.TIF").resolve()
    array = io.imread(img_path)
    mask = ia.microbundle_mask_v1(array)
    assert np.max(mask) == 1
    assert np.min(mask) == 0
    assert mask.shape[0] == array.shape[0]
    assert mask.shape[1] == array.shape[1]


def test_microbundle_mask_all_v1():
    folder_path = example_path("segmentation_examples")
    path_save = files_path()
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    mask_list = []
    for kk in range(0, len(tiff_list)):
        mask = ia.microbundle_mask_v1(tiff_list[kk])
        mask_list.append(mask)
        plt.figure()
        plt.imshow(mask)
        plt.savefig(str(path_save) + "%i_v1.png" % (kk))
    assert len(mask_list) == 5


