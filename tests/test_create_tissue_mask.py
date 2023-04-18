import glob
from microbundlecompute import create_tissue_mask as ctm
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import io
from skimage.filters import sobel
from skimage import morphology


def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    example_path = data_path.joinpath(example_name).resolve()
    return example_path


def glob_movie(example_name):
    folder_path = example_path(example_name)
    movie_path = folder_path.joinpath("movie").resolve()
    name_list = glob.glob(str(movie_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list


def test_apply_median_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = np.ones((10, 10))
    found = ctm.apply_median_filter(array, filter_size)
    assert np.allclose(known, found)


def test_apply_gaussian_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = ndimage.gaussian_filter(array, filter_size)
    found = ctm.apply_gaussian_filter(array, filter_size)
    assert np.allclose(known, found)


def test_apply_sobel_filter():
    array = np.random.random((10, 10))
    known = sobel(array)
    found = ctm.apply_sobel_filter(array)
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
    found = ctm.compute_otsu_thresh(x1)
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
    found = ctm.apply_otsu_thresh(x1)
    assert np.allclose(known, found)


def test_gabor_filter():
    array = np.zeros((10, 10))
    gabor_all = ctm.gabor_filter(array)
    assert np.sum(gabor_all) == 0
    file_path = glob_movie("real_example_masks")[0]
    img = io.imread(file_path)
    gabor_all = ctm.gabor_filter(img)
    assert np.allclose(gabor_all, img) is False
    theta_range = 9
    ff_max = 6
    ff_mult = 0.05
    gabor_all_2 = ctm.gabor_filter(img, theta_range, ff_max, ff_mult)
    assert np.allclose(gabor_all_2, img) is False


def test_invert_mask():
    array_half = np.zeros((10, 10))
    array_half[0:5, :] = 1
    array_invert = ctm.invert_mask(array_half)
    assert np.allclose(array_invert + array_half, np.ones((10, 10)))


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
    num_regions = 2
    regions_list = ctm.get_largest_regions(array, num_regions)
    assert len(regions_list) == 2
    assert regions_list[0].area == np.sum(disk_1)
    assert regions_list[1].area == np.sum(disk_2)


def test_region_to_mask():
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
    region = ctm.get_largest_regions(array)[0]
    mask = ctm.region_to_mask(region, array)
    assert np.sum(mask) == np.sum(disk_1)


def test_close_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_missing = np.copy(array)
    array_missing[5, 5] = 0
    array_closed = ctm.close_region(array_missing)
    assert np.allclose(array_closed, array)


def test_segment_mask_1():
    file_path = glob_movie("real_example_masks")[0]
    img = io.imread(file_path)
    mask = ctm.segment_mask_1(img)
    assert mask.shape == img.shape


def test_segment_mask_2():
    file_path = glob_movie("real_example_masks")[1]
    img = io.imread(file_path)
    mask = ctm.segment_mask_2(img)
    assert mask.shape == img.shape

def test_segment_mask_3():
    file_path = glob_movie("real_example_masks/movie/0003")
    img_list = io.imread_collection(file_path)
    mask = ctm.segment_mask_3(img_list)
    assert mask.shape == img_list[0].shape

def test_save_mask():
    folder_path = example_path("real_example_masks")
    file_path = glob_movie("real_example_masks")[1]
    img = io.imread(file_path)
    mask = ctm.segment_mask_2(img)
    mask_file_path, img_path = ctm.save_mask(folder_path, mask, "test_save")
    assert mask_file_path.is_file()
    assert img_path.is_file()


def test_run_create_tissue_mask():
    folder_path = example_path("real_example_masks")
    seg_fcn_num = 1
    fname = "example_0"
    frame_num = 0
    file_path, img_path = ctm.run_create_tissue_mask(folder_path, seg_fcn_num, fname, frame_num)
    assert file_path.is_file()
    assert img_path.is_file()
    folder_path = example_path("real_example_masks")
    seg_fcn_num = 2
    fname = "example_1"
    frame_num = 1
    file_path, img_path = ctm.run_create_tissue_mask(folder_path, seg_fcn_num, fname, frame_num)
    assert file_path.is_file()
    assert img_path.is_file()
    folder_path = example_path("real_example_masks")
    seg_fcn_num = 2
    fname = "example_2"
    frame_num = 2
    file_path, img_path = ctm.run_create_tissue_mask(folder_path, seg_fcn_num, fname, frame_num)
    assert file_path.is_file()
    assert img_path.is_file()
    folder_path = example_path("real_example_masks/movie/0003")
    seg_fcn_num = 3
    fname = "example_3"
    frame_num = 0
    file_path, img_path = ctm.run_create_tissue_mask(folder_path, seg_fcn_num, fname, frame_num, "minimum")
    assert file_path.is_file()
    assert img_path.is_file()
    folder_path = example_path("real_example_masks/movie/0003")
    seg_fcn_num = 3
    fname = "example_3_max"
    frame_num = 0
    file_path, img_path = ctm.run_create_tissue_mask(folder_path, seg_fcn_num, fname, frame_num, "maximum")
    assert file_path.is_file()
    assert img_path.is_file()
