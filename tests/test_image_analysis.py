import matplotlib.pyplot as plt
from microbundlecompute import image_analysis as ia
from microbundlecompute import image_analysis as ia
import numpy as np
import os
from pathlib import Path
import pytest
from skimage import io
from skimage.transform import estimate_transform, warp
from scipy import signal
import warnings


def test_hello_world():
    # simple test to let the user know that install has worked
    res = ia.hello_microbundle_compute()
    assert res == "Hello World!"


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


def tissue_mask_path(example_name):
    ex_path = example_path(example_name)
    mask_path = ex_path.joinpath("masks").resolve()
    t_m_path = mask_path.joinpath("tissue_mask.txt").resolve()
    return t_m_path


def test_read_tiff():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    known = io.imread(img_path)
    found = ia.read_tiff(img_path)
    assert np.allclose(known, found)


def test_create_folder_guaranteed_conditions():
    folder_path = example_path("io_testing_examples")
    new_folder_name = "test_create_folder_%i" % (np.random.random() * 1000000)
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()


def test_image_folder_to_path_list():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    assert len(name_list_path) == 5
    for kk in range(0, 5):
        assert os.path.isfile(name_list_path[kk])


def test_read_all_tiff():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    assert len(tiff_list) == 5
    assert tiff_list[0].shape == (512, 512)


def test_uint16_to_uint8():
    array_8 = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    array_8[0, 0] = 0
    array_8[1, 0] = 255
    array_16 = array_8.astype(np.uint16) * 100
    found = ia.uint16_to_uint8(array_16)
    assert np.allclose(array_8, found)


def test_uint16_to_uint8_all():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    uint8_list = ia.uint16_to_uint8_all(tiff_list)
    for img in uint8_list:
        assert img.dtype is np.dtype('uint8')


def test_bool_to_uint8():
    arr_bool = np.random.random((10, 10)) > 0.5
    arr_uint8 = ia.bool_to_uint8(arr_bool)
    assert np.max(arr_uint8) == 1
    assert np.min(arr_uint8) == 0
    assert arr_uint8.dtype == np.dtype("uint8")


def test_read_txt_as_mask():
    file_path = tissue_mask_path("real_example_super_short")
    arr = ia.read_txt_as_mask(file_path)
    assert arr.dtype is np.dtype("uint8")


def test_shrink_pair():
    v0 = 100
    v1 = 200
    sf = 0.1
    new_v0, new_v1 = ia.shrink_pair(v0, v1, sf)
    diff = new_v1 - new_v0
    assert diff == 90


def test_remove_pillar_region():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    new_mask = ia.remove_pillar_region(mask, clip_columns = True, clip_rows = False )
    assert np.sum(mask) > np.sum(new_mask)
    box = ia.mask_to_box(mask)
    _, _, c0, c1 = ia.box_to_bound(box)
    new_box = ia.mask_to_box(new_mask)
    _, _, c0_new, c1_new = ia.box_to_bound(new_box)
    assert c0_new > c0
    assert c1_new < c1
    new_mask = ia.remove_pillar_region(mask, clip_columns = False, clip_rows = True )
    assert np.sum(mask) > np.sum(new_mask)
    box = ia.mask_to_box(mask)
    r0, r1, _, _ = ia.box_to_bound(box)
    new_box = ia.mask_to_box(new_mask)
    r0_new, r1_new, _, _ = ia.box_to_bound(new_box)
    assert r0_new > r0
    assert r1_new < r1


def test_box_to_bound():
    r0 = 20
    r1 = 50
    c0 = 40
    c1 = 80
    box = np.asarray([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])
    r0_found, r1_found, c0_found, c1_found = ia.box_to_bound(box)
    assert r0 == r0_found
    assert r1 == r1_found
    assert c0 == c0_found
    assert c1 == c1_found


def test_bound_to_box():
    r0 = 20
    r1 = 50
    c0 = 40
    c1 = 80
    box_known = np.asarray([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])
    box_found = ia.bound_to_box(r0, r1, c0, c1)
    assert np.allclose(box_known, box_found)


def test_is_in_box():
    box = ia.bound_to_box(10, 100, 30, 900)
    assert ia.is_in_box(box, 1000, 300) is False
    assert ia.is_in_box(box, 50, 400) is True
    assert ia.is_in_box(box, 50, 1000) is False
    assert ia.is_in_box(box, 3, 500) is False


def test_sub_division_markers():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    img_uint8 = img_list_uint8[0]

    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    box_mask = ia.mask_to_box(mask)
    r0, r1, c0, c1 = ia.box_to_bound(box_mask)

    sub_division_dim_pix = 20
    num_tile_row = int(np.floor((r1 - r0) / sub_division_dim_pix))
    num_tile_col = int(np.floor((c1 - c0) / sub_division_dim_pix))
    
    feature_params, _ = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, mask, feature_params)
    tracker_row = track_points_0[:, 0, 1]
    tracker_col = track_points_0[:, 0, 0]
    
    for rr in range(0, num_tile_row):
        for cc in range(0, num_tile_col):
            tile_box = ia.bound_to_box(r0 + rr * sub_division_dim_pix, r0 + (rr + 1) * sub_division_dim_pix, c0 + cc * sub_division_dim_pix, c0 + (cc + 1) * sub_division_dim_pix)
            num_sub_pts = ia.sub_division_markers(tracker_row, tracker_col, tile_box) 
            assert num_sub_pts < len(tracker_col)


def test_sub_division_mask():
    mask = np.zeros((60,60))
    mask[::2,::2] = 1

    r0, r1, c0, c1 = 0 , 59, 0, 59

    sub_division_dim_pix = 20
    num_tile_row = int(np.floor((r1 - r0) / sub_division_dim_pix))
    num_tile_col = int(np.floor((c1 - c0) / sub_division_dim_pix))

    for rr in range(0, num_tile_row):
        for cc in range(0, num_tile_col):
            tile_box = ia.bound_to_box(r0 + rr * sub_division_dim_pix, r0 + (rr + 1) * sub_division_dim_pix, c0 + cc * sub_division_dim_pix, c0 + (cc + 1) * sub_division_dim_pix)
            mask_in_div = ia.sub_division_mask(mask, tile_box) 
            assert mask_in_div < sub_division_dim_pix**2


def test_get_tracking_param_dicts():
    feature_params, lk_params = ia.get_tracking_param_dicts()
    assert feature_params["maxCorners"] == 10000
    assert feature_params["qualityLevel"] == 0.1  # 0.005
    assert feature_params["minDistance"] == 3
    assert feature_params["blockSize"] == 3
    assert lk_params["winSize"][0] == 5
    assert lk_params["winSize"][1] == 5
    assert lk_params["maxLevel"] == 5
    assert lk_params["criteria"][1] == 10
    assert lk_params["criteria"][2] == 0.03


def test_compute_local_coverage():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    img_uint8 = img_list_uint8[0]

    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    
    feature_params, _ = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, mask, feature_params)
    sub_division_dim_pix = 20
    all_local_coverage = ia.compute_local_coverage(mask, track_points_0, sub_division_dim_pix)
    assert np.max(all_local_coverage) <= sub_division_dim_pix**2


def test_adjust_qualityLevel():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    img_uint8 = img_list_uint8[0]

    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params_init, _ = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, mask, feature_params_init)
    min_coverage = 40
    coverage_init = np.sum(mask) / track_points_0.shape[0]
    qL_init = feature_params_init["qualityLevel"]
    qualityLevel, coverage = ia.adjust_qualityLevel(feature_params_init, img_uint8, mask, min_coverage)
    assert qL_init >= qualityLevel
    assert coverage <= coverage_init

def test_adjust_feature_param_dicts():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8 = ia.uint16_to_uint8(img)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params_init, _ = ia.get_tracking_param_dicts()
    qL_init = feature_params_init["qualityLevel"]
    minDist_init = feature_params_init["minDistance"]
    feature_params_new = ia.adjust_feature_param_dicts(feature_params_init, img_uint8, mask)
    assert qL_init >= feature_params_new["qualityLevel"]
    assert minDist_init <= feature_params_new["minDistance"]


def test_mask_to_track_points():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8 = ia.uint16_to_uint8(img)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, _ = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, mask, feature_params)
    assert track_points_0.shape[1] == 1
    assert track_points_0.shape[2] == 2


def test_mask_to_track_points_synthetic():
    img = np.zeros((100, 100))
    for kk in range(1, 10):
        img[int(kk * 10), int(kk * 5)] = 1
    img_uint8 = ia.uint16_to_uint8(img)
    mask = np.ones(img_uint8.shape)
    feature_params, _ = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, mask, feature_params)
    tp_0_ix_col = np.sort(track_points_0[:, 0, 0])  # note col
    tp_0_ix_row = np.sort(track_points_0[:, 0, 1])  # note row
    for kk in range(1, 10):
        tp_0 = tp_0_ix_row[kk - 1]
        tp_1 = tp_0_ix_col[kk - 1]
        val0 = int(kk * 10)
        val1 = int(kk * 5)
        assert np.isclose(tp_0, val0, atol=1)
        assert np.isclose(tp_1, val1, atol=1)


def test_track_one_step():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8_0 = ia.uint16_to_uint8(img)
    img_path = mov_path.joinpath("ex1_0001.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8_1 = ia.uint16_to_uint8(img)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8_0, mask, feature_params)
    track_points_1 = ia.track_one_step(img_uint8_0, img_uint8_1, track_points_0, lk_params)
    assert track_points_1.shape[1] == 1
    assert track_points_1.shape[2] == 2
    assert track_points_1.shape[0] == track_points_0.shape[0]


def test_track_one_step_synthetic():
    img_0 = np.zeros((100, 100))
    for kk in range(1, 10):
        img_0[int(kk * 10), int(kk * 5)] = 1
    img_1 = np.zeros((100, 100))
    for kk in range(1, 10):
        img_1[int(kk * 10 + 1), int(kk * 5 + 1)] = 1
    img_uint8_0 = ia.uint16_to_uint8(img_0)
    img_uint8_1 = ia.uint16_to_uint8(img_1)
    mask = np.ones(img_uint8_0.shape)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8_0, mask, feature_params)
    track_points_1 = ia.track_one_step(img_uint8_0, img_uint8_1, track_points_0, lk_params)
    compare = np.abs(track_points_1 - track_points_0)
    assert np.max(compare) < np.sqrt(2)


def test_track_all_steps():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    tracker_0, tracker_1 = ia.track_all_steps(img_list_uint8, mask, feature_params, lk_params)
    assert tracker_0.shape[1] == len(tiff_list)
    assert tracker_1.shape[1] == len(tiff_list)


def test_track_all_steps_with_adjust_param_dicts():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
    diff_0 = np.abs(tracker_0[:, 0] - tracker_0[:, -1])
    diff_1 = np.abs(tracker_1[:, 0] - tracker_1[:, -1])
    _, lk_params = ia.get_tracking_param_dicts()
    assert np.max(diff_0) < lk_params["winSize"][0]
    assert np.max(diff_1) < lk_params["winSize"][1]



def test_track_all_steps_with_adjust_param_dicts_warping():
    # import first image
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    img_uint8_0 = img_list_uint8[0]
    track_points_0 = ia.mask_to_track_points(img_uint8_0, mask, feature_params)
    # warp image by a known amount
    img_0 = img_uint8_0
    src = np.dstack([track_points_0[:, 0, 0].flat, track_points_0[:, 0, 1].flat])[0]
    diff_value = 2.0
    dst = src + diff_value
    tform = estimate_transform('projective', src, dst)
    tform.estimate(src, dst)
    img_1 = warp(img_0, tform, order=1, preserve_range=True)
    # perform tracking
    tiff_list = [img_0, img_1]
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
    diff_0 = np.abs(tracker_0[:, 0] - tracker_0[:, -1])
    diff_1 = np.abs(tracker_1[:, 0] - tracker_1[:, -1])
    # measure difference wrt ground truth
    assert np.mean(diff_0) > diff_value - 0.02
    assert np.mean(diff_0) < diff_value + 0.02
    assert np.mean(diff_1) > diff_value - 0.02
    assert np.mean(diff_1) < diff_value + 0.02


def test_compute_abs_position_timeseries():
    num_pts = 3
    num_frames = 100
    tracker_0 = 100 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    tracker_1 = 50 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    disp_abs_mean, disp_abs_all, disp_0_all, disp_1_all = ia.compute_abs_position_timeseries(tracker_0, tracker_1)
    assert disp_abs_mean.shape[0] == num_frames
    assert np.max(disp_abs_mean) < np.sqrt(2.0)
    assert disp_abs_all.shape[0] == num_pts
    assert disp_abs_all.shape[1] == num_frames
    assert disp_0_all.shape[0] == num_pts
    assert disp_0_all.shape[1] == num_frames
    assert np.max(disp_0_all) < 1
    assert disp_1_all.shape[0] == num_pts
    assert disp_1_all.shape[1] == num_frames
    assert np.max(disp_1_all) < 1


def test_subpixel_abs_position_timeseries():
    # import first image
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    img_uint8_0 = img_list_uint8[0]
    track_points_0 = ia.mask_to_track_points(img_uint8_0, mask, feature_params)
    # warp image by a known amount
    img_0 = img_uint8_0
    src = np.dstack([track_points_0[:, 0, 0].flat, track_points_0[:, 0, 1].flat])[0]
    diff_value = 0.01
    dst = src + diff_value
    tform = estimate_transform('projective', src, dst)
    tform.estimate(src, dst)
    img_1 = warp(img_0, tform, order=1, preserve_range=True)
    # perform tracking
    tiff_list = [img_0, img_1]
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    with warnings.catch_warnings(record=True) as record:
        tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
        _, disp_abs_all, _, _ = ia.compute_abs_position_timeseries(tracker_0, tracker_1)
    assert np.max(disp_abs_all) < 1
    assert len(record) == 1


def test_superpixel_abs_position_timeseries():
    # import first image
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    img_uint8_0 = img_list_uint8[0]
    track_points_0 = ia.mask_to_track_points(img_uint8_0, mask, feature_params)
    # warp image by a known amount
    img_0 = img_uint8_0
    src = np.dstack([track_points_0[:, 0, 0].flat, track_points_0[:, 0, 1].flat])[0]
    diff_value = 2
    dst = src + diff_value
    tform = estimate_transform('projective', src, dst)
    tform.estimate(src, dst)
    img_1 = warp(img_0, tform, order=1, preserve_range=True)
    # perform tracking
    tiff_list = [img_0, img_1]
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    with warnings.catch_warnings(record=True) as record:
        tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
        _, disp_abs_all, _, _ = ia.compute_abs_position_timeseries(tracker_0, tracker_1)
    assert np.max(disp_abs_all) > 1
    assert len(record) == 0


def test_get_time_segment_param_dicts():
    time_seg_params = ia.get_time_segment_param_dicts()
    assert time_seg_params["peakDist"] == 20
    assert time_seg_params["prom"] == 0.1


def test_adjust_time_seg_params():
    x = np.linspace(0, 500 * np.pi * 2.0, 500)
    timeseries = np.sin(x / (np.pi * 2.0) / 20 - np.pi / 2.0)
    time_seg_params = ia.get_time_segment_param_dicts()
    pd_init = time_seg_params["peakDist"]
    time_seg_params_new = ia.adjust_time_seg_params(time_seg_params, timeseries)
    assert time_seg_params_new["peakDist"] > pd_init
    assert time_seg_params_new["prom"] == 0.1


def test_compute_valleys():
    x = np.linspace(0, 500 * np.pi * 2.0, 500)
    timeseries = np.sin(x / (np.pi * 2.0) / 20 - np.pi / 2.0)
    info = ia.compute_valleys(timeseries)
    assert info.shape[0] == 2
    assert np.isclose(timeseries[info[0, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[0, 2]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 2]], -1, atol=.01)
    li = 10 * [-0.99] + list(timeseries) + 10 * [-0.99]
    timeseries = np.asarray(li)
    info = ia.compute_valleys(timeseries)
    assert np.isclose(timeseries[info[0, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[0, 2]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 2]], -1, atol=.01)


def test_compute_peaks():
    x = np.linspace(0, 500 * np.pi * 2.0, 500)
    timeseries = np.sin(x / (np.pi * 2.0) / 20 - np.pi / 2.0)
    peaks = ia.compute_peaks(timeseries)
    assert np.isclose(timeseries[peaks[0]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[1]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[2]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[3]], 1, atol=.01)
    li = 10 * [-0.99] + list(timeseries) + 10 * [-0.99]
    timeseries = np.asarray(li)
    peaks = ia.compute_peaks(timeseries)
    assert np.isclose(timeseries[peaks[0]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[1]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[2]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[3]], 1, atol=.01)


def test_compute_beat_frequency():
    x = np.linspace(0, 500 * np.pi * 2.0, 500)
    timeseries = np.sin(x / (np.pi * 2.0) / 20 - np.pi / 2.0)
    fps = 1
    info = ia.compute_valleys(timeseries)
    freq = ia.compute_beat_frequency(info, fps)
    assert np.isclose(freq, fps *1 / (np.pi * 2.0) / 20, atol=.01)
    li = 10 * [-0.99] + list(timeseries) + 10 * [-0.99]
    timeseries = np.asarray(li)
    info = ia.compute_valleys(timeseries)
    fps = 2
    freq = ia.compute_beat_frequency(info, fps)
    assert np.isclose(freq, fps * 1 / (np.pi * 2.0) / 20, atol=.01)
    x = np.linspace(0, 800 * np.pi * 2.0, 800)
    timeseries = np.sin(x / (np.pi * 2.0) / 20 - np.pi / 2.0)
    fps = 1
    info = ia.compute_valleys(timeseries)
    freq = ia.compute_beat_frequency(info, fps)
    assert np.isclose(freq, fps *1 / (np.pi * 2.0) / 20, atol=.01)


def test_compute_beat_amplitude():
    length_scale = 1
    info = [[0, 30, 60], [1, 60, 90], [2, 90, 120]]
    info = np.asarray(info)
    num_beat_frames = 30
    num_pts = 10
    t = np.linspace(0, 1, num_beat_frames)

    tracker_0_beat = np.ones((num_pts, num_beat_frames)) + 2.5*signal.sawtooth(2 * np.pi * t, 0.5)+1
    tracker_0_all = np.hstack((tracker_0_beat,tracker_0_beat,tracker_0_beat,tracker_0_beat,tracker_0_beat))
    tracker_0 = [tracker_0_beat,tracker_0_beat,tracker_0_beat,tracker_0_beat]

    tracker_1_beat = 0.1 * np.ones((num_pts, num_beat_frames)) + np.random.random((num_pts, num_beat_frames))
    tracker_1_all = np.hstack((tracker_1_beat,tracker_1_beat,tracker_1_beat,tracker_1_beat,tracker_1_beat))
    tracker_1 = [tracker_1_beat,tracker_1_beat,tracker_1_beat,tracker_1_beat]
    timeseries, _, _, _ = ia.compute_abs_position_timeseries(tracker_0_all, tracker_1_all)
    amplitude = ia.compute_beat_amplitude(timeseries, tracker_0, tracker_1, info, length_scale)
    known_amplitude = np.max(timeseries) - np.min(timeseries)
    assert np.isclose(amplitude, known_amplitude, atol=.01)


def test_save_beat_info():
    frequency = float(1 / (np.pi * 2.0) / 20)
    amplitude = float(2)
    folder_path = example_path("real_example_super_short")
    saved_path = ia.save_beat_info(folder_path = folder_path, frequency = frequency, amplitude = amplitude)
    assert saved_path.is_file()


def test_test_frame_0_valley():
    mov_path = movie_path("example_frame0_not_valley")
    name_list_path = ia.image_folder_to_path_list(mov_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    mask_path = tissue_mask_path("example_frame0_not_valley")
    mask = ia.read_txt_as_mask(mask_path)
    tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
    timeseries, _, _, _  = ia. compute_abs_position_timeseries(tracker_0, tracker_1)
    info = ia.compute_valleys(timeseries)
    with warnings.catch_warnings(record=True) as record:
        ia.test_frame_0_valley (timeseries, info)
    assert len(record) == 1


def test_test_frame_0_true_valley():
    mov_path = movie_path("real_example_short")
    name_list_path = ia.image_folder_to_path_list(mov_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    mask_path = tissue_mask_path("real_example_short")
    mask = ia.read_txt_as_mask(mask_path)
    tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
    timeseries, _, _, _  = ia. compute_abs_position_timeseries(tracker_0, tracker_1)
    info = ia.compute_valleys(timeseries)
    with warnings.catch_warnings(record=True) as record:
        ia.test_frame_0_valley (timeseries, info)
    assert len(record) == 0


def test_split_tracking():
    tracker_0 = np.zeros((10, 100))
    tracker_1 = np.ones((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    tracker_0_all, tracker_1_all = ia.split_tracking(tracker_0, tracker_1, info)
    assert len(tracker_0_all) == 3
    assert len(tracker_1_all) == 3
    for kk in range(0, 3):
        assert tracker_0_all[kk].shape[0] == 10
        assert tracker_0_all[kk].shape[1] == info[kk, 2] - info[kk, 1]
        assert tracker_1_all[kk].shape[0] == 10
        assert tracker_1_all[kk].shape[1] == info[kk, 2] - info[kk, 1]


def test_save_tracking():
    tracker_0 = np.zeros((10, 100))
    tracker_1 = np.ones((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    rot_info = [[100, 150], [1, 0]]
    rot_info = np.asarray(rot_info)
    tracker_0_all, tracker_1_all = ia.split_tracking(tracker_0, tracker_1, info)
    folder_path = example_path("real_example_super_short")
    saved_paths = ia.save_tracking(folder_path=folder_path, tracker_col_all=tracker_0_all, tracker_row_all=tracker_1_all, info=info)
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == info.shape[0] * 2 + 1
    saved_paths = ia.save_tracking(folder_path=folder_path, tracker_col_all=tracker_0_all, tracker_row_all=tracker_1_all, info=info, is_rotated=True, rot_info=rot_info)
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == info.shape[0] * 2 + 1


def test_run_tracking():
    folder_path = example_path("real_example_short")
    fps = 1
    length_scale = 1
    saved_paths = ia.run_tracking(folder_path, fps, length_scale)
    assert len(saved_paths) == 3
    for pa in saved_paths:
        assert pa.is_file()


def test_load_tracking_results():
    folder_path = example_path("real_example_short")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    tracker_row_all, tracker_col_all, info, rot_info = ia.load_tracking_results(folder_path=folder_path)
    assert len(tracker_row_all) == 1
    assert len(tracker_row_all) == 1
    assert len(tracker_col_all) == 1
    assert len(tracker_col_all) == 1
    assert info.shape[1] == 3
    assert rot_info is None
    folder_path = example_path("io_testing_examples")
    folder_path_0 = folder_path.joinpath("fake_example_0").resolve()
    with pytest.raises(FileNotFoundError) as error:
        ia.load_tracking_results(folder_path=folder_path_0)
    assert error.typename == "FileNotFoundError"
    folder_path_1 = folder_path.joinpath("fake_example_1").resolve()
    with pytest.raises(FileNotFoundError) as error:
        ia.load_tracking_results(folder_path=folder_path_1, is_rotated=True)
    assert error.typename == "FileNotFoundError"
    folder_path = example_path("real_example_short")
    _ = ia.run_rotation(folder_path, True)
    tracker_row_all, tracker_col_all, info, rot_info = ia.load_tracking_results(folder_path=folder_path, is_rotated=True)
    assert len(tracker_row_all) == 1
    assert len(tracker_row_all) == 1
    assert len(tracker_col_all) == 1
    assert len(tracker_col_all) == 1
    assert info.shape[1] == 3
    assert rot_info.shape == (2, 2)
    _ = ia.run_scale_and_center_coordinates(folder_path, 100, 100, 0.5, 0.5)
    tracker_row_all, tracker_col_all, info, rot_info = ia.load_tracking_results(folder_path=folder_path, is_translated=True)
    assert len(tracker_row_all) == 1
    assert len(tracker_row_all) == 1
    assert len(tracker_col_all) == 1
    assert len(tracker_col_all) == 1
    assert info.shape[1] == 3
    assert rot_info is None
    _ = ia.run_scale_and_center_coordinates(folder_path, 100, 100, 0.5, 0.5, True)
    tracker_row_all, tracker_col_all, info, rot_info = ia.load_tracking_results(folder_path=folder_path, is_translated=True, is_rotated=True)
    assert len(tracker_row_all) == 1
    assert len(tracker_row_all) == 1
    assert len(tracker_col_all) == 1
    assert len(tracker_col_all) == 1
    assert info.shape[1] == 3
    assert rot_info.shape == (2, 2)
    tracker_row_all, tracker_col_all, info, rot_info = ia.load_tracking_results(folder_path=folder_path, is_translated=True, is_rotated=True, fname="rotated_translated")
    assert len(tracker_row_all) == 1
    assert len(tracker_row_all) == 1
    assert len(tracker_col_all) == 1
    assert len(tracker_col_all) == 1
    assert info.shape[1] == 3
    assert rot_info.shape == (2, 2)


def test_get_title_fname():
    kk = 1
    beat = 1
    ti, fn, fn_gif, fn_row_gif, fn_col_gif = ia.get_title_fname(kk, beat, True, True)
    assert ti == "rotated frame %i, beat %i, with interpolation" % (kk, beat)
    assert fn == "rotated_%04d_disp_with_interp.png" % (kk)
    assert fn_gif == "rotated_abs_disp_with_interp.gif"
    assert fn_row_gif == "rotated_row_disp_with_interp.gif"
    assert fn_col_gif == "rotated_column_disp_with_interp.gif"
    ti, fn, fn_gif, fn_row_gif, fn_col_gif = ia.get_title_fname(kk, beat, True, False)
    assert ti == "rotated frame %i, beat %i" % (kk, beat)
    assert fn == "rotated_%04d_disp.png" % (kk)
    assert fn_gif == "rotated_abs_disp.gif"
    assert fn_row_gif == "rotated_row_disp.gif"
    assert fn_col_gif == "rotated_column_disp.gif"
    ti, fn, fn_gif, fn_row_gif, fn_col_gif = ia.get_title_fname(kk, beat, False, True)
    assert ti == "frame %i, beat %i, with interpolation" % (kk, beat)
    assert fn == "%04d_disp_with_interp.png" % (kk)
    assert fn_gif == "abs_disp_with_interp.gif"
    assert fn_row_gif == "row_disp_with_interp.gif"
    assert fn_col_gif == "column_disp_with_interp.gif"
    ti, fn, fn_gif, fn_row_gif, fn_col_gif = ia.get_title_fname(kk, beat, False, False)
    assert ti == "frame %i, beat %i" % (kk, beat)
    assert fn == "%04d_disp.png" % (kk)
    assert fn_gif == "abs_disp.gif"
    assert fn_row_gif == "row_disp.gif"
    assert fn_col_gif == "column_disp.gif"


def test_compute_min_max_disp ():
    info = [[0, 30, 60], [1, 60, 90], [2, 90, 120]]
    info = np.asarray(info)
    num_beat_frames = 30
    num_pts = 10
    t = np.linspace(0, 1, num_beat_frames)

    tracker_0_beat = np.ones((num_pts, num_beat_frames)) + 2.5*signal.sawtooth(2 * np.pi * t, 0.5)+1
    tracker_0_all = np.hstack((tracker_0_beat,tracker_0_beat,tracker_0_beat,tracker_0_beat,tracker_0_beat))
    tracker_0 = [tracker_0_beat,tracker_0_beat,tracker_0_beat,tracker_0_beat]

    tracker_1_beat = 0.1 * np.ones((num_pts, num_beat_frames)) + np.random.random((num_pts, num_beat_frames))
    tracker_1_all = np.hstack((tracker_1_beat,tracker_1_beat,tracker_1_beat,tracker_1_beat,tracker_1_beat))
    tracker_1 = [tracker_1_beat,tracker_1_beat,tracker_1_beat,tracker_1_beat]
    _, disp_abs_all, disp_0_all, disp_1_all = ia.compute_abs_position_timeseries(tracker_0_all, tracker_1_all)
    min_abs = np.min(disp_abs_all)
    max_abs = np.max(disp_abs_all)
    min_row = np.min(disp_0_all)
    max_row = np.max(disp_0_all)
    min_col = np.min(disp_1_all)
    max_col = np.max(disp_1_all)
    min_abs_disp_clim, max_abs_disp_clim, min_row_disp_clim, max_row_disp_clim, min_col_disp_clim, max_col_disp_clim = ia.compute_min_max_disp(tracker_0, tracker_1, info) 
    assert np.isclose(min_abs,min_abs_disp_clim, atol=0.01)
    assert np.isclose(max_abs,max_abs_disp_clim, atol=0.01)
    assert np.isclose(min_row,min_row_disp_clim, atol=0.01)
    assert np.isclose(max_row,max_row_disp_clim, atol=0.01)
    assert np.isclose(min_col,min_col_disp_clim, atol=0.01)
    assert np.isclose(max_col,max_col_disp_clim, atol=0.01)

def test_create_pngs_gif():
    folder_path = example_path("real_example_short")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    tracker_row_all, tracker_col_all, info, _ = ia.load_tracking_results(folder_path=folder_path)
    mov_path = movie_path("real_example_short")
    name_list_path = ia.image_folder_to_path_list(mov_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    col_max = 3
    col_min = 0
    col_map = plt.cm.viridis
    path_list = ia.create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "abs", col_min, col_max, col_map, save_eps = True)
    for pa in path_list:
        assert pa.is_file()
    gif_path = ia.create_gif(folder_path, path_list, "abs")
    assert gif_path.is_file()
    col_max = 3
    col_min = -3
    path_list = ia.create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "row", col_min, col_max, col_map)
    for pa in path_list:
        assert pa.is_file()
    gif_path = ia.create_gif(folder_path, path_list, "row")
    assert gif_path.is_file()
    path_list = ia.create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "col", col_min, col_max, col_map)
    for pa in path_list:
        assert pa.is_file()
    gif_path = ia.create_gif(folder_path, path_list, "col")
    assert gif_path.is_file()
    # mp4_path = ia.create_mp4(folder_path, gif_path)
    # assert mp4_path.is_file()

def test_run_visualization():
    folder_path = example_path("real_example_short")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    col_min_abs = 0
    col_max_abs = 8
    col_min_row = -3
    col_max_row = 4.5
    col_min_col = -3
    col_max_col = 4.5
    col_map = plt.cm.viridis
    abs_png_path_list, row_png_path_list, col_png_path_list,  abs_gif_path, row_gif_path, col_gif_path = ia.run_visualization(folder_path, True, col_min_abs, col_max_abs, col_min_row, col_max_row, col_min_col, col_max_col, col_map)
    for pa in abs_png_path_list:
        assert pa.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    assert row_gif_path.is_file()
    assert col_gif_path.is_file()
    abs_png_path_list, row_png_path_list, col_png_path_list,  abs_gif_path, row_gif_path, col_gif_path = ia.run_visualization(folder_path, False, col_min_abs, col_max_abs, col_min_row, col_max_row, col_min_col, col_max_col, col_map)
    for pa in abs_png_path_list:
        assert pa.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    assert row_gif_path.is_file()
    assert col_gif_path.is_file()


def interp_fcn_example(xy_vec):
    x_vec = xy_vec[:, 0]
    y_vec = xy_vec[:, 1]
    x_vec_new = np.sin(x_vec) + x_vec * 5.0
    y_vec_new = x_vec * y_vec
    return np.hstack((x_vec_new.reshape((-1, 1)), y_vec_new.reshape((-1, 1))))


def test_interpolate_points():
    x_vec = np.linspace(0, 10, 20)
    y_vec = np.linspace(0, 10, 20)
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    row_col_pos = np.hstack((x_grid.reshape((-1, 1)), y_grid.reshape((-1, 1))))
    x_vec_sample = np.linspace(0.5, 9.5, 4)
    y_vec_sample = np.linspace(0.5, 9.5, 4)
    x_grid_s, y_grid_s = np.meshgrid(x_vec_sample, y_vec_sample)
    row_col_sample = np.hstack((x_grid_s.reshape((-1, 1)), y_grid_s.reshape((-1, 1))))
    row_col_vals = row_col_pos * 2.0
    row_col_sample_vals = ia.interpolate_points(row_col_pos, row_col_vals, row_col_sample)
    assert np.allclose(row_col_sample_vals, row_col_sample * 2.0, atol=0.01)
    row_col_vals = interp_fcn_example(row_col_pos)
    row_col_sample_gt = interp_fcn_example(row_col_sample)
    row_col_sample_vals = ia.interpolate_points(row_col_pos, row_col_vals, row_col_sample)
    assert np.allclose(row_col_sample_gt, row_col_sample_vals, atol=0.01)


def test_compute_distance():
    x1 = 0
    x2 = 10
    y1 = 0
    y2 = 0
    dist = ia.compute_distance(x1, x2, y1, y2)
    assert np.isclose(dist, 10)


def test_compute_unit_vector():
    x1 = 0
    x2 = 10
    y1 = 0
    y2 = 0
    vec = ia.compute_unit_vector(x1, x2, y1, y2)
    assert np.allclose(vec, np.asarray([1, 0]))


def test_insert_borders():
    mask = np.ones((50, 50))
    border = 10
    mask = ia.insert_borders(mask, border)
    assert np.sum(mask) == 30 * 30


def test_axis_from_mask():
    # create an artificial mask
    mask = np.zeros((100, 100))
    mask[25:75, 45:55] = 1
    center_row, center_col, vec = ia.axis_from_mask(mask)
    assert np.allclose(vec, np.asarray([1, 0])) or np.allclose(vec, np.asarray([-1, 0]))
    assert np.isclose(center_row, (25 + 74) / 2.0, atol=2)
    assert np.isclose(center_col, (46 + 53) / 2.0, atol=2)
    mask = np.zeros((100, 100))
    mask[45:55, 25:75] = 1
    center_row, center_col, vec = ia.axis_from_mask(mask)
    assert np.allclose(vec, np.asarray([0, 1])) or np.allclose(vec, np.asarray([0, -1]))
    assert np.isclose(center_col, (25 + 74) / 2.0, atol=2)
    assert np.isclose(center_row, (46 + 53) / 2.0, atol=2)
    # real example
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    center_row, center_col, vec = ia.axis_from_mask(mask)
    assert np.isclose(center_row, mask.shape[0] / 2.0, atol=10)
    assert np.isclose(center_col, mask.shape[0] / 2.0, atol=10)
    assert np.allclose(vec, np.asarray([0, 1]), atol=.1) or np.allclose(vec, np.asarray([0, -1]), atol=.1)
    # rotated example
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
    center_row, center_col, vec = ia.axis_from_mask(mask)
    assert np.isclose(center_row, (10 + 50) / 2.0, atol=4)
    assert np.isclose(center_col, (30 + 80) / 2.0, atol=4)
    assert np.allclose(vec, np.asarray([np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]))


def test_box_to_center_points():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    center_row, center_col = ia.box_to_center_points(box)
    assert np.isclose(center_row, 2.5)
    assert np.isclose(center_col, 5.0)


def test_mask_to_box():
    mask = np.zeros((100, 100))
    mask[25:75, 45:55] = 1
    box = ia.mask_to_box(mask)
    assert box.shape == (4, 2)
    assert np.isclose(np.min(box[:, 0]), 25, atol=3)
    assert np.isclose(np.max(box[:, 0]), 74, atol=3)
    assert np.isclose(np.min(box[:, 1]), 45, atol=3)
    assert np.isclose(np.max(box[:, 1]), 54, atol=3)
    mask[39, 60] = 1
    box = ia.mask_to_box(mask)


def test_corners_to_mask():
    img = np.zeros((100, 100))
    r0 = 20
    r1 = 50
    c0 = 40
    c1 = 80
    new_mask = ia.corners_to_mask(img, r0, r1, c0, c1)
    assert new_mask[30, 60] == 1
    assert new_mask[10, 10] == 0


def test_box_to_unit_vec():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    vec = ia.box_to_unit_vec(box)
    assert np.allclose(vec, np.asarray([0, 1]), atol=.1) or np.allclose(vec, np.asarray([0, -1]), atol=.1)
    box = np.asarray([[0, 0], [0, 5], [10, 5], [10, 0]])
    vec = ia.box_to_unit_vec(box)
    assert np.allclose(vec, np.asarray([1, 0])) or np.allclose(vec, np.asarray([-1, 0]))


def test_rot_vec_to_rot_mat_and_angle():
    vec = [1, 0]
    (rot_mat, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
    assert np.isclose(ang, np.pi / 2.0)
    assert np.allclose(rot_mat, np.asarray([[0, -1], [1, 0]]))
    vec = [0, 1]
    (rot_mat, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
    assert np.isclose(ang, 0)
    assert np.allclose(rot_mat, np.asarray([[1, 0], [0, 1]]))
    vec = [np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]
    (rot_mat, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
    assert np.isclose(ang, np.pi / 4.0)


def test_get_tissue_width():
    mask = np.zeros((512,512))
    mask[226:286,206:300] = 1
    gt_tissue_width = 286 - 226 - 1
    tissue_width = ia.get_tissue_width(mask)
    assert np.isclose(tissue_width,gt_tissue_width,atol=0.01)


def test_save_tissue_width_info():
    folder_path = example_path("real_example_super_short")
    tissue_width = float(52)
    file_path = ia.save_tissue_width_info(folder_path, tissue_width)
    assert file_path.is_file()


def test_check_square_image_true():
    img = np.zeros((512,512))
    square = ia.check_square_image(img)
    assert square == True


def test_check_square_image_false():
    img = np.zeros((512,482))
    square = ia.check_square_image(img)
    assert square == False


def test_rot_image():
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
    center_row, center_col, vec = ia.axis_from_mask(mask)
    (_, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
    new_img = ia.rot_image(mask, center_row, center_col, ang)
    new_center_row, new_center_col, new_vec = ia.axis_from_mask(new_img)
    assert np.isclose(center_row, new_center_row, atol=2)
    assert np.isclose(center_col, new_center_col, atol=2)
    assert np.allclose(new_vec, np.asarray([0, 1]))


def test_rotate_points():
    row_pts = []
    col_pts = []
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
        row_pts.append(kk)
        col_pts.append(kk + 25)
    center_row, center_col, vec = ia.axis_from_mask(mask)
    (rot_mat, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
    row_pts = np.asarray(row_pts)
    col_pts = np.asarray(col_pts)
    new_row_pts, new_col_pts = ia.rotate_points(row_pts, col_pts, rot_mat, center_row, center_col)
    new_img = ia.rot_image(mask, center_row, center_col, ang)
    # plt.figure()
    # plt.imshow(new_img)
    # plt.plot(new_col_pts, new_row_pts, "r.")
    # plt.figure()
    # plt.imshow(mask)
    # plt.plot(col_pts, row_pts, "r.")
    vals = np.nonzero(new_img)
    min_col = np.min(vals[1])
    max_col = np.max(vals[1])
    mean_col = np.mean(vals[1])
    assert np.allclose(new_row_pts, center_row * np.ones(new_row_pts.shape[0]), atol=1)
    assert np.isclose(min_col, np.min(new_col_pts), atol=5)
    assert np.isclose(max_col, np.max(new_col_pts), atol=5)
    assert np.isclose(mean_col, np.mean(new_col_pts), atol=5)


def test_rotate_points_array():
    row_pts = []
    col_pts = []
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
        row_pts.append(kk)
        col_pts.append(kk + 25)
    center_row, center_col, vec = ia.axis_from_mask(mask)
    (rot_mat, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
    new_img = ia.rot_image(mask, center_row, center_col, ang)
    vals = np.nonzero(new_img)
    min_col = np.min(vals[1])
    max_col = np.max(vals[1])
    mean_col = np.mean(vals[1])
    row_pts = np.asarray(row_pts)
    col_pts = np.asarray(col_pts)
    num_reps = 5
    row_pts_array = np.tile(row_pts.reshape((-1, 1)), (1, num_reps))
    col_pts_array = np.tile(col_pts.reshape((-1, 1)), (1, num_reps))
    new_row_pts_array, new_col_pts_array = ia.rotate_points_array(row_pts_array, col_pts_array, rot_mat, center_row, center_col)
    for kk in range(0, num_reps):
        new_row_pts = new_row_pts_array[:, kk]
        new_col_pts = new_col_pts_array[:, kk]
        assert np.allclose(new_row_pts, center_row * np.ones(new_row_pts.shape[0]), atol=1)
        assert np.isclose(min_col, np.min(new_col_pts), atol=5)
        assert np.isclose(max_col, np.max(new_col_pts), atol=5)
        assert np.isclose(mean_col, np.mean(new_col_pts), atol=5)


def test_rotate_pts_all():
    row_pts = []
    col_pts = []
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
        row_pts.append(kk)
        col_pts.append(kk + 25)
    center_row, center_col, vec = ia.axis_from_mask(mask)
    (rot_mat, _) = ia.rot_vec_to_rot_mat_and_angle(vec)
    row_pts = np.asarray(row_pts)
    col_pts = np.asarray(col_pts)
    num_reps = 5
    row_pts_array = np.tile(row_pts.reshape((-1, 1)), (1, num_reps))
    col_pts_array = np.tile(col_pts.reshape((-1, 1)), (1, num_reps))
    row_pts_array_list = [row_pts_array, row_pts_array, row_pts_array]
    col_pts_array_list = [col_pts_array, col_pts_array, col_pts_array]
    rot_row_pts_array_list, rot_col_pts_array_list = ia.rotate_pts_all(row_pts_array_list, col_pts_array_list, rot_mat, center_row, center_col)
    assert np.allclose(rot_row_pts_array_list[0], rot_row_pts_array_list[1])
    assert np.allclose(rot_row_pts_array_list[1], rot_row_pts_array_list[2])
    assert np.allclose(rot_col_pts_array_list[0], rot_col_pts_array_list[1])
    assert np.allclose(rot_col_pts_array_list[1], rot_col_pts_array_list[2])


def test_translate_points():
    num_pts = 10
    num_beat_frames = 100
    pts_row = 24 * np.ones((num_pts, num_beat_frames)) + np.random.random((num_pts, num_beat_frames))
    pts_col = 15 * np.ones((num_pts, num_beat_frames)) + np.random.random((num_pts, num_beat_frames))
    trans_r = 4.5
    trans_c = 3.87
    trans_pts_row,trans_pts_col = ia.translate_points(pts_row, pts_col, trans_r, trans_c)
    diff_trans_pt_row = trans_pts_row - pts_row
    diff_trans_pt_col = trans_pts_col - pts_col
    assert np.allclose(diff_trans_pt_row, trans_r, atol=0.01)
    assert np.allclose(diff_trans_pt_col, trans_c, atol=0.01)

def test_translate_pts_all():
    num_pts = 10
    num_beat_frames = 100
    pts_row = 24 * np.ones((num_pts, num_beat_frames)) + np.random.random((num_pts, num_beat_frames))
    pts_col = 15 * np.ones((num_pts, num_beat_frames)) + np.random.random((num_pts, num_beat_frames))
    row_pts_array_list = [pts_row, pts_row, pts_row]
    col_pts_array_list = [pts_col, pts_col, pts_col]
    trans_r = 4.5
    trans_c = 3.87
    trans_row_pts_array_list, trans_col_pts_array_list = ia.translate_pts_all(row_pts_array_list, col_pts_array_list, trans_r, trans_c)
    diff_trans_pt_row_sample = trans_row_pts_array_list[0] - pts_row
    diff_trans_pt_col_sample = trans_col_pts_array_list[0] - pts_col
    assert len(trans_row_pts_array_list) == len(row_pts_array_list)
    assert len(trans_col_pts_array_list) == len(col_pts_array_list)
    assert np.allclose(diff_trans_pt_row_sample, trans_r, atol=0.01)
    assert np.allclose(diff_trans_pt_col_sample, trans_c, atol=0.01)


def test_rotate_imgs_all():
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
    center_row, center_col, vec = ia.axis_from_mask(mask)
    (_, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
    mask_list = [mask, mask, mask]
    rot_mask_list = ia.rotate_imgs_all(mask_list, ang, center_row, center_col)
    for kk in range(0, len(mask_list)):
        new_img = rot_mask_list[kk]
        new_center_row, new_center_col, new_vec = ia.axis_from_mask(new_img)
        assert np.isclose(center_row, new_center_row, atol=2)
        assert np.isclose(center_col, new_center_col, atol=2)
        assert np.allclose(new_vec, np.asarray([0, 1]))


def test_pad_img_to_square():
    img = np.zeros((320,512))
    square_image, _, _ = ia.pad_img_to_square(img)
    img_r, img_c = square_image.shape
    assert img_r == img_c


def test_pad_all_imgs_to_square():
    img = np.zeros((220,400))
    img_list = [img,img,img]
    padded_list, _, _ = ia.pad_all_imgs_to_square (img_list)
    img_r, img_c = padded_list[0].shape
    assert img_r == img_c
    assert len(padded_list) == len(img_list)


def test_get_rotation_info():
    # check case where all values are provided
    center_row_known = 100
    center_col_known = 200
    vec_known = np.asarray([1, 0])
    (rot_mat_known, ang_known) = ia.rot_vec_to_rot_mat_and_angle(vec_known)
    (center_row_found, center_col_found, rot_mat_found, ang_found, vec_found) = ia.get_rotation_info(center_row_input=center_row_known, center_col_input=center_col_known, vec_input=vec_known)
    assert np.allclose(rot_mat_known, rot_mat_found)
    assert np.isclose(ang_known, ang_found)
    assert np.isclose(center_row_known, center_row_found)
    assert np.isclose(center_col_known, center_col_found)
    assert np.allclose(vec_known, vec_found)
    # check case where only mask is provided
    file_path = tissue_mask_path("real_example_short_rotated")
    mask = ia.read_txt_as_mask(file_path)
    center_row_known, center_col_known, vec_known = ia.axis_from_mask(mask)
    (rot_mat_known, ang_known) = ia.rot_vec_to_rot_mat_and_angle(vec_known)
    (center_row_found, center_col_found, rot_mat_found, ang_found, vec_found) = ia.get_rotation_info(mask=mask)
    assert np.allclose(rot_mat_known, rot_mat_found)
    assert np.isclose(ang_known, ang_found)
    assert np.isclose(center_row_known, center_row_found)
    assert np.isclose(center_col_known, center_col_found)
    assert np.allclose(vec_known, vec_found)
    center_row_known = 10
    (center_row_found, center_col_found, rot_mat_found, ang_found, vec_found) = ia.get_rotation_info(mask=mask, center_row_input=center_row_known)
    assert np.allclose(rot_mat_known, rot_mat_found)
    assert np.isclose(ang_known, ang_found)
    assert np.isclose(center_row_known, center_row_found)
    assert np.isclose(center_col_known, center_col_found)
    assert np.allclose(vec_known, vec_found)


def test_run_rotation():
    folder_path = example_path("real_example_short_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    input_mask = True
    saved_paths = ia.run_rotation(folder_path, input_mask)
    for pa in saved_paths:
        assert pa.is_file()
    input_mask = False
    cri = 100
    cci = 150
    vec_i = np.asarray([1, 0])
    saved_paths = ia.run_rotation(folder_path, input_mask, center_row_input=cri, center_col_input=cci, vec_input=vec_i)
    for pa in saved_paths:
        assert pa.is_file()


def test_rotate_test_img():
    folder_path = example_path("real_example_short_rotated")
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    vec = [1,1]
    rot_mat, ang = ia.rot_vec_to_rot_mat_and_angle(vec)
    center_row = (tiff_list[0].shape[0])/2
    center_col = (tiff_list[0].shape[1])/2
    file_path = ia.rotate_test_img(folder_path, tiff_list, ang, center_row, center_col, rot_mat)
    assert file_path.is_file()

def test_rotate_non_square_test_img():
    folder_path = example_path("real_non_square_example_short_rotated")
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    vec = [1,1]
    rot_mat, ang = ia.rot_vec_to_rot_mat_and_angle(vec)
    center_row = (tiff_list[0].shape[0])/2
    center_col = (tiff_list[0].shape[1])/2
    file_path = ia.rotate_test_img(folder_path, tiff_list, ang, center_row, center_col, rot_mat)
    assert file_path.is_file()

def test_rotate_small_angle_test_img():
    folder_path = example_path("real_example_short_small_angle_rotated")
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    vec = [0,-1]
    rot_mat, ang = ia.rot_vec_to_rot_mat_and_angle(vec)
    center_row = (tiff_list[0].shape[0])/2
    center_col = (tiff_list[0].shape[1])/2
    file_path = ia.rotate_test_img(folder_path, tiff_list, ang, center_row, center_col, rot_mat)
    assert file_path.is_file()

def test_run_rotation_visualization():
    folder_path = example_path("real_example_short_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    input_mask = True
    _ = ia.run_rotation(folder_path, input_mask)
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.run_rotation_visualization(folder_path)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()

def test_run_rotation_visualization_manual_limits():
    folder_path = example_path("real_example_short_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    input_mask = True
    _ = ia.run_rotation(folder_path, input_mask)
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.run_rotation_visualization(folder_path,  automatic_color_constraint = False, col_min_abs=0, col_max_abs=8, col_min_row=-3, col_max_row=4.5, col_min_col=-3, col_max_col=4.5)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()
    

def test_run_rotation_visualization_non_square():
    folder_path = example_path("real_non_square_example_short_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    input_mask = True
    _ = ia.run_rotation(folder_path, input_mask)
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.run_rotation_visualization(folder_path)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()


def test_run_rotation_visualization_small_angle():
    folder_path = example_path("real_example_short_small_angle_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    input_mask = True
    _ = ia.run_rotation(folder_path, input_mask)
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.run_rotation_visualization(folder_path)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()
    

def test_scale_scale_array_in_list():
    arr1 = np.random.random((20, 20))
    arr2 = np.random.random((20, 20))
    arr3 = np.random.random((20, 20))
    tracker_list = [arr1, arr2, arr3]
    new_origin = 10.0
    scale_mult = 10
    updated_tracker_list = ia.scale_array_in_list(tracker_list, new_origin, scale_mult)
    assert np.allclose(scale_mult * (arr1 - new_origin), updated_tracker_list[0])
    assert np.allclose(scale_mult * (arr2 - new_origin), updated_tracker_list[1])
    assert np.allclose(scale_mult * (arr3 - new_origin), updated_tracker_list[2])


def test_run_scale_and_center_coordinates():
    folder_path = example_path("real_example_short")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    pixel_origin_row = 100
    pixel_origin_col = 150
    microns_per_pixel_row = 0.25
    microns_per_pixel_col = 0.25
    saved_paths = ia.run_scale_and_center_coordinates(folder_path, pixel_origin_row, pixel_origin_col, microns_per_pixel_row, microns_per_pixel_col)
    for pa in saved_paths:
        assert pa.is_file()
    input_mask = True
    _ = ia.run_rotation(folder_path, input_mask)
    saved_paths = ia.run_scale_and_center_coordinates(folder_path, pixel_origin_row, pixel_origin_col, microns_per_pixel_row, microns_per_pixel_col, True)


def test_interpolate_pos_from_tracking_arrays_and_interpolate_pos_from_tracking_lists():
    vec1 = np.random.random(100).reshape((-1, 1))
    vec2 = np.random.random(100).reshape((-1, 1))
    vec1b = vec1 + 0.05 * vec1 * vec2
    vec2b = vec2 + 0.05 * vec1 * vec2
    vec1c = vec1 + 0.1 * vec1 * vec2
    vec2c = vec2 + 0.1 * vec1 * vec2
    vec1_arr = np.hstack((vec1, vec1b, vec1c))
    vec2_arr = np.hstack((vec2, vec2b, vec2c))
    vec1_sample = np.random.random(10).reshape((-1, 1))
    vec2_sample = np.random.random(10).reshape((-1, 1))
    vec1b_sample = vec1_sample + 0.05 * vec1_sample * vec2_sample
    vec2b_sample = vec2_sample + 0.05 * vec1_sample * vec2_sample
    vec1c_sample = vec1_sample + 0.1 * vec1_sample * vec2_sample
    vec2c_sample = vec2_sample + 0.1 * vec1_sample * vec2_sample
    vec1_arr_sample = np.hstack((vec1_sample, vec1b_sample, vec1c_sample))
    vec2_arr_sample = np.hstack((vec2_sample, vec2b_sample, vec2c_sample))
    row_col_sample = np.hstack((vec1_sample, vec2_sample))
    row_sample, col_sample = ia.interpolate_pos_from_tracking_arrays(vec1_arr, vec2_arr, row_col_sample)
    assert np.allclose(row_sample, vec1_arr_sample, atol=0.1)
    assert np.allclose(col_sample, vec2_arr_sample, atol=0.1)
    tracker_row_all = [vec1_arr, vec1_arr, vec1_arr]
    tracker_col_all = [vec2_arr, vec2_arr, vec2_arr]
    row_sample_list, col_sample_list = ia.interpolate_pos_from_tracking_lists(tracker_row_all, tracker_col_all, row_col_sample)
    assert len(row_sample_list) == 3
    assert len(col_sample_list) == 3


def test_run_interpolate():
    folder_path = example_path("real_example_short")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    row_vec = np.linspace(215, 305, 12)
    col_vec = np.linspace(120, 400, 30)
    row_grid, col_grid = np.meshgrid(row_vec, col_vec)
    row_sample = row_grid.reshape((-1, 1))
    col_sample = col_grid.reshape((-1, 1))
    row_col_sample = np.hstack((row_sample, col_sample))
    saved_paths = ia.run_interpolate(folder_path, row_col_sample)
    for pa in saved_paths:
        assert pa.is_file()


def test_visualize_interpolate():
    folder_path = example_path("real_example_short")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    input_mask = True
    _ = ia.run_rotation(folder_path, input_mask)
    row_vec = np.linspace(220, 310, 12)
    col_vec = np.linspace(120, 400, 30)
    row_grid, col_grid = np.meshgrid(row_vec, col_vec)
    row_sample = row_grid.reshape((-1, 1))
    col_sample = col_grid.reshape((-1, 1))
    row_col_sample = np.hstack((row_sample, col_sample))
    saved_paths = ia.run_interpolate(folder_path, row_col_sample, "interpolation", True)
    for pa in saved_paths:
        assert pa.is_file()
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.visualize_interpolate(folder_path, is_rotated=True)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()


def test_visualize_interpolate_rotated():
    folder_path = example_path("real_example_short_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    _ = ia.run_rotation(folder_path, True)
    row_vec = np.linspace(230, 320, 12)
    col_vec = np.linspace(105, 375, 26)
    row_grid, col_grid = np.meshgrid(row_vec, col_vec)
    row_sample = row_grid.reshape((-1, 1))
    col_sample = col_grid.reshape((-1, 1))
    row_col_sample = np.hstack((row_sample, col_sample))
    saved_paths = ia.run_interpolate(folder_path, row_col_sample, is_rotated=True)
    for pa in saved_paths:
        assert pa.is_file()
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.visualize_interpolate(folder_path, is_rotated=True)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()


def test_visualize_interpolate_rotated_non_square():
    folder_path = example_path("real_non_square_example_short_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    _ = ia.run_rotation(folder_path, True)
    row_vec = np.linspace(230, 320, 12)
    col_vec = np.linspace(105, 375, 26)
    row_grid, col_grid = np.meshgrid(row_vec, col_vec)
    row_sample = row_grid.reshape((-1, 1))
    col_sample = col_grid.reshape((-1, 1))
    row_col_sample = np.hstack((row_sample, col_sample))
    saved_paths = ia.run_interpolate(folder_path, row_col_sample, is_rotated=True)
    for pa in saved_paths:
        assert pa.is_file()
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.visualize_interpolate(folder_path, is_rotated=True)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()


def test_visualize_interpolate_rotated_small_angle():
    folder_path = example_path("real_example_short_small_angle_rotated")
    fps = 1
    length_scale = 1
    _ = ia.run_tracking(folder_path, fps, length_scale)
    _ = ia.run_rotation(folder_path, True)
    row_vec = np.linspace(230, 320, 12)
    col_vec = np.linspace(105, 375, 26)
    row_grid, col_grid = np.meshgrid(row_vec, col_vec)
    row_sample = row_grid.reshape((-1, 1))
    col_sample = col_grid.reshape((-1, 1))
    row_col_sample = np.hstack((row_sample, col_sample))
    saved_paths = ia.run_interpolate(folder_path, row_col_sample, is_rotated=True)
    for pa in saved_paths:
        assert pa.is_file()
    abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path = ia.visualize_interpolate(folder_path, is_rotated=True)
    for pa in abs_png_path_list:
        assert pa.is_file()
    assert abs_gif_path.is_file()
    for pa in row_png_path_list:
        assert pa.is_file()
    assert row_gif_path.is_file()
    for pa in col_png_path_list:
        assert pa.is_file()
    assert col_gif_path.is_file()


def test_compute_pillar_secnd_moment():
    pillar_width = 163
    pillar_thickness = 33.2
    pillar_diameter = 40
    pillar_secnd_moment_area = ia.compute_pillar_secnd_moment_rectangular(pillar_width, pillar_thickness)
    assert np.isclose(pillar_secnd_moment_area, (pillar_width*(pillar_thickness)**3)/12, atol=1)
    pillar_secnd_moment_area = ia.compute_pillar_secnd_moment_circular(pillar_diameter)
    assert np.isclose(pillar_secnd_moment_area, (np.pi*(pillar_diameter)**4)/64, atol=1)


def test_compute_pillar_stiffnes():
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40   
    pillar_length = 199
    force_location = 163
    pillar_profile_rect = 'rectangular'
    pillar_profile_circ = 'circular'
    pillar_profile_wrong = 'circ'
    I_rect = ia.compute_pillar_secnd_moment_rectangular(pillar_width, pillar_thickness)
    I_circ = ia.compute_pillar_secnd_moment_circular(pillar_diameter)
    pillar_stiffness_gt_rect = (6*pillar_modulus*I_rect)/((force_location**2)*(3*pillar_length-force_location))
    pillar_stiffness_gt_circ = (6*pillar_modulus*I_circ)/((force_location**2)*(3*pillar_length-force_location))
    pillar_stiffness_rect = ia.compute_pillar_stiffnes(pillar_profile_rect, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    pillar_stiffness_circ = ia.compute_pillar_stiffnes(pillar_profile_circ, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    pillar_stiffness_wrong = ia.compute_pillar_stiffnes(pillar_profile_wrong, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    assert np.isclose(pillar_stiffness_rect, pillar_stiffness_gt_rect, atol=0.01)
    assert np.isclose(pillar_stiffness_circ, pillar_stiffness_gt_circ, atol=0.01)
    assert np.isclose(pillar_stiffness_wrong, 0, atol=0.01)
     


def test_compute_pillar_force():
    pillar_stiffness = 0.42
    pillar_avg_deflection = 2
    length_scale = 1
    pillar_force_gt = pillar_stiffness*pillar_avg_deflection*length_scale
    pillar_force = ia.compute_pillar_force(pillar_stiffness, pillar_avg_deflection, length_scale)
    assert np.isclose(pillar_force, pillar_force_gt, atol=0.01)


def test_compute_pillar_position_timeseries():
    num_pts = 3
    num_frames = 100
    tracker_0 = 100 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    tracker_1 = 50 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    disp_abs_mean, mean_disp_0_all, mean_disp_1_all = ia.compute_pillar_position_timeseries(tracker_0, tracker_1)
    assert disp_abs_mean.shape[0] == num_frames
    assert np.max(disp_abs_mean) < np.sqrt(2.0)
    assert mean_disp_0_all.shape[0] == num_frames
    assert np.max(mean_disp_0_all) < 1
    assert mean_disp_1_all.shape[0] == num_frames
    assert np.max(mean_disp_1_all) < 1


def test_pillar_force_all_steps():
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2   
    pillar_diameter = 40  
    pillar_length = 199
    force_location = 163
    num_frames = 100
    pillar_mean_abs_disp = np.sqrt(2)*np.ones(num_frames)
    pillar_mean_disp_row = np.ones(num_frames)
    pillar_mean_disp_col = np.ones(num_frames)
    length_scale = 1

    pillar_k = ia.compute_pillar_stiffnes(pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    pillar_F_abs, pillar_F_row, pillar_F_col = ia.pillar_force_all_steps(pillar_mean_abs_disp,pillar_mean_disp_row, pillar_mean_disp_col, pillar_stiffnes, pillar_profile, 
                                                                         pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale)
    assert len(pillar_F_abs) == num_frames
    assert np.allclose(pillar_F_abs/pillar_k, pillar_mean_abs_disp, atol=0.01)
    assert len(pillar_F_row) == num_frames
    assert np.allclose(pillar_F_row/pillar_k, pillar_mean_disp_row, atol=0.01)
    assert len(pillar_F_col) == num_frames
    assert np.allclose(pillar_F_col/pillar_k, pillar_mean_disp_col, atol=0.01)


def test_pillar_force_all_steps_given_K():
    pillar_stiffnes = 0.42
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    num_frames = 100
    pillar_mean_abs_disp = np.sqrt(2)*np.ones(num_frames)
    pillar_mean_disp_row = np.ones(num_frames)
    pillar_mean_disp_col = np.ones(num_frames)
    length_scale = 1

    pillar_F_abs, pillar_F_row, pillar_F_col = ia.pillar_force_all_steps(pillar_mean_abs_disp,pillar_mean_disp_row, pillar_mean_disp_col, pillar_stiffnes, pillar_profile, 
                                                                         pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale)
    pillar_k = ia.compute_pillar_stiffnes(pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    assert len(pillar_F_abs) == num_frames
    assert np.allclose(pillar_F_abs/pillar_stiffnes, pillar_mean_abs_disp, atol=0.01)
    assert len(pillar_F_row) == num_frames
    assert np.allclose(pillar_F_row/pillar_stiffnes, pillar_mean_disp_row, atol=0.01)
    assert len(pillar_F_col) == num_frames
    assert np.allclose(pillar_F_col/pillar_stiffnes, pillar_mean_disp_col, atol=0.01)
    assert pillar_stiffnes != pillar_k


def test_save_pillar_position():
    folder_path = example_path("real_example_pillar_short")
    tracker_row_all = np.zeros((10, 100))
    tracker_col_all = np.zeros((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    saved_paths = ia.save_pillar_position(folder_path=folder_path, tracker_row_all=tracker_row_all, tracker_col_all = tracker_col_all, info = info, split_track = False, fname = None)
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == 3


def test_save_pillar_position_split():
    folder_path = example_path("real_example_pillar_short")
    tracker_row_all = np.zeros((10, 100))
    tracker_col_all = np.zeros((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    saved_paths = ia.save_pillar_position(folder_path=folder_path, tracker_row_all=tracker_row_all, tracker_col_all = tracker_col_all, info = info, split_track = True, fname = None)
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == 6


def test_save_pillar_position_fname():
    folder_path = example_path("real_example_pillar_short")
    tracker_row_all = np.zeros((10, 100))
    tracker_col_all = np.zeros((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    saved_paths = ia.save_pillar_position(folder_path=folder_path, tracker_row_all=tracker_row_all, tracker_col_all = tracker_col_all, info = info, split_track = False, fname = 'Pillar_1_')
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == 3


def test_save_pillar_force():
    folder_path = example_path("real_example_pillar_short")
    pillar_force_abs = np.sqrt(2)*np.ones(100)
    pillar_force_row = np.ones(100)
    pillar_force_col = np.ones(100)
    saved_paths = ia.save_pillar_force(folder_path=folder_path, pillar_force_abs = pillar_force_abs, pillar_force_row = pillar_force_row, pillar_force_col = pillar_force_col, fname=None)
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == 3


def test_save_pillar_force_fname():
    folder_path = example_path("real_example_pillar_short")
    pillar_force_abs = np.sqrt(2)*np.ones(100)
    pillar_force_row = np.ones(100)
    pillar_force_col = np.ones(100)
    saved_paths = ia.save_pillar_force(folder_path=folder_path, pillar_force_abs = pillar_force_abs, pillar_force_row = pillar_force_row, pillar_force_col = pillar_force_col, fname="Pillar_1_")
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == 3


def test_run_pillar_tracking():
    folder_path = example_path("real_example_pillar_short")
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    split_track = False
    saved_paths_pos, saved_paths_force = ia.run_pillar_tracking(folder_path, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, split_track)
    for pa_p in saved_paths_pos:
        assert pa_p.is_file()
    for pa_f in saved_paths_force:
        assert pa_f.is_file()


def test_run_one_pillar_tracking_split():
    folder_path = example_path("real_example_pillar_short")
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    saved_paths_pos, saved_paths_force = ia.run_pillar_tracking(folder_path, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, split_track = True)
    for pa_p in saved_paths_pos:
        assert pa_p.is_file()
    for pa_f in saved_paths_force:
        assert pa_f.is_file()


def test_load_pillar_tracking_results():
    folder_path = example_path("real_example_pillar_short")
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    _, _ = ia.run_pillar_tracking(folder_path, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, split_track = False)
    _, _, _, _, _ = ia.load_pillar_tracking_results(folder_path=folder_path, split_track = False)
    
    folder_path = example_path("real_example_one_pillar_short")
    _, _ = ia.run_pillar_tracking(folder_path, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, split_track = True)
    _, _, _, _, _ = ia.load_pillar_tracking_results(folder_path=folder_path, split_track = True)

    folder_path = example_path("io_testing_examples")
    folder_path_0 = folder_path.joinpath("fake_example_0").resolve()
    with pytest.raises(FileNotFoundError) as error:
        ia.load_pillar_tracking_results(folder_path=folder_path_0)
    assert error.typename == "FileNotFoundError"
    folder_path_1 = folder_path.joinpath("fake_example_3").resolve()
    with pytest.raises(FileNotFoundError) as error:
        ia.load_pillar_tracking_results(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"


def test_visualize_pillar_tracking():
    folder_path = example_path("real_example_pillar_short")
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    _, _ = ia.run_pillar_tracking(folder_path, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, split_track = False)
    saved_path = ia.visualize_pillar_tracking(folder_path, split_track = False)
    assert saved_path.is_file
    
    folder_path = example_path("real_example_one_pillar_short")
    _, _ = ia.run_pillar_tracking(folder_path, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, split_track = True)
    saved_path = ia.visualize_pillar_tracking(folder_path, split_track = True)
    assert saved_path.is_file

    
