import matplotlib.pyplot as plt
from microbundlecomputelite import image_analysis as ia
import numpy as np
import os
from pathlib import Path
import pytest
from skimage import io
from skimage.transform import estimate_transform, warp


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


def test_get_tracking_param_dicts():
    feature_params, lk_params = ia.get_tracking_param_dicts()
    assert feature_params["maxCorners"] == 10000
    assert feature_params["qualityLevel"] == 0.1  # 0.005
    assert feature_params["minDistance"] == 5
    assert feature_params["blockSize"] == 5
    assert lk_params["winSize"][0] == 10
    assert lk_params["winSize"][1] == 10
    assert lk_params["maxLevel"] == 10
    assert lk_params["criteria"][1] == 10
    assert lk_params["criteria"][2] == 0.03


def test_adjust_feature_param_dicts():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8 = ia.uint16_to_uint8(img)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params_init, _ = ia.get_tracking_param_dicts()
    qL_init = feature_params_init["qualityLevel"]
    feature_params_new = ia.adjust_feature_param_dicts(feature_params_init, img_uint8, mask)
    assert qL_init >= feature_params_new["qualityLevel"]


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
    compare = np.abs(track_points_1 - track_points_0)
    assert np.max(compare[:, 0, 0]) < lk_params["winSize"][0]
    assert np.max(compare[:, 0, 1]) < lk_params["winSize"][1]


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
    tracker_0, tracker_1 = ia.track_all_steps(img_list_uint8, mask)
    diff_0 = np.abs(tracker_0[:, 0] - tracker_0[:, -1])
    diff_1 = np.abs(tracker_1[:, 0] - tracker_1[:, -1])
    _, lk_params = ia.get_tracking_param_dicts()
    assert np.max(diff_0) < lk_params["winSize"][0]
    assert np.max(diff_1) < lk_params["winSize"][1]


def test_track_all_steps_warping():
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
    tracker_0, tracker_1 = ia.track_all_steps(img_list_uint8, mask)
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
    disp_abs_mean, disp_abs_all = ia.compute_abs_position_timeseries(tracker_0, tracker_1)
    assert disp_abs_mean.shape[0] == num_frames
    assert np.max(disp_abs_mean) < np.sqrt(2.0)
    assert disp_abs_all.shape[1] == num_frames


def test_get_time_segment_param_dicts():
    time_seg_params = ia.get_time_segment_param_dicts()
    assert time_seg_params["peakDist"] == 20


def test_adjust_time_seg_params():
    x = np.linspace(0, 500 * np.pi * 2.0, 500)
    timeseries = np.sin(x / (np.pi * 2.0) / 20 - np.pi / 2.0)
    time_seg_params = ia.get_time_segment_param_dicts()
    pd_init = time_seg_params["peakDist"]
    time_seg_params_new = ia.adjust_time_seg_params(time_seg_params, timeseries)
    assert time_seg_params_new["peakDist"] > pd_init


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
    saved_paths = ia.run_tracking(folder_path)
    assert len(saved_paths) == 3
    for pa in saved_paths:
        assert pa.is_file()


def test_load_tracking_results():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
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


def test_create_pngs_gif():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
    tracker_row_all, tracker_col_all, info, _ = ia.load_tracking_results(folder_path=folder_path)
    mov_path = movie_path("real_example_short")
    name_list_path = ia.image_folder_to_path_list(mov_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    col_max = 3
    col_map = plt.cm.viridis
    path_list = ia.create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, col_max, col_map)
    for pa in path_list:
        assert pa.is_file()
    gif_path = ia.create_gif(folder_path, path_list)
    assert gif_path.is_file()
    # mp4_path = ia.create_mp4(folder_path, gif_path)
    # assert mp4_path.is_file()


def test_get_title_fname():
    kk = 1
    beat = 1
    ti, fn, fn_gif = ia.get_title_fname(kk, beat, True, True)
    assert ti == "rotated frame %i, beat %i, with interpolation" % (kk, beat)
    assert fn == "rotated_%04d_disp_with_interp.png" % (kk)
    assert fn_gif == "rotated_abs_disp_with_interp.gif"
    ti, fn, fn_gif = ia.get_title_fname(kk, beat, True, False)
    assert ti == "rotated frame %i, beat %i" % (kk, beat)
    assert fn == "rotated_%04d_disp.png" % (kk)
    assert fn_gif == "rotated_abs_disp.gif"
    ti, fn, fn_gif = ia.get_title_fname(kk, beat, False, True)
    assert ti == "frame %i, beat %i, with interpolation" % (kk, beat)
    assert fn == "%04d_disp_with_interp.png" % (kk)
    assert fn_gif == "abs_disp_with_interp.gif"
    ti, fn, fn_gif = ia.get_title_fname(kk, beat, False, False)
    assert ti == "frame %i, beat %i" % (kk, beat)
    assert fn == "%04d_disp.png" % (kk)
    assert fn_gif == "abs_disp.gif"


def test_run_visualization():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
    col_max = 3
    col_map = plt.cm.viridis
    png_path_list, gif_path = ia.run_visualization(folder_path, col_max, col_map)
    for pa in png_path_list:
        assert pa.is_file()
    assert gif_path.is_file()


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
    _ = ia.run_tracking(folder_path)
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


def test_run_rotation_visualization():
    folder_path = example_path("real_example_short_rotated")
    _ = ia.run_tracking(folder_path)
    input_mask = True
    _ = ia.run_rotation(folder_path, input_mask)
    png_path_list, gif_path = ia.run_rotation_visualization(folder_path)
    for pa in png_path_list:
        assert pa.is_file()
    assert gif_path.is_file()


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
    _ = ia.run_tracking(folder_path)
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
    _ = ia.run_tracking(folder_path)
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
    _ = ia.run_tracking(folder_path)
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
    png_path_list, gif_path = ia.visualize_interpolate(folder_path, col_max=3, is_rotated=True)
    for pa in png_path_list:
        assert pa.is_file()
    assert gif_path.is_file()


def test_visualize_interpolate_rotated():
    folder_path = example_path("real_example_short_rotated")
    _ = ia.run_tracking(folder_path)
    _ = ia.run_rotation(folder_path, True)
    row_vec = np.linspace(195, 195 + 90, 12)
    col_vec = np.linspace(125, 400, 30)
    row_grid, col_grid = np.meshgrid(row_vec, col_vec)
    row_sample = row_grid.reshape((-1, 1))
    col_sample = col_grid.reshape((-1, 1))
    row_col_sample = np.hstack((row_sample, col_sample))
    saved_paths = ia.run_interpolate(folder_path, row_col_sample, is_rotated=True)
    for pa in saved_paths:
        assert pa.is_file()
    png_path_list, gif_path = ia.visualize_interpolate(folder_path, col_max=3, is_rotated=True)
    for pa in png_path_list:
        assert pa.is_file()
    assert gif_path.is_file()
