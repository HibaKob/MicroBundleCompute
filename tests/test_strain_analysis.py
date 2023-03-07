from microbundlecomputelite import image_analysis as ia
from microbundlecomputelite import strain_analysis as sa
import numpy as np
from pathlib import Path
import pytest


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


def test_box_to_mask():
    mask = np.zeros((100, 100))
    box = np.asarray([[30, 50], [30, 60], [50, 60], [50, 50]])
    new_mask = sa.box_to_mask(mask, box)
    assert new_mask[10, 10] == 0
    assert new_mask[35, 55] == 1


def test_corners_to_mask():
    img = np.zeros((100, 100))
    r0 = 20
    r1 = 50
    c0 = 40
    c1 = 80
    new_mask = sa.corners_to_mask(img, r0, r1, c0, c1)
    assert new_mask[30, 60] == 1
    assert new_mask[10, 10] == 0


def test_box_to_bound():
    r0 = 20
    r1 = 50
    c0 = 40
    c1 = 80
    box = np.asarray([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])
    r0_found, r1_found, c0_found, c1_found = sa.box_to_bound(box)
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
    box_found = sa.bound_to_box(r0, r1, c0, c1)
    assert np.allclose(box_known, box_found)


def test_shrink_pair():
    v0 = 100
    v1 = 200
    sf = 0.1
    new_v0, new_v1 = sa.shrink_pair(v0, v1, sf)
    diff = new_v1 - new_v0
    assert diff == 90


def test_shrink_box():
    r0 = 100
    r1 = 200
    c0 = 200
    c1 = 400
    box = np.asarray([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])
    shrink_row = 0.1
    shrink_col = 0.2
    box_shrink = sa.shrink_box(box, shrink_row, shrink_col)
    r0_found, r1_found, c0_found, c1_found = sa.box_to_bound(box_shrink)
    assert r1_found - r0_found == 90
    assert c1_found - c0_found == 160


def test_remove_pillar_region():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    new_mask = sa.remove_pillar_region(mask)
    assert np.sum(mask) > np.sum(new_mask)
    box = ia.mask_to_box(mask)
    _, _, c0, c1 = sa.box_to_bound(box)
    new_box = ia.mask_to_box(new_mask)
    _, _, c0_new, c1_new = sa.box_to_bound(new_box)
    assert c0_new > c0
    assert c1_new < c1


def test_create_sub_domains():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    tile_box_list, tile_dim_pix, num_tile_row, num_tile_col = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    assert len(tile_box_list) == 15
    assert tile_dim_pix == 40
    assert num_tile_row == 3
    assert num_tile_col == 5
    tile_style = 2
    tile_box_list, tile_dim_pix, num_tile_row, num_tile_col = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=10, num_tile_col=6, tile_style=tile_style)
    assert len(tile_box_list) == 60
    assert num_tile_row == 10
    assert num_tile_col == 6


def test_is_in_box():
    box = sa.bound_to_box(10, 100, 30, 900)
    assert sa.is_in_box(box, 1000, 300) is False
    assert sa.is_in_box(box, 50, 400) is True
    assert sa.is_in_box(box, 50, 1000) is False
    assert sa.is_in_box(box, 3, 500) is False


def test_isolate_sub_domain_markers():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    tile_box_list, _, _, _ = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col]
    tracker_row_all = [tracker_row]
    for sd_box in tile_box_list:
        r0, r1, c0, c1 = sa.box_to_bound(sd_box)
        sd_tracker_row_all, sd_tracker_col_all = sa.isolate_sub_domain_markers(tracker_row_all, tracker_col_all, sd_box)
        assert len(sd_tracker_row_all) == 1
        assert len(sd_tracker_col_all) == 1
        assert np.max(sd_tracker_row_all[0][:, 0]) < r1
        assert np.min(sd_tracker_row_all[0][:, 0]) > r0
        assert np.max(sd_tracker_col_all[0][:, 0]) < c1
        assert np.min(sd_tracker_col_all[0][:, 0]) > c0


def test_compute_F_from_Lambda_mat():
    Lambda_0 = np.asarray([[1, 0], [0, 1]])
    F_known = np.asarray([[10, 0], [0, 10]])
    Lambda_t = np.dot(F_known, Lambda_0)
    F_found = sa.compute_F_from_Lambda_mat(Lambda_0, Lambda_t)
    assert np.allclose(F_known, F_found)
    Lambda_0 = np.asarray([[1, 0, 10, 4, -3, -2, 1], [0, 1, 3, -2, -2, -10, 0]])
    F_known = np.asarray([[10, 1], [-1, 3]])
    Lambda_t = np.dot(F_known, Lambda_0)
    F_found = sa.compute_F_from_Lambda_mat(Lambda_0, Lambda_t)


def test_compute_Lambda_from_pts():
    row_pos = np.asarray([1, 2, 3, 4, 5])
    col_pos = np.asarray([6, 7, 8, 9, 10])
    Lambda_mat = sa.compute_Lambda_from_pts(row_pos, col_pos)
    num_vec = 0
    for kk in range(0, row_pos.shape[0]):
        for jj in range(kk + 1, row_pos.shape[0]):
            num_vec += 1
    assert Lambda_mat.shape[1] == num_vec
    assert Lambda_mat.shape[0] == 2


def test_get_box_center():
    box = np.asarray([[10, 20], [10, 70], [30, 70], [30, 20]])
    center_row, center_col = sa.get_box_center(box)
    assert center_row == 20
    assert center_col == 45


def test_compute_sub_domain_strain():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    tile_box_list, _, _, _ = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col]
    tracker_row_all = [tracker_row]
    sd_box = tile_box_list[0]
    sd_tracker_row_all, sd_tracker_col_all = sa.isolate_sub_domain_markers(tracker_row_all, tracker_col_all, sd_box)
    sd_F_arr = sa.compute_sub_domain_strain(sd_tracker_row_all[0], sd_tracker_col_all[0])
    assert np.isclose(sd_F_arr[0, 0], 1)
    assert np.isclose(sd_F_arr[0, 1], 0)
    assert np.isclose(sd_F_arr[0, 2], 0)
    assert np.isclose(sd_F_arr[0, 3], 1)
    assert sd_F_arr[0, 0] != sd_F_arr[1, 0]
    assert sd_F_arr.shape[1] == 4
    assert sd_F_arr.shape[0] == sd_tracker_row_all[0].shape[1]


def test_compute_sub_domain_position():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    tile_box_list, _, _, _ = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col]
    tracker_row_all = [tracker_row]
    sd_box = tile_box_list[0]
    center_row, center_col = sa.get_box_center(sd_box)
    sd_tracker_row_all, sd_tracker_col_all = sa.isolate_sub_domain_markers(tracker_row_all, tracker_col_all, sd_box)
    sd_row, sd_col = sa.compute_sub_domain_position(sd_tracker_row_all[0], sd_tracker_col_all[0], sd_box)
    assert sd_row.shape[0] == tracker_col.shape[1]
    assert sd_col.shape[0] == tracker_row.shape[1]
    assert np.isclose(sd_row[0], center_row)
    assert np.isclose(sd_col[0], center_col)


def test_compute_sub_domain_position_strain_all():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    sd_box_list, _, _, _ = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col, tracker_col]
    tracker_row_all = [tracker_row, tracker_row]
    sub_domain_F_all, sub_domain_row_all, sub_domain_col_all = sa.compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sd_box_list)
    num_subdomains = len(sd_box_list)
    assert len(sub_domain_F_all) == num_subdomains
    assert len(sub_domain_row_all) == num_subdomains
    assert len(sub_domain_col_all) == num_subdomains
    assert len(sub_domain_F_all[0]) == len(tracker_row_all)
    assert len(sub_domain_row_all[0]) == len(tracker_row_all)
    assert len(sub_domain_col_all[0]) == len(tracker_row_all)
    assert sub_domain_F_all[0][0].shape[0] == tracker_row.shape[1]
    assert sub_domain_row_all[0][0].shape[0] == tracker_row.shape[1]
    assert sub_domain_col_all[0][0].shape[0] == tracker_row.shape[1]


def test_format_F_for_save():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    sd_box_list, _, _, _ = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col, tracker_col]
    tracker_row_all = [tracker_row, tracker_row]
    sub_domain_F_all, _, _ = sa.compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sd_box_list)
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all = sa.format_F_for_save(sub_domain_F_all)
    num_subdomains = len(sd_box_list)
    num_frames = tracker_col.shape[1]
    assert len(sub_domain_F_rr_all) == len(tracker_row_all)
    assert len(sub_domain_F_rc_all) == len(tracker_row_all)
    assert len(sub_domain_F_cr_all) == len(tracker_row_all)
    assert len(sub_domain_F_cc_all) == len(tracker_row_all)
    assert sub_domain_F_rr_all[0].shape == (num_subdomains, num_frames)
    assert sub_domain_F_rc_all[0].shape == (num_subdomains, num_frames)
    assert sub_domain_F_cr_all[0].shape == (num_subdomains, num_frames)
    assert sub_domain_F_cc_all[0].shape == (num_subdomains, num_frames)


def test_format_sd_row_col_for_save():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    sd_box_list, _, _, _ = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col, tracker_col]
    tracker_row_all = [tracker_row, tracker_row]
    _, sub_domain_row_all, sub_domain_col_all = sa.compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sd_box_list)
    sub_domain_row_all_new, sub_domain_col_all_new = sa.format_sd_row_col_for_save(sub_domain_row_all, sub_domain_col_all)
    num_subdomains = len(sd_box_list)
    num_frames = tracker_col.shape[1]
    assert len(sub_domain_row_all_new) == len(tracker_row_all)
    assert len(sub_domain_col_all_new) == len(tracker_row_all)
    assert sub_domain_row_all_new[0].shape == (num_subdomains, num_frames)
    assert sub_domain_col_all_new[0].shape == (num_subdomains, num_frames)


def test_save_sub_domain_strain():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    sd_box_list, tile_dim_pix, num_tile_row, num_tile_col = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col, tracker_col]
    tracker_row_all = [tracker_row, tracker_row]
    sub_domain_F_all, sub_domain_row_all, sub_domain_col_all = sa.compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sd_box_list)
    strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix, tile_dim_pix], [200, 200], [0, 1]])
    input_path = example_path("real_example_super_short")
    saved_paths = sa.save_sub_domain_strain(input_path, sub_domain_F_all, sub_domain_row_all, sub_domain_col_all, strain_sub_domain_info)
    for sp in saved_paths:
        assert sp.is_file()


def test_load_sub_domain_strain():
    file_path = tissue_mask_path("real_example_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    sd_box_list, tile_dim_pix, num_tile_row, num_tile_col = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col, tracker_col]
    tracker_row_all = [tracker_row, tracker_row]
    sub_domain_F_all, sub_domain_row_all, sub_domain_col_all = sa.compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sd_box_list)
    strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix, tile_dim_pix], [200, 200], [0, 1]])
    input_path = example_path("real_example_short")
    _ = ia.run_tracking(input_path)
    _ = sa.save_sub_domain_strain(input_path, sub_domain_F_all, sub_domain_row_all, sub_domain_col_all, strain_sub_domain_info)
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all, sub_domain_row_all, sub_domain_col_all, _, _ = sa.load_sub_domain_strain(input_path)
    num_subdomains = len(sd_box_list)
    num_frames = tracker_col.shape[1]
    assert len(sub_domain_F_rr_all) == len(tracker_row_all)
    assert len(sub_domain_F_rc_all) == len(tracker_row_all)
    assert len(sub_domain_F_cr_all) == len(tracker_row_all)
    assert len(sub_domain_F_cc_all) == len(tracker_row_all)
    assert sub_domain_F_rr_all[0].shape == (num_subdomains, num_frames)
    assert sub_domain_F_rc_all[0].shape == (num_subdomains, num_frames)
    assert sub_domain_F_cr_all[0].shape == (num_subdomains, num_frames)
    assert sub_domain_F_cc_all[0].shape == (num_subdomains, num_frames)
    assert len(sub_domain_row_all) == len(tracker_row_all)
    assert len(sub_domain_col_all) == len(tracker_row_all)
    assert sub_domain_row_all[0].shape == (num_subdomains, num_frames)
    assert sub_domain_col_all[0].shape == (num_subdomains, num_frames)


def test_load_sub_domain_strain_errors():
    folder_path = example_path("io_testing_examples")
    folder_path_0 = folder_path.joinpath("fake_example_0").resolve()
    with pytest.raises(FileNotFoundError) as error:
        sa.load_sub_domain_strain(folder_path=folder_path_0)
    assert error.typename == "FileNotFoundError"
    folder_path_1 = folder_path.joinpath("fake_example_1").resolve()
    with pytest.raises(FileNotFoundError) as error:
        sa.load_sub_domain_strain(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"


def test_F_to_Ecc():
    F_rr = 1
    F_rc = 0
    F_cr = 0
    F_cc = 1
    E_cc = sa.F_to_Ecc(F_rr, F_rc, F_cr, F_cc)
    assert E_cc == 0
    F_rr = 2
    F_rc = 0
    F_cr = 0
    F_cc = 4
    E_cc = sa.F_to_Ecc(F_rr, F_rc, F_cr, F_cc)
    assert np.isclose(E_cc, 7.5)


def test_F_to_Ecc_all():
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tile_style = 1
    sd_box_list, _, _, _ = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    tracker_col, tracker_row = ia.track_all_steps(img_list_uint8, mask)
    tracker_col_all = [tracker_col, tracker_col]
    tracker_row_all = [tracker_row, tracker_row]
    sub_domain_F_all, _, _ = sa.compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sd_box_list)
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all = sa.format_F_for_save(sub_domain_F_all)
    sub_domain_Ecc_all = sa.F_to_Ecc_all(sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all)
    assert len(sub_domain_Ecc_all) == len(tracker_col_all)
    assert sub_domain_Ecc_all[0].shape == (len(sd_box_list), sub_domain_F_rr_all[0].shape[1])


def test_run_sub_domain_strain_analysis():
    input_path = example_path("real_example_short")
    _ = ia.run_tracking(input_path)
    saved_paths = sa.run_sub_domain_strain_analysis(input_path)
    for sap in saved_paths:
        assert sap.is_file()


def test_get_text_str():
    known_str = "A1"
    found_str = sa.get_text_str(0, 0)
    assert known_str == found_str
    known_str = "E7"
    found_str = sa.get_text_str(4, 6)
    assert known_str == found_str


def test_png_sub_domains_numbered():
    folder_path = example_path("real_example_short")
    movie_folder = movie_path("real_example_short")
    name_list_path = ia.image_folder_to_path_list(movie_folder)
    tiff_list = ia.read_all_tiff(name_list_path)
    example_tiff = tiff_list[0]
    _ = ia.run_tracking(folder_path)
    _ = sa.run_sub_domain_strain_analysis(folder_path)
    _, _, _, _, sub_domain_row_all, sub_domain_col_all, _, strain_info = sa.load_sub_domain_strain(folder_path)
    # update to using loaded info!
    sub_domain_row = sub_domain_row_all[0]
    sun_domain_col = sub_domain_col_all[0]
    sub_domain_side = strain_info[1, 0]
    num_sd_row = int(strain_info[0, 0])
    num_sd_col = int(strain_info[0, 1])
    img_path = sa.png_sub_domains_numbered(folder_path, example_tiff, sub_domain_row, sun_domain_col, sub_domain_side, num_sd_row, num_sd_col)
    assert img_path.is_file()


def test_png_sub_domain_strain_timeseries_all():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
    _ = sa.run_sub_domain_strain_analysis(folder_path)
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all, sub_domain_row_all, sub_domain_col_all, info, strain_info = sa.load_sub_domain_strain(folder_path)
    sub_domain_Ecc_all = sa.F_to_Ecc_all(sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all)
    mask_path = tissue_mask_path("real_example_short")
    mask = ia.read_txt_as_mask(mask_path)
    tile_style = 1
    _, _, num_sd_row, num_sd_col = sa.create_sub_domains(mask, pillar_clip_fraction=0.5, shrink_row=0.1, shrink_col=0.1, tile_dim_pix=40, num_tile_row=5, num_tile_col=3, tile_style=tile_style)
    img_path = sa.png_sub_domain_strain_timeseries_all(folder_path, sub_domain_Ecc_all, num_sd_row, num_sd_col)
    for ig in img_path:
        assert ig.is_file()


def test_pngs_sub_domain_strain_and_gif():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
    _ = sa.run_sub_domain_strain_analysis(folder_path)
    movie_folder = movie_path("real_example_short")
    name_list_path = ia.image_folder_to_path_list(movie_folder)
    tiff_list = ia.read_all_tiff(name_list_path)
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all, sub_domain_row_all, sub_domain_col_all, info, strain_info = sa.load_sub_domain_strain(folder_path)
    sub_domain_side = strain_info[1, 0]
    sub_domain_Ecc_all = sa.F_to_Ecc_all(sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all)
    saved_paths = sa.pngs_sub_domain_strain(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, sub_domain_Ecc_all, sub_domain_side, info)
    for sap in saved_paths:
        assert sap.is_file()
    gif_path = sa.create_gif(folder_path, saved_paths)
    assert gif_path.is_file()


def test_visualize_sub_domain_strain():
    folder_path = example_path("real_example_short")
    png_path_list, gif_path, loc_legend_path, timeseries_path_list = sa.visualize_sub_domain_strain(folder_path)
    for sap in png_path_list:
        assert sap.is_file()
    assert gif_path.is_file()
    assert loc_legend_path.is_file()
    for tpl in timeseries_path_list:
        assert tpl.is_file()
