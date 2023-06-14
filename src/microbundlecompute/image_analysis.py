import cv2
import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.animation as animation
# import moviepy.editor as mp
import numpy as np
import os
from pathlib import Path
from skimage import measure
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial import distance_matrix
from scipy.signal import find_peaks
from skimage import exposure
from skimage import img_as_ubyte
from skimage import io
from skimage.transform import rotate
from typing import List, Tuple, Union
import warnings


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


def create_folder(folder_path: Path, new_folder_name: str) -> Path:
    """Given a path to a directory and a folder name. Will create a directory in the given directory."""
    new_path = folder_path.joinpath(new_folder_name).resolve()
    if new_path.exists() is False:
        os.mkdir(new_path)
    return new_path


def uint16_to_uint8(img_16: np.ndarray) -> np.ndarray:
    """Given a uint16 image. Will normalize + rescale and convert to uint8."""
    img_8 = img_as_ubyte(exposure.rescale_intensity(img_16))
    return img_8


def bool_to_uint8(arr_bool: np.ndarray) -> np.ndarray:
    """Given a boolean array. Will return a uint8 array."""
    arr_uint8 = (1. * arr_bool).astype("uint8")
    return arr_uint8


def uint16_to_uint8_all(img_list: List) -> List:
    """Given an image list of uint16. Will return the same list all as uint8."""
    uint8_list = []
    for img in img_list:
        img8 = uint16_to_uint8(img)
        uint8_list.append(img8)
    return uint8_list


def read_txt_as_mask(file_path: Path) -> np.ndarray:
    """Given a path to a saved txt file array. Will return an array formatted as unit8."""
    img = np.loadtxt(file_path)
    img_uint8 = bool_to_uint8(img)
    return img_uint8


def get_tracking_param_dicts() -> dict:
    """Will return dictionaries specifying the feature parameters and tracking parameters.
    In future, these may vary based on version."""
    feature_params = dict(maxCorners=10000, qualityLevel=0.1, minDistance=3, blockSize=3)
    window = 5
    
    lk_params = dict(winSize=(window, window), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return feature_params, lk_params


def mask_to_track_points(img_uint8: np.ndarray, mask: np.ndarray, feature_params: dict) -> np.ndarray:
    """Given an image and a mask. Will return the good features to track within the mask region."""
    # ensure that the mask is uint8
    mask_uint8 = bool_to_uint8(mask)
    track_points_0 = cv2.goodFeaturesToTrack(img_uint8, mask=mask_uint8, **feature_params)
    return track_points_0


def shrink_pair(v0: int, v1: int, sf: float) -> int:
    """Given two values and an amount to shrink their difference by. Will return the new values."""
    dist = v1 - v0
    new_v0 = v0 + int(dist * sf * 0.5)
    new_v1 = v1 - int(dist * sf * 0.5)
    return new_v0, new_v1


def remove_pillar_region(mask: np.ndarray, clip_fraction: float = 0.5, clip_columns: bool = True, clip_rows: bool = False ) -> np.ndarray:
    """Given a mask. Will approximately remove the pillar region."""
    box_mask = mask_to_box(mask)
    r0, r1, c0, c1 = box_to_bound(box_mask)
    new_r0, new_r1, new_c0, new_c1 = r0, r1, c0, c1
    # because we only accept oriented masks, we can perform this operation by clipping columns and rows
    if clip_columns:
        new_c0, new_c1 = shrink_pair(c0, c1, clip_fraction)
    if clip_rows:
        new_r0, new_r1 = shrink_pair(r0,r1, clip_fraction)
    clip_mask = corners_to_mask(mask, new_r0, new_r1, new_c0, new_c1)
    # perform clipping
    new_mask = (mask * clip_mask > 0).astype("uint8")
    return new_mask


def box_to_bound(box: np.ndarray) -> int:
    """Given a grid aligned box. Will convert it to bounds format."""
    r0 = int(np.min(box[:, 0]))
    r1 = int(np.max(box[:, 0]))
    c0 = int(np.min(box[:, 1]))
    c1 = int(np.max(box[:, 1]))
    return r0, r1, c0, c1


def bound_to_box(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Given some bounds. Will return them formatted as a box"""
    box = np.asarray([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])
    return box


def is_in_box(box: np.ndarray, rr: int, cc: int) -> bool:
    """Given a box and a point. Will return True if the point is inside the box, False otherwise."""
    r0, r1, c0, c1 = box_to_bound(box)
    if rr > r0 and rr < r1 and cc > c0 and cc < c1:
        return True
    else:
        return False


def sub_division_markers(tracker_row: np.ndarray, tracker_col: np.ndarray, sd_box: np.ndarray) -> List:
    """Given tracker row and column arrays and sub-domain box. """
    div_tracker_row = []
    for kk in range(0, len(tracker_row)):
        rr = tracker_row[kk]
        cc = tracker_col[kk]
        if is_in_box(sd_box, rr, cc):
            div_tracker_row.append(rr)
    div_tracker_row = np.asarray(div_tracker_row)
    num_sub_pts = len(div_tracker_row)

    return num_sub_pts


def sub_division_mask(mask: np.ndarray, box: np.ndarray) -> float:
    r0, r1, c0, c1 = box_to_bound(box)
    mask_in_div = np.sum(mask[r0:r1,c0:c1])
    return mask_in_div


def compute_local_coverage(mask: np.ndarray, track_0_pts: np.ndarray, sub_division_dim_pix: int = 20) -> List:
    """Given a mask and tracker points. Will compute the local marker coverage within each mask subdivision."""
    center_row, center_col,rot_mat, ang,_ = get_rotation_info(center_row_input=None, center_col_input=None, vec_input=None, mask=mask)
    rot_mask = rot_image(mask, center_row, center_col,ang)
    new_mask = remove_pillar_region(rot_mask, clip_fraction = 0.3, clip_columns = True, clip_rows = True)

    box_mask = mask_to_box(new_mask)
    r0, r1, c0, c1 = box_to_bound(box_mask)
    
    num_tile_row = int(np.floor((r1 - r0) / sub_division_dim_pix))
    num_tile_col = int(np.floor((c1 - c0) / sub_division_dim_pix))

    marker_row_orig = track_0_pts[:, 0, 1]
    marker_col_orig = track_0_pts[:, 0, 0]
    
    marker_row, marker_col = rotate_points(marker_row_orig, marker_col_orig, rot_mat, center_row, center_col)
    
    all_local_coverage = []
    
    for rr in range(0, num_tile_row):
        for cc in range(0, num_tile_col):
            tile_box = bound_to_box(r0 + rr * sub_division_dim_pix, r0 + (rr + 1) * sub_division_dim_pix, c0 + cc * sub_division_dim_pix, c0 + (cc + 1) * sub_division_dim_pix)
            div_mask = sub_division_mask(new_mask,tile_box)
                
            if div_mask==0:
                pass
            else:
                num_div_markers = sub_division_markers(marker_row,marker_col,tile_box)
                
                if num_div_markers == 0:
                    local_coverage = 0
                else: 
                    local_coverage = div_mask/num_div_markers
               
                all_local_coverage.append(local_coverage)
    return all_local_coverage


def adjust_qualityLevel(feature_params: dict, img_uint8: np.ndarray, mask: np.ndarray, min_coverage: Union[float, int]):
    track_points_0 = mask_to_track_points(img_uint8, mask, feature_params)
    qualityLevel = feature_params["qualityLevel"]
    coverage = np.sum(mask) / track_points_0.shape[0]
    iter = 0
    
    while coverage > min_coverage and iter < 15:
        
        qualityLevel = qualityLevel * 10 ** (np.log10(0.1) / 10)  # this value raised to 10 is 0.1, so it will lower quality by an order of magnitude in 10 iterations
        feature_params["qualityLevel"] = qualityLevel
        
        track_points_0 = mask_to_track_points(img_uint8, mask, feature_params)
        
        coverage = np.sum(mask) / track_points_0.shape[0]
        iter +=1
    return qualityLevel, coverage


def adjust_feature_param_dicts(feature_params: dict, img_uint8: np.ndarray, mask: np.ndarray, min_coverage: Union[float, int] = 40, min_local_coverage: Union[float, int] = 50) -> dict:
    """Given feature parameters, an image, and a mask. Will automatically update the feature quality and minimum distance to ensure sufficient coverage.
    (min_coverage refers to the number of pixels that should be attributed to 1 tracking point)"""
    _,_ = adjust_qualityLevel(feature_params, img_uint8, mask, min_coverage)
    track_points_0 = mask_to_track_points(img_uint8, mask, feature_params)
    local_coverage = compute_local_coverage(mask,track_points_0)
    local_coverage = sorted(local_coverage, reverse=True)
    minDist = feature_params["minDistance"]  
    iter = 0
    while np.min(local_coverage[:3]) >= min_local_coverage and iter < 2:
        minDist+=1
        feature_params["minDistance"] = minDist
        _,_ = adjust_qualityLevel(feature_params, img_uint8, mask, min_coverage)
        track_points_0 = mask_to_track_points(img_uint8, mask, feature_params)
        local_coverage = compute_local_coverage(mask,track_points_0)
        local_coverage = sorted(local_coverage, reverse=True)
        iter += 1
    return feature_params
  
def track_one_step(img_uint8_0: np.ndarray, img_uint8_1: np.ndarray, track_points_0: np.ndarray, lk_params: dict):
    """Given img_0, img_1, tracking points p0, and tracking parameters.
    Will return the tracking points new location. Note that for now standard deviation and error are ignored."""
    track_points_1, _, _ = cv2.calcOpticalFlowPyrLK(img_uint8_0, img_uint8_1, track_points_0, None, **lk_params)
    return track_points_1

def track_all_steps(img_list_uint8: List, mask: np.ndarray,feature_params: dict, lk_params: dict) -> np.ndarray:
    """Given the image list in order, mask, feature parameters and tracking parameters. Will run tracking through the whole img list in order.
    Note that the returned order of tracked points will match order_list."""
    img_0 = img_list_uint8[0]
    track_points = mask_to_track_points(img_0, mask, feature_params)
    num_track_pts = track_points.shape[0]
    num_imgs = len(img_list_uint8)
    tracker_0 = np.zeros((num_track_pts, num_imgs))
    tracker_1 = np.zeros((num_track_pts, num_imgs))
    for kk in range(0, num_imgs - 1):
        tracker_0[:, kk] = track_points[:, 0, 0]
        tracker_1[:, kk] = track_points[:, 0, 1]
        img_0 = img_list_uint8[kk]
        img_1 = img_list_uint8[kk + 1]
        track_points = track_one_step(img_0, img_1, track_points, lk_params)
    tracker_0[:, kk + 1] = track_points[:, 0, 0]
    tracker_1[:, kk + 1] = track_points[:, 0, 1]
    return tracker_0, tracker_1


def track_all_steps_with_adjust_param_dicts(img_list_uint8: List, mask: np.ndarray) -> dict:
    """Given image list and mask. Will automatically update the feature parameters and tracking parameters to ensure accurate and robust tracking.
    Will return tracked points through the whole img list in order. """
    feature_params, lk_params = get_tracking_param_dicts()
    img_0 = img_list_uint8[0]
    feature_params = adjust_feature_param_dicts(feature_params, img_0, mask)
    tracker_0, tracker_1 = track_all_steps(img_list_uint8, mask, feature_params, lk_params)
    _, disp_abs_all, _, _  = compute_abs_position_timeseries(tracker_0, tracker_1)
    max_disp_abs = np.max(disp_abs_all)
    window_size = lk_params["winSize"][0]
    iter = 0  
    while window_size < max_disp_abs and iter < 15:
        window_size += 5
        lk_params["winSize"] = (window_size,window_size)
        tracker_0, tracker_1 = track_all_steps(img_list_uint8, mask, feature_params, lk_params)
        _, disp_abs_all, _, _  = compute_abs_position_timeseries(tracker_0, tracker_1)
        max_disp_abs = np.max(disp_abs_all)
        iter += 1
    if max_disp_abs < 1:
        warnings.warn("All tracked displacements are subpixel displacements. Results have limited accuracy!")
    return tracker_0, tracker_1


def compute_abs_position_timeseries(tracker_0: np.ndarray, tracker_1: np.ndarray) -> np.ndarray:
    """Given tracker arrays. Will return single timeseries of absolute displacement."""
    disp_0_all = np.zeros(tracker_0.shape)
    disp_1_all = np.zeros(tracker_1.shape)
    for kk in range(tracker_0.shape[1]):
        disp_0_all[:, kk] = tracker_0[:, kk] - tracker_0[:, 0]
        disp_1_all[:, kk] = tracker_1[:, kk] - tracker_1[:, 0]
    disp_abs_all = (disp_0_all ** 2.0 + disp_1_all ** 2.0) ** 0.5
    disp_abs_mean = np.mean(disp_abs_all, axis=0)
    return disp_abs_mean, disp_abs_all, -1*disp_0_all, disp_1_all


def get_time_segment_param_dicts() -> dict:
    """Will return dictionaries specifying the parameters for timeseries segmentation."""
    time_seg_params = dict(peakDist=20, prom = 0.1)
    return time_seg_params


def adjust_time_seg_params(time_seg_params: dict, timeseries: np.ndarray) -> dict:
    """Given time segmentation parameters and a timeseries mean absolute displacement. 
    Will automatically update the minimum distance between peaks to ensure more robust time segmentation."""
    timeseries_offset = timeseries - np.mean(timeseries)
    signs = np.sign(timeseries_offset)
    diff = np.diff(signs)
    indices_of_zero_crossing = np.where(diff)[0]
    total_points = np.diff(indices_of_zero_crossing)
    period = np.mean(total_points) * 2.0
    time_seg_params["peakDist"] = period * 0.75
    time_seg_params["prom"] = 0.1
    return time_seg_params


def compute_valleys(timeseries: np.ndarray) -> np.ndarray:
    """Given a timeseries. Will compute peaks and valleys."""
    time_seg_params = get_time_segment_param_dicts()
    time_seg_params = adjust_time_seg_params(time_seg_params, timeseries)
    peaks, _ = find_peaks(timeseries, distance=time_seg_params["peakDist"], prominence=time_seg_params["prom"])
    valleys = []
    for kk in range(0, len(peaks) - 1):
        valleys.append(int(0.5 * peaks[kk] + 0.5 * peaks[kk + 1]))
    info = []
    for kk in range(0, len(valleys) - 1):
        # beat number, start index wrt movie, end index wrt movie
        info.append([kk, valleys[kk], valleys[kk + 1]])
    return np.asarray(info)


def compute_peaks(timeseries: np.ndarray) -> np.ndarray:
    """Given a timeseries. Will compute peaks."""
    time_seg_params = get_time_segment_param_dicts()
    time_seg_params = adjust_time_seg_params(time_seg_params, timeseries)
    peaks, _ = find_peaks(timeseries, distance=time_seg_params["peakDist"], prominence=time_seg_params["prom"])
    return peaks


def compute_beat_frequency(info: np.ndarray, fps: int) -> float:
    """Given valleys and frames/sec (fps). Will compute the frequency of the beats."""
    num_beats = info.shape[0]
    valley_pairs = info.T[1:]
    if num_beats > 2:
        period_all = valley_pairs[1]-valley_pairs[0]
        period = np.mean(period_all[1:-1])
    else:
        period = np.mean(valley_pairs[1]-valley_pairs[0])
    freq = (1/period)*fps
    return freq


def compute_beat_amplitude(timeseries: np.ndarray, tracker_row_all: List, tracker_col_all: List, info: np.ndarray, length_scale: float) -> float:
    """Given a timeseries and split tracking results. Will compute the amplitude of the beats."""
    all_beat_ampl = []
    num_beats = info.shape[0]
    peaks = compute_peaks(timeseries)
    actual_peaks = peaks[1:-1]
    for beat in range(0, num_beats):
        tracker_row = tracker_row_all[beat]
        tracker_col = tracker_col_all[beat]
        mean_disp_all, _, _, _ = compute_abs_position_timeseries(tracker_row, tracker_col)
        idx_start = info[beat, 1]
        beat_peak = int(actual_peaks[beat]) - idx_start
        beat_ampl = mean_disp_all[beat_peak]
        all_beat_ampl.append(beat_ampl)   
    if num_beats > 2:
        ampl_px = np.mean(all_beat_ampl[1:-1])
    else:
        ampl_px = np.mean(all_beat_ampl)
    
    ampl = ampl_px*length_scale
    return ampl


def save_beat_info(folder_path: Path,frequency: float, amplitude: float) -> Path:
    """Given frequency and amplitude. Will save the results into a text file."""
    res_folder_path = create_folder(folder_path,"results")
    file_path = res_folder_path.joinpath("beat_info.txt").resolve()
    beat_info = np.asarray([frequency,amplitude])
    np.savetxt(str(file_path), beat_info)
    return file_path


def test_frame_0_valley(timeseries: np.ndarray, info: np.ndarray):
    """Given full tracked mean absolute displacement and info. Will check if the movie does not start from a valley position."""
    valley_pairs = info.T[1:]
    valleys = np.unique(valley_pairs.ravel())
    valleys = np.asarray(valleys, dtype=int)
    min_valley_mean_abs_disp = np.min(timeseries[valleys])
    tracked_timeseries = timeseries[valleys[0]:]
    min_nonzero_mean_abs_disp = np.min(tracked_timeseries[tracked_timeseries!=0])
    valley_error = (min_valley_mean_abs_disp - min_nonzero_mean_abs_disp)/min_nonzero_mean_abs_disp*100
    if valley_error <= 20:
        pass
    else:
        warnings.warn('Input video does not start from a valley position. Consider adjusting the video using the preprocessing function "adjust_first_valley".')


def split_tracking(tracker_0: np.ndarray, tracker_1: np.ndarray, info: np.ndarray) -> Path:
    """Given full tracking arrays and info. Will split tracking array by beat according to info."""
    tracker_0_all = []
    tracker_1_all = []
    for kk in range(0, info.shape[0]):
        ix_start = info[kk, 1]
        ix_end = info[kk, 2]
        tracker_0_all.append(tracker_0[:, ix_start:ix_end])
        tracker_1_all.append(tracker_1[:, ix_start:ix_end])
    return tracker_0_all, tracker_1_all


def save_tracking(*, folder_path: Path, tracker_row_all: List, tracker_col_all: List, info: np.ndarray = None, is_rotated: bool = False, rot_info: np.ndarray = None, is_translated: bool = False, fname: str = None) -> List:
    """Given tracking results. Will save as text files."""
    new_path = create_folder(folder_path, "results")
    num_beats = len(tracker_row_all)
    saved_paths = []
    for kk in range(0, num_beats):
        if fname is not None:
            file_path = new_path.joinpath(fname + "_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath(fname + "_beat%i_col.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all[kk])
        elif is_translated and is_rotated:
            file_path = new_path.joinpath("rotated_translated_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("rotated_translated_beat%i_col.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all[kk])
        elif is_translated:
            file_path = new_path.joinpath("translated_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("translated_beat%i_col.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all[kk])
        elif is_rotated:
            file_path = new_path.joinpath("rotated_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("rotated_beat%i_col.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all[kk])
        else:
            file_path = new_path.joinpath("beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("beat%i_col.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all[kk])
    if info is not None:
        file_path = new_path.joinpath("info.txt").resolve()
        np.savetxt(str(file_path), info)
    if rot_info is not None:
        file_path = new_path.joinpath("rot_info.txt").resolve()
        np.savetxt(str(file_path), rot_info)
    saved_paths.append(file_path)
    return saved_paths


def run_tracking(folder_path: Path, fps: int, length_scale: float) -> List:
    """Given a folder path. Will perform tracking and save results as text files."""
    # read images and mask file
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    img_list_uint8 = uint16_to_uint8_all(tiff_list)
    mask_file_path = folder_path.joinpath("masks").resolve().joinpath("tissue_mask.txt").resolve()
    mask = read_txt_as_mask(mask_file_path)
    # get tissue width
    tissue_width = get_tissue_width(mask)
    # perform tracking
    tracker_0, tracker_1 = track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
    # perform timeseries segmentation
    timeseries, _, _, _ = compute_abs_position_timeseries(tracker_0, tracker_1)
    info =  compute_valleys(timeseries)
    # test if frame 0 is a valley frame or not: warn the user
    test_frame_0_valley(timeseries, info)
    # split tracking results
    tracker_0_all, tracker_1_all = split_tracking(tracker_0, tracker_1, info)
    # compute beat frequency and amplitude
    frequency = compute_beat_frequency(info, fps)
    amplitude = compute_beat_amplitude(timeseries, tracker_0_all, tracker_1_all, info, length_scale)
    # save tracking results
    saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=tracker_0_all, tracker_row_all=tracker_1_all, info=info)
    # save beat info
    save_beat_info(folder_path, frequency, amplitude)
    # save tissue width 
    save_tissue_width_info(folder_path, tissue_width)
    #return saved_paths
    return saved_paths


def load_tracking_results(*, folder_path: Path, is_rotated: bool = False, is_translated: bool = False, fname: str = None) -> List:
    """Given the folder path. Will load tracking results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present -- tracking must be run before visualization")
    rev_file_0 = res_folder_path.joinpath("rotated_beat0_col.txt").resolve()
    if is_rotated:
        if rev_file_0.is_file() is False:
            raise FileNotFoundError("rotated tracking results are not present -- rotated tracking must be run before rotated visualization")
    num_files = len(glob.glob(str(res_folder_path) + "/beat*.txt"))
    num_beats = int((num_files) / 2)
    tracker_row_all = []
    tracker_col_all = []
    for kk in range(0, num_beats):
        if fname is not None:
            tracker_row = np.loadtxt(str(res_folder_path) + "/" + fname + "_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/" + fname + "_beat%i_col.txt" % (kk))
        elif is_rotated and is_translated:
            tracker_row = np.loadtxt(str(res_folder_path) + "/rotated_translated_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/rotated_translated_beat%i_col.txt" % (kk))
        elif is_translated:
            tracker_row = np.loadtxt(str(res_folder_path) + "/translated_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/translated_beat%i_col.txt" % (kk))
        elif is_rotated:
            tracker_row = np.loadtxt(str(res_folder_path) + "/rotated_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/rotated_beat%i_col.txt" % (kk))
        else:
            tracker_row = np.loadtxt(str(res_folder_path) + "/beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/beat%i_col.txt" % (kk))
        tracker_row_all.append(tracker_row)
        tracker_col_all.append(tracker_col)
    info = np.loadtxt(str(res_folder_path) + "/info.txt")
    info_reshape = np.reshape(info, (-1, 3))
    if is_rotated:
        rot_info = np.loadtxt(str(res_folder_path) + "/rot_info.txt")
    else:
        rot_info = None
    return tracker_row_all, tracker_col_all, info_reshape, rot_info


def get_title_fname(kk: int, beat: int, is_rotated: bool = False, include_interp: bool = False) -> str:
    """XX -- TODO: add in the option to include a custom title."""
    if is_rotated and include_interp:
        ti = "rotated frame %i, beat %i, with interpolation" % (kk, beat)
        fn = "rotated_%04d_disp_with_interp.png" % (kk)
        fn_gif = "rotated_abs_disp_with_interp.gif"
        fn_row_gif = "rotated_row_disp_with_interp.gif"
        fn_col_gif = "rotated_column_disp_with_interp.gif"
    elif is_rotated:
        ti = "rotated frame %i, beat %i" % (kk, beat)
        fn = "rotated_%04d_disp.png" % (kk)
        fn_gif = "rotated_abs_disp.gif"
        fn_row_gif = "rotated_row_disp.gif"
        fn_col_gif = "rotated_column_disp.gif"
    elif include_interp:
        ti = "frame %i, beat %i, with interpolation" % (kk, beat)
        fn = "%04d_disp_with_interp.png" % (kk)
        fn_gif = "abs_disp_with_interp.gif"
        fn_row_gif = "row_disp_with_interp.gif"
        fn_col_gif = "column_disp_with_interp.gif"
    else:
        ti = "frame %i, beat %i" % (kk, beat)
        fn = "%04d_disp.png" % (kk)
        fn_gif = "abs_disp.gif"
        fn_row_gif = "row_disp.gif"
        fn_col_gif = "column_disp.gif"
    return ti, fn, fn_gif, fn_row_gif, fn_col_gif


def compute_min_max_disp (
    tracker_row_all: List,
    tracker_col_all: List,
    info: np.ndarray
) -> float: 
    """Given tracking results. Will find the minimum and maximum displacement over all beats."""
    num_beats = info.shape[0]
    min_disp_all, max_disp_all = [],[]
    min_row_disp_all, max_row_disp_all = [],[]
    min_col_disp_all, max_col_disp_all = [],[]

    for beat in range(0, num_beats):
        tracker_row = tracker_row_all[beat]
        tracker_col = tracker_col_all[beat]
        _, disp_all, disp_0_all, disp_1_all = compute_abs_position_timeseries(tracker_row, tracker_col)
        
        min_disp_all.append(np.min(disp_all))
        max_disp_all.append(np.max(disp_all))
        
        min_row_disp_all.append(np.min(disp_0_all))
        max_row_disp_all.append(np.max(disp_0_all))
        
        min_col_disp_all.append(np.min(disp_1_all))
        max_col_disp_all.append(np.max(disp_1_all))

    min_abs_disp_clim = np.percentile(min_disp_all,5)
    max_abs_disp_clim = np.percentile(max_disp_all,85)
    
    min_row_disp_clim = np.percentile(min_row_disp_all,5)
    max_row_disp_clim = np.percentile(max_row_disp_all,85)
    
    min_col_disp_clim = np.percentile(min_col_disp_all,5)
    max_col_disp_clim = np.percentile(max_col_disp_all,85)
    
    return min_abs_disp_clim, max_abs_disp_clim, min_row_disp_clim, max_row_disp_clim, min_col_disp_clim, max_col_disp_clim


def create_pngs(
    folder_path: Path,
    tiff_list: List,
    tracker_row_all: List,
    tracker_col_all: List,
    info: np.ndarray,
    output: str, 
    col_min: Union[float, int],
    col_max: Union[float, int],
    col_map: object,
    *,
    is_rotated: bool = False,
    include_interp: bool = False,
    interp_tracker_row_all: List = None,
    interp_tracker_col_all: List = None,
    save_eps: bool = False
) -> List:
    """Given tracking results. Will create png version of the visualizations."""
    vis_folder_path = create_folder(folder_path, "visualizations")
    main_pngs_folder_path = create_folder(vis_folder_path, "pngs")
    
    if output == 'abs':
        pngs_folder_path = create_folder(main_pngs_folder_path, "pngs_abs")
    elif output == 'row':
        pngs_folder_path = create_folder(main_pngs_folder_path, "pngs_row")
    elif output == 'col':
        pngs_folder_path = create_folder(main_pngs_folder_path, "pngs_col")
    
    path_list = []
    num_beats = info.shape[0]
    for beat in range(0, num_beats):
        tracker_row = tracker_row_all[beat]
        tracker_col = tracker_col_all[beat]
        _, disp_all, disp_0_all, disp_1_all = compute_abs_position_timeseries(tracker_row, tracker_col)
        if include_interp:
            interp_tracker_row = interp_tracker_row_all[beat]
            interp_tracker_col = interp_tracker_col_all[beat]
            _, interp_disp_all, interp_disp_0_all, interp_disp_1_all = compute_abs_position_timeseries(interp_tracker_row, interp_tracker_col)
        start_idx = int(info[beat, 1])
        end_idx = int(info[beat, 2])
        for kk in range(start_idx, end_idx):
            ti, fn, _, _, _ = get_title_fname(kk, beat, is_rotated, include_interp)
            plt.figure()
            plt.imshow(tiff_list[kk], cmap=plt.cm.gray)
            jj = kk - start_idx
            
            if output == 'abs':
                # plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=disp_all[:, jj], s=10, cmap=col_map, vmin=0, vmax=col_max)
                plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=disp_all[:, jj], s=4, cmap=col_map, vmin=col_min, vmax=col_max)
                if include_interp:
                    # plt.scatter(interp_tracker_col[:, jj], interp_tracker_row[:, jj], c=interp_disp_all[:, jj], s=7, cmap=col_map, vmin=0, vmax=col_max, linewidths=1, edgecolors=(0, 0, 0))
                    plt.scatter(interp_tracker_col[:, jj], interp_tracker_row[:, jj], c=interp_disp_all[:, jj], s=7, cmap=col_map, vmin=col_min, vmax=col_max, linewidths=0.5, edgecolors=(0, 0, 0))
                cbar = plt.colorbar()
                cbar.ax.get_yaxis().labelpad = 15
                cbar.set_label("absolute displacement (pixels)", rotation=270)
                
            elif output == 'row':
                plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=disp_0_all[:, jj], s=4, cmap=col_map, vmin=col_min, vmax=col_max)
                if include_interp:
                    plt.scatter(interp_tracker_col[:, jj], interp_tracker_row[:, jj], c=interp_disp_0_all[:, jj], s=7, cmap=col_map, vmin=col_min, vmax=col_max, linewidths=0.5, edgecolors=(0, 0, 0))
                cbar = plt.colorbar()
                cbar.ax.get_yaxis().labelpad = 15
                cbar.set_label("row (vertical) displacement (pixels)", rotation=270)
                
            elif output == 'col':
                plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=disp_1_all[:, jj], s=4, cmap=col_map, vmin=col_min, vmax=col_max)
                if include_interp:
                    plt.scatter(interp_tracker_col[:, jj], interp_tracker_row[:, jj], c=interp_disp_1_all[:, jj], s=7, cmap=col_map, vmin=col_min, vmax=col_max, linewidths=0.5, edgecolors=(0, 0, 0))
                cbar = plt.colorbar()
                cbar.ax.get_yaxis().labelpad = 15
                cbar.set_label("column (horizontal) displacement (pixels)", rotation=270)
            
            plt.title(ti)
            plt.axis("off")
            path = pngs_folder_path.joinpath(fn).resolve()
            plt.savefig(str(path))
            if save_eps:
                plt.savefig(str(path)[0:-4]+'.eps', format='eps')
            plt.close()
            path_list.append(path)
    return path_list


def create_gif(folder_path: Path, png_path_list: List, output: str, is_rotated: bool = False, include_interp: bool = False) -> Path:
    """Given the pngs path list. Will create a gif."""
    img_list = []
    img = plt.imread(png_path_list[0])
    img_r, img_c,_ = img.shape
    fig, ax = plt.subplots(figsize=(img_c/100,img_r/100))
    plt.axis('off')
    plt.tight_layout(pad=0.08, h_pad=None, w_pad=None, rect=None)
    for pa in png_path_list:
        img = ax.imshow(plt.imread(pa),animated=True)
        img_list.append([img])
    _, _, fn_gif, fn_row_gif, fn_col_gif = get_title_fname(0, 0, is_rotated, include_interp)    
    if output == 'abs':
        gif_path = folder_path.joinpath("visualizations").resolve().joinpath(fn_gif).resolve()
    elif output == 'row':
        gif_path = folder_path.joinpath("visualizations").resolve().joinpath(fn_row_gif).resolve()
    elif output == 'col':
        gif_path = folder_path.joinpath("visualizations").resolve().joinpath(fn_col_gif).resolve()
    ani = animation.ArtistAnimation(fig, img_list,interval=100)
    ani.save(gif_path,dpi=100)
    plt.close()
    return gif_path
# ==================================================================================================

# def create_mp4(folder_path: Path, gif_path: Path) -> Path:
#     """Given the gif path. Will create a mp4."""
#     clip = mp.VideoFileClip(str(gif_path))
#     mp4_path = folder_path.joinpath("visualizations").resolve().joinpath("abs_disp.mp4").resolve()
#     clip.write_videofile(str(mp4_path))
#     return mp4_path

# ==================================================================================================

def run_visualization(folder_path: Path, automatic_color_constraint: bool = True, col_min_abs: Union[int, float] = 0, col_max_abs: Union[int, float] = 8, col_min_row: Union[int, float] = -3, col_max_row: Union[int, float] = 4.5, col_min_col: Union[int, float] = -3, col_max_col: Union[int, float] = 4.5, col_map: object = plt.cm.viridis) -> List:
    """Given a folder path where tracking has already been run. Will save visualizations."""
    # read image files
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    # read tracking results
    tracker_row_all, tracker_col_all, info, _ = load_tracking_results(folder_path=folder_path)
    if automatic_color_constraint:
        # find limits of colormap
        col_min_abs, col_max_abs, col_min_row, col_max_row, col_min_col, col_max_col = compute_min_max_disp(tracker_row_all,tracker_col_all,info)
    # create pngs
    abs_png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "abs", col_min_abs, col_max_abs, col_map,save_eps = False)
    row_png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "row", col_min_row, col_max_row, col_map,save_eps = False)
    col_png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "col", col_min_col, col_max_col, col_map,save_eps = False)
    # create gif
    abs_gif_path = create_gif(folder_path, abs_png_path_list, "abs")
    row_gif_path = create_gif(folder_path, row_png_path_list, "row")
    col_gif_path = create_gif(folder_path, col_png_path_list, "col")
    # create mp4
    # mp4_path = create_mp4(folder_path, gif_path)
    return abs_png_path_list, row_png_path_list, col_png_path_list,  abs_gif_path, row_gif_path, col_gif_path


def interpolate_points(
    row_col_pos: np.ndarray,
    row_col_vals: np.ndarray,
    row_col_sample: np.ndarray,
) -> np.ndarray:
    """Given row/column positions, row/column values, and sample positions.
    Will interpolate the values and return values at the sample positions."""
    # fit interpolation function and perform interpolation
    row_sample_vals = RBFInterpolator(row_col_pos, row_col_vals[:, 0])(row_col_sample)
    col_sample_vals = RBFInterpolator(row_col_pos, row_col_vals[:, 1])(row_col_sample)
    row_col_sample_vals = np.hstack((row_sample_vals.reshape((-1, 1)), col_sample_vals.reshape((-1, 1))))
    return row_col_sample_vals


def interpolate_pos_from_tracking_arrays(
    tracker_row: np.ndarray,
    tracker_col: np.ndarray,
    row_col_sample: np.ndarray,
) -> np.ndarray:
    """Given tracking results for one beat and sample locations.
    Will return interpolated tracking results at the sample points."""
    num_frames = tracker_row.shape[1]
    num_sample_pts = row_col_sample.shape[0]
    row_sample = np.zeros((num_sample_pts, num_frames))
    col_sample = np.zeros((num_sample_pts, num_frames))
    row_sample[:, 0] = row_col_sample[:, 0]
    col_sample[:, 0] = row_col_sample[:, 1]
    row_col_pos = np.hstack((tracker_row[:, 0].reshape((-1, 1)), tracker_col[:, 0].reshape((-1, 1))))
    for kk in range(1, num_frames):
        row_col_vals = np.hstack((tracker_row[:, kk].reshape((-1, 1)) - tracker_row[:, 0].reshape((-1, 1)), tracker_col[:, kk].reshape((-1, 1)) - tracker_col[:, 0].reshape((-1, 1))))
        row_col_sample_vals = interpolate_points(row_col_pos, row_col_vals, row_col_sample)
        row_sample[:, kk] = row_col_sample_vals[:, 0] + row_col_sample[:, 0]
        col_sample[:, kk] = row_col_sample_vals[:, 1] + row_col_sample[:, 1]
    return row_sample, col_sample


def interpolate_pos_from_tracking_lists(
    tracker_row_all: List,
    tracker_col_all: List,
    row_col_sample: np.ndarray,
) -> List:
    """Given tracking results in a list and interpolation sample points. Will interpolate for all frames."""
    row_sample_list = []
    col_sample_list = []
    num_beats = len(tracker_row_all)
    for kk in range(0, num_beats):
        row_sample, col_sample = interpolate_pos_from_tracking_arrays(tracker_row_all[kk], tracker_col_all[kk], row_col_sample)
        row_sample_list.append(row_sample)
        col_sample_list.append(col_sample)
    return row_sample_list, col_sample_list


def compute_distance(x1: Union[int, float], x2: Union[int, float], y1: Union[int, float], y2: Union[int, float]) -> Union[int, float]:
    """Given two 2D points. Will return the distance between them."""
    dist = ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0) ** 0.5
    return dist


def compute_unit_vector(x1: Union[int, float], x2: Union[int, float], y1: Union[int, float], y2: Union[int, float]) -> np.ndarray:
    """Given two 2D points. Will return the unit vector between them"""
    dist = compute_distance(x1, x2, y1, y2)
    vec = np.asarray([(x2 - x1) / dist, (y2 - y1) / dist])
    return vec


def insert_borders(mask: np.ndarray, border: int = 10) -> np.ndarray:
    """Given a mask. Will make the borders around it 0."""
    mask[0:border, :] = 0
    mask[-border:, :] = 0
    mask[:, 0:border] = 0
    mask[:, -border:] = 0
    return mask


def box_to_unit_vec(box: np.ndarray) -> np.ndarray:
    """Given the rectangular box. Will compute the unit vector of the longest side."""
    side_1 = compute_distance(box[0, 0], box[1, 0], box[0, 1], box[1, 1])
    side_2 = compute_distance(box[1, 0], box[2, 0], box[1, 1], box[2, 1])
    if side_1 > side_2:
        # side_1 is the long axis
        vec = compute_unit_vector(box[0, 0], box[1, 0], box[0, 1], box[1, 1])
    else:
        # side_2 is the long axis
        vec = compute_unit_vector(box[1, 0], box[2, 0], box[1, 1], box[2, 1])
    return vec


def box_to_center_points(box: np.ndarray) -> float:
    """Given the rectangular box. Will compute the center as the midpoint of a diagonal."""
    center_row = np.mean([box[0, 0], box[2, 0]])
    center_col = np.mean([box[0, 1], box[2, 1]])
    return center_row, center_col


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    """Given a mask. Will return the minimum area bounding rectangle."""
    # insert borders to the mask
    border = 10
    mask_mod = insert_borders(mask, border)
    # find contour
    mask_mod_one = (mask_mod > 0).astype(np.float64)
    mask_thresh_blur = ndimage.gaussian_filter(mask_mod_one, 1)
    cnts = measure.find_contours(mask_thresh_blur, 0.75)[0].astype(np.int32)
    # find minimum area bounding rectangle
    rect = cv2.minAreaRect(cnts)
    box = np.int0(cv2.boxPoints(rect))
    return box


def corners_to_mask(img: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Given a mask (for dimensions) and a unrotated corners. Will return a mask of the inside of the corners."""
    new_mask = np.zeros(img.shape)
    new_mask[r0:r1, c0:c1] = 1
    return new_mask


def axis_from_mask(mask: np.ndarray) -> np.ndarray:
    """Given a folder path. Will import the mask and determine its long axis."""
    box = mask_to_box(mask)
    vec = box_to_unit_vec(box)
    center_row, center_col = box_to_center_points(box)
    return center_row, center_col, vec


def rot_vec_to_rot_mat_and_angle(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """Given a rotation vector. Will return a rotation matrix and rotation angle."""
    ang = np.arctan2(vec[0], vec[1])
    rot_mat = np.asarray([[np.cos(ang), -1.0 * np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return (rot_mat, ang)


def get_tissue_width(mask: np.ndarray) -> float:
    """Given a mask of the tissue. Will compute the width of the tissue at the center."""
    center_row, center_col,_, ang,_ = get_rotation_info(center_row_input=None, center_col_input=None, vec_input=None, mask=mask)
    rot_mask = rot_image(mask, center_row, center_col,ang)
    mask_box = mask_to_box(rot_mask)
    center_row, center_col = box_to_center_points(mask_box)
    center_width = np.nonzero(rot_mask[:,int(center_col)]>0)
    min_row = np.min(center_width)
    max_row = np.max(center_width)
    tissue_width = max_row - min_row
    return tissue_width


def save_tissue_width_info(folder_path: Path,tissue_width: float) -> Path:
    """Given tissue width. Will save the results into a text file."""
    res_folder_path = create_folder(folder_path,"results")
    file_path = res_folder_path.joinpath("tissue_info.txt").resolve()
    tissue_info = np.asarray([tissue_width])
    np.savetxt(str(file_path), tissue_info)
    return file_path


def check_square_image(img: np.ndarray) -> bool:
    """Given an image. Will check if the image width and height dimensions are equal."""
    img_r, img_c = img.shape
    square = np.isclose(img_r,img_c,atol=5)
    return square


def rot_image(
    img: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int],
    ang: float
) -> np.ndarray:
    """Given an image and rotation information. Will return rotated image."""
    new_img = rotate(img, ang / (np.pi) * 180, center=(center_col, center_row))
    return new_img


def rotate_points(
    row_pts: np.ndarray,
    col_pts: np.ndarray,
    rot_mat: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given array vectors of points, rotation matrix, and point to rotate about.
    Will perform rotation and return rotated points"""
    row_pts_centered = row_pts - center_row
    col_pts_centered = col_pts - center_col
    pts = np.hstack((row_pts_centered.reshape((-1, 1)), col_pts_centered.reshape((-1, 1)))).T
    pts_rotated = rot_mat @ pts
    new_row_pts = pts_rotated[0, :] + center_row
    new_col_pts = pts_rotated[1, :] + center_col
    return new_row_pts, new_col_pts


def rotate_points_array(
    row_pts_array: np.ndarray,
    col_pts_array: np.ndarray,
    rot_mat: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given an array of row points and column points. Will rotate the whole array."""
    rot_row_pts_array = np.zeros(row_pts_array.shape)
    rot_col_pts_array = np.zeros(col_pts_array.shape)
    num_steps = row_pts_array.shape[1]
    for kk in range(0, num_steps):
        row_pts = row_pts_array[:, kk]
        col_pts = col_pts_array[:, kk]
        rot_row_pts, rot_col_pts = rotate_points(row_pts, col_pts, rot_mat, center_row, center_col)
        rot_row_pts_array[:, kk] = rot_row_pts
        rot_col_pts_array[:, kk] = rot_col_pts
    return rot_row_pts_array, rot_col_pts_array


def rotate_pts_all(
    row_pts_array_list: List,
    col_pts_array_list: List,
    rot_mat: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given a list of row and column point arrays. Will rotate all of them."""
    rot_row_pts_array_list = []
    rot_col_pts_array_list = []
    num_arrays = len(row_pts_array_list)
    for kk in range(0, num_arrays):
        row_pts_array = row_pts_array_list[kk]
        col_pts_array = col_pts_array_list[kk]
        rot_row_pts_array, rot_col_pts_array = rotate_points_array(row_pts_array, col_pts_array, rot_mat, center_row, center_col)
        rot_row_pts_array_list.append(rot_row_pts_array)
        rot_col_pts_array_list.append(rot_col_pts_array)
    return rot_row_pts_array_list, rot_col_pts_array_list


def translate_points(pts_row: np.ndarray, pts_col: np.ndarray, trans_r: float, trans_c: float) -> float:
    """Given an array of row points and column points. Will translate the whole array."""
    trans_pt_row = pts_row + trans_r
    trans_pt_col = pts_col + trans_c
    return trans_pt_row,trans_pt_col
    

def translate_pts_all(row_pts_array_list: List, col_pts_array_list: List, trans_r: float, trans_c: float) -> np.ndarray:
    """Given a list of row and column point arrays. Will translate all of them."""
    trans_row_pts_array_list = []
    trans_col_pts_array_list = []
    for kk in range(0,len(row_pts_array_list)):
        trans_row_pts_array, trans_col_pts_array = translate_points(row_pts_array_list[kk],col_pts_array_list[kk],trans_r,trans_c)
        trans_row_pts_array_list.append(trans_row_pts_array)
        trans_col_pts_array_list.append(trans_col_pts_array)        
    return trans_row_pts_array_list, trans_col_pts_array_list


def rotate_imgs_all(
    tiff_list: List,
    ang: float,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given tiff_list and rotation info. Will return all images rotated."""
    rot_tiff_list = []
    for kk in range(0, len(tiff_list)):
        rot_img = rot_image(tiff_list[kk], center_row, center_col, ang)
        rot_tiff_list.append(rot_img)
    return rot_tiff_list


def pad_img_to_square(image:np.ndarray) -> Tuple[np.ndarray, Union[float, int]]:
    """Given a non-square image. Will pad the image to have square size"""
    img_r, img_c = image.shape
    max_size = np.max([img_r,img_c])
    delta_r = max_size - img_r
    delta_c = max_size - img_c
    top, bottom = delta_r//2, delta_r - delta_r//2
    left, right = delta_c//2,  delta_c - delta_c//2
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
    translate_center_row = top
    translate_center_col = left
    return padded_image, translate_center_row, translate_center_col


def pad_all_imgs_to_square(tiff_list: List) -> Tuple[np.ndarray, Union[float, int]]:
    """Given a list of non-square images. Will pad images to have square size"""
    _, translate_center_row, translate_center_col = pad_img_to_square(tiff_list[0])
    padded_tiff_list = []
    for kk in range(0, len(tiff_list)):
        padded_img,_,_ = pad_img_to_square(tiff_list[kk])
        padded_tiff_list.append(padded_img)
    return padded_tiff_list, translate_center_row, translate_center_col


def rotate_test_img(
    folder_path: Path,
    tiff_list: List,
    ang: float,
    center_row: Union[float, int],
    center_col: Union[float, int],
    rot_mat: np.ndarray
):
    """Given tiff_list and rotation info. Will return the first image rotated for directionality reference."""
    img = tiff_list[0]
    img_r, img_c = img.shape
    img_center = np.array([img_r/2, img_c/2])
    
    # row and column positions of horizontal and vertical vectors 
    cr = np.array([img_center[1],img_center[1]])
    rr = np.array([img_center[0],img_center[0]-img_r/5])
    rc = np.array([img_center[0],img_center[0]])
    cc = np.array([img_center[1],img_center[1]+img_r/5])
      
    if abs(ang) < 0.96*np.pi:
        #check if image is square shape and perform appropriate rotation
        square = check_square_image(tiff_list[0])
        # rotate direction vectors
        rot_rr, rot_cr = rotate_points(rr, cr, rot_mat, center_row, center_col)
        rot_rc, rot_cc = rotate_points(rc, cc, rot_mat, center_row, center_col)
        if square == False:
            # pad image
            padded_img, translate_r, translate_c = pad_img_to_square(img)
            # translate center of rotation and direction vectors
            trans_center_row, trans_center_col = translate_points(center_row, center_col,translate_r,translate_c)
            rot_rr, rot_cr = translate_points(rot_rr,rot_cr,translate_r,translate_c)
            rot_rc, rot_cc = translate_points(rot_rc,rot_cc,translate_r,translate_c)
            # rotate image
            rot_img = rot_image(padded_img, trans_center_row, trans_center_col,ang)

        else:
            rot_img = rot_image(img, center_row, center_col, ang) 
    else:
        rot_img = img
        rot_rr, rot_cr = rr, cr
        rot_rc, rot_cc = rc, cc 

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.tight_layout(pad=0.08, h_pad=None, w_pad=None, rect=None)
    
    ax1.imshow(img, cmap='gray')
    ax1.plot(cr,rr,color ='firebrick',linewidth=2, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], zorder=10, label='vertical')
    ax1.arrow(cr[1], rr[1],(cr[1]-cr[0])/20 ,(rr[1]-rr[0])/20 , shape='full', lw=0, width=0, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], color='firebrick',length_includes_head=False, head_width=20,zorder=20)
    
    ax1.plot(cc,rc,color ='dodgerblue',linewidth=2, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], zorder=10, label='horizontal')
    ax1.arrow(cc[1], rc[1],(cc[1]-cc[0])/20 ,(rc[1]-rc[0])/20 , shape='full', lw=0, width=0, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], color='dodgerblue',length_includes_head=False, head_width=20,zorder=20)
    
    ax1.legend(loc='lower right', fontsize='small')
    ax1.set_title('Original Orientation')
    ax1.axis("off")
    
    ax2.imshow(rot_img,cmap='gray')
    ax2.plot(rot_cr,rot_rr,color ='firebrick',linewidth=2, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], zorder=10,label='vertical')
    ax2.arrow(rot_cr[1], rot_rr[1],(rot_cr[1]-rot_cr[0])/20 ,(rot_rr[1]-rot_rr[0])/20 , shape='full', lw=0, width=0, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], color='firebrick',length_includes_head=False, head_width=20,zorder=20)
    
    ax2.plot(rot_cc,rot_rc,color ='dodgerblue',linewidth=2, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], zorder=10, label='horizontal')
    ax2.arrow(rot_cc[1], rot_rc[1],(rot_cc[1]-rot_cc[0])/20 ,(rot_rc[1]-rot_rc[0])/20 , shape='full', lw=0, width=0, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], color='dodgerblue',length_includes_head=False, head_width=20,zorder=20)

    ax2.legend(loc='lower right', fontsize='small')
    ax2.set_title('Rotated Orientation')
    ax2.axis("off")
    
    filename = 'rotation direction.pdf'
    img_path = folder_path.joinpath(filename).resolve()
    fig.savefig(str(img_path), bbox_inches='tight', format='pdf')
    plt.close()
    return img_path


def get_rotation_info(
    *,
    center_row_input: Union[float, int] = None,
    center_col_input: Union[float, int] = None,
    vec_input: np.ndarray = None,
    mask: np.ndarray = None
) -> Tuple[Union[float, int], Union[float, int], np.ndarray, Union[float, int]]:
    """Given either prescribed rotation or mask.
    Will compute rotation information (rotation matrix and angle).
    Prescribed rotation will override rotation computed by the mask."""
    if mask is not None:
        center_row, center_col, vec = axis_from_mask(mask)
    if center_row_input is not None:
        center_row = center_row_input
    if center_col_input is not None:
        center_col = center_col_input
    if vec_input is not None:
        vec = vec_input
    (rot_mat, ang) = rot_vec_to_rot_mat_and_angle(vec)
    return (center_row, center_col, rot_mat, ang, vec)


def run_rotation(
    folder_path: Path,
    input_mask: bool = True,
    *,
    center_row_input: Union[float, int] = None,
    center_col_input: Union[float, int] = None,
    vec_input: np.ndarray = None
) -> List:
    """Given rotation information. Will rotate the points according to the provided information."""
    if input_mask:
        mask_file_path = folder_path.joinpath("masks").resolve().joinpath("tissue_mask.txt").resolve()
        mask = read_txt_as_mask(mask_file_path)
        (center_row, center_col, rot_mat, ang, vec) = get_rotation_info(center_row_input=center_row_input, center_col_input=center_col_input, vec_input=vec_input, mask=mask)
    else:
        (center_row, center_col, rot_mat, ang, vec) = get_rotation_info(center_row_input=center_row_input, center_col_input=center_col_input, vec_input=vec_input)
    # read tracking results
    tracker_row_all, tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path)  
    if abs(ang) < 0.96*np.pi:
        # perform rotation
        rot_tracker_row_all, rot_tracker_col_all = rotate_pts_all(tracker_row_all, tracker_col_all, rot_mat, center_row, center_col)
    else:
        # do not rotate
        rot_tracker_row_all, rot_tracker_col_all = tracker_row_all, tracker_col_all
    # save rotation info
    rot_info = np.asarray([[center_row, center_col], [vec[0], vec[1]]])
    # save rotation
    saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=rot_tracker_col_all, tracker_row_all=rot_tracker_row_all, is_rotated=True, rot_info=rot_info)
    return saved_paths


def run_rotation_visualization(folder_path: Path, automatic_color_constraint: bool = True, col_min_abs: Union[int, float] = 0, col_max_abs: Union[int, float] = 8, col_min_row: Union[int, float] = -3, col_max_row: Union[int, float] = 4.5, col_min_col: Union[int, float] = -3, col_max_col: Union[int, float] = 4.5, col_map: object = plt.cm.viridis) -> List:
    """Given a folder path where rotated tracking has already been run. Will save visualizations."""
    # read image files
    movie_folder_path = folder_path.joinpath("movie").resolve()
    vis_folder_path = create_folder(folder_path, "visualizations")
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    # read rotated tracking results
    rot_tracker_row_all, rot_tracker_col_all, info, rot_info = load_tracking_results(folder_path=folder_path, is_rotated=True)
    # rotate tiffs
    center_row = rot_info[0, 0]
    center_col = rot_info[0, 1]
    vec = np.asarray([rot_info[1, 0], rot_info[1, 1]])
    (rot_mat, ang) = rot_vec_to_rot_mat_and_angle(vec)
    if abs(ang) < 0.96*np.pi:
        # check if images in list are square shape and perform appropriate rotation
        square = check_square_image(tiff_list[0])
        if square == False:
            # pad image list
            padded_tiff_list, translate_r, translate_c = pad_all_imgs_to_square(tiff_list)
            # translate center of rotation
            trans_center_row, trans_center_col = translate_points(center_row,center_col,translate_r,translate_c)
            # rotate all images in list
            rot_tiff_list = rotate_imgs_all(padded_tiff_list, ang, trans_center_row, trans_center_col)
            # translate rotated tracker points to account for padding 
            tracker_row_all_pad, tracker_col_all_pad = translate_pts_all(rot_tracker_row_all, rot_tracker_col_all, translate_r, translate_c)
        else:
            rot_tiff_list = rotate_imgs_all(tiff_list, ang, center_row, center_col) 
            tracker_row_all_pad, tracker_col_all_pad = rot_tracker_row_all, rot_tracker_col_all
    else:
        rot_tiff_list = tiff_list
        tracker_row_all_pad, tracker_col_all_pad = rot_tracker_row_all, rot_tracker_col_all   
    if automatic_color_constraint:
        # find limits of colormap
        col_min_abs, col_max_abs, col_min_row, col_max_row, col_min_col, col_max_col = compute_min_max_disp(rot_tracker_row_all,rot_tracker_col_all,info)
    # create rotated test image
    rotate_test_img(vis_folder_path, tiff_list, ang, center_row, center_col, rot_mat)
    # create pngs
    abs_png_path_list = create_pngs(folder_path, rot_tiff_list, tracker_row_all_pad, tracker_col_all_pad, info, "abs", col_min_abs, col_max_abs, col_map, is_rotated=True, save_eps = False)
    row_png_path_list = create_pngs(folder_path, rot_tiff_list, tracker_row_all_pad, tracker_col_all_pad, info, "row", col_min_row, col_max_row, col_map, is_rotated=True, save_eps = False)
    col_png_path_list = create_pngs(folder_path, rot_tiff_list, tracker_row_all_pad, tracker_col_all_pad, info, "col", col_min_col, col_max_col, col_map, is_rotated=True, save_eps = False)
    # create gif
    abs_gif_path = create_gif(folder_path, abs_png_path_list, "abs", is_rotated=True)
    row_gif_path = create_gif(folder_path, row_png_path_list, "row", is_rotated=True)
    col_gif_path = create_gif(folder_path, col_png_path_list, "col", is_rotated=True)
    return abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path


def scale_array_in_list(tracker_list: List, new_origin: Union[int, float], scale_mult: Union[int, float]) -> List:
     """Given a list of arrays of coordinates, new origin (in pixel coordinates), and new scale. Will subtract the origin and then multiply by the scale."""
     updated_tracker_list = []
     num_beats = len(tracker_list)
     for kk in range(0, num_beats):
         val_array = tracker_list[kk]
         new_val_array = (val_array - new_origin) * scale_mult
         updated_tracker_list.append(new_val_array)
     return updated_tracker_list


def run_scale_and_center_coordinates(
     folder_path: Path,
     pixel_origin_row: Union[int, float],
     pixel_origin_col: Union[int, float],
     microns_per_pixel_row: Union[int, float],
     microns_per_pixel_col: Union[int, float],
     use_rotated: bool = False,
     fname: str = None,
     new_fname: str = None
 ) -> List:
     """Given information to transform the coordinate system (translation only). """
     tracker_row_all, tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path, is_rotated=use_rotated, fname=fname)
     updated_tracker_row_all = scale_array_in_list(tracker_row_all, pixel_origin_row, microns_per_pixel_row)
     updated_tracker_col_all = scale_array_in_list(tracker_col_all, pixel_origin_col, microns_per_pixel_col)
     saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=updated_tracker_col_all, tracker_row_all=updated_tracker_row_all, is_translated=True, is_rotated=use_rotated, fname=new_fname)
     return saved_paths


def run_interpolate(
    folder_path: Path,
    row_col_sample: np.ndarray,
    interpolation_fname: str = "interpolation",
    is_rotated: bool = False,
    is_translated: bool = False
) -> List:
    """Given a folder path, information, and sample points. Will compute and save interpolation at the sample points."""
    # load tracking results
    tracker_row_all, tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path, is_rotated=is_rotated, is_translated=is_translated)
    # perform interpolation of tracking results
    row_sample_list, col_sample_list = interpolate_pos_from_tracking_lists(tracker_row_all, tracker_col_all, row_col_sample)
    # save interpolated results
    saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=col_sample_list, tracker_row_all=row_sample_list, fname=interpolation_fname)
    return saved_paths


def visualize_interpolate(
    folder_path: Path,
    *,
    is_rotated: bool = False,
    is_translated: bool = False,
    interpolation_fname: str = "interpolation",
    automatic_color_constraint: bool = True,
    col_min_abs: Union[int, float] = 0, 
    col_max_abs: Union[int, float] = 8, 
    col_min_row: Union[int, float] = -3, 
    col_max_row: Union[int, float] = 4.5, 
    col_min_col: Union[int, float] = -3, 
    col_max_col: Union[int, float] = 4.5,
    col_map: object = plt.cm.viridis
) -> List:
    """Given folder path and plotting information. Will run and save visualization."""
    # read image files and tracking results
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    tracker_row_all, tracker_col_all, info, rot_info = load_tracking_results(folder_path=folder_path, is_rotated=is_rotated, is_translated=is_translated)
    # read interpolated results
    interp_tracker_row_all, interp_tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path, is_rotated=is_rotated, is_translated=is_translated, fname=interpolation_fname)
    if is_rotated:
        center_row = rot_info[0, 0]
        center_col = rot_info[0, 1]
        vec = np.asarray([rot_info[1, 0], rot_info[1, 1]])
        (_, ang) = rot_vec_to_rot_mat_and_angle(vec)
        if abs(ang) < 0.96*np.pi:
            # check if images in list are square shape and perform appropriate rotation
            square = check_square_image(tiff_list[0])
            if square == False:
                # pad image list
                padded_tiff_list, translate_r, translate_c = pad_all_imgs_to_square(tiff_list)
                # translate center of rotation
                trans_center_row, trans_center_col = translate_points(center_row,center_col,translate_r,translate_c)
                # rotate all images in list
                tiff_list = rotate_imgs_all(padded_tiff_list, ang, trans_center_row, trans_center_col)
                # translate tracker points and interpolated points to account for padding 
                tracker_row_all, tracker_col_all = translate_pts_all(tracker_row_all, tracker_col_all, translate_r, translate_c)
                interp_tracker_row_all, interp_tracker_col_all = translate_pts_all(interp_tracker_row_all, interp_tracker_col_all, translate_r, translate_c)
            else:
                tiff_list = rotate_imgs_all(tiff_list, ang, center_row, center_col) 
    if automatic_color_constraint:
        # find limits of colormap
        col_min_abs, col_max_abs, col_min_row, col_max_row, col_min_col, col_max_col = compute_min_max_disp(tracker_row_all,tracker_col_all,info)
    # create pngs
    abs_png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "abs", col_min_abs, col_max_abs, col_map, is_rotated=is_rotated, include_interp=True, interp_tracker_row_all=interp_tracker_row_all, interp_tracker_col_all=interp_tracker_col_all, save_eps = False)
    row_png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "row", col_min_row, col_max_row, col_map, is_rotated=is_rotated, include_interp=True, interp_tracker_row_all=interp_tracker_row_all, interp_tracker_col_all=interp_tracker_col_all, save_eps = False)
    col_png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, "col", col_min_col, col_max_col, col_map, is_rotated=is_rotated, include_interp=True, interp_tracker_row_all=interp_tracker_row_all, interp_tracker_col_all=interp_tracker_col_all, save_eps = False)
    # create gif
    abs_gif_path = create_gif(folder_path, abs_png_path_list, "abs", is_rotated=is_rotated, include_interp=True)
    row_gif_path = create_gif(folder_path, row_png_path_list, "row", is_rotated=is_rotated, include_interp=True)
    col_gif_path = create_gif(folder_path, col_png_path_list, "col" ,is_rotated=is_rotated, include_interp=True)   
    return abs_png_path_list, row_png_path_list, col_png_path_list, abs_gif_path, row_gif_path, col_gif_path


#================================================ Pillar Tracking ================================================#
def compute_pillar_secnd_moment_rectangular(pillar_width: float, pillar_thickness: float)-> float: 
    """Given pillar width and thickness in micrometers (um).
    Will compute the pillar (taken as a rectangular beam) second moment of area in (um)^4."""
    secnd_moment_area = (pillar_width*pillar_thickness**3)/12
    return secnd_moment_area

def compute_pillar_secnd_moment_circular(pillar_diameter: float)-> float: 
    """Given pillar width and thickness in micrometers (um).
    Will compute the pillar (taken as a circular beam) second moment of area in (um)^4."""
    secnd_moment_area = (np.pi*pillar_diameter**4)/64
    return secnd_moment_area

def compute_pillar_stiffnes(pillar_profile: str, pillar_modulus: float, pillar_width: float, 
                            pillar_thickness: float, pillar_diameter: float, pillar_length: float, 
                            force_location: float) -> float:
    """Given pillar material Elastic modulus (in MPa), width, thickness, length and force application location 
    in micrometers (um). Will compute the pillar stiffness in (uN/um)."""
    if pillar_profile == 'rectangular':
        I = compute_pillar_secnd_moment_rectangular(pillar_width, pillar_thickness)
    elif pillar_profile == 'circular':
        I = compute_pillar_secnd_moment_circular(pillar_diameter)
    else:
        print("Pillar_profile should be either 'rectangular' or 'circular'")
        I = 0
    pillar_stiffness = (6*pillar_modulus*I)/((force_location**2)*(3*pillar_length-force_location))
    return pillar_stiffness


def compute_pillar_force(pillar_stiffness: float, pillar_avg_deflection: float, length_scale: float) -> np.ndarray:
    """Given pillar stiffness in (uN/um), pillar average deflection in pixels and a length scale 
    conversion from pixels to micrometers (um). Will compute pillar force in microNewtons (uN)."""
    pillar_force = pillar_stiffness*pillar_avg_deflection*length_scale
    return pillar_force


def compute_pillar_position_timeseries(tracker_0: np.ndarray, tracker_1: np.ndarray) -> np.ndarray:
    """Given tracker arrays. Will return single timeseries of mean absolute displacement, 
    mean row displacement and mean column displacement."""
    mean_tracker_0 = np.mean(tracker_0,axis=0)
    mean_tracker_1 = np.mean(tracker_1,axis=0)
    
    mean_tracker_0_0 = np.ones(np.shape(mean_tracker_0))*mean_tracker_0[0]
    mean_disp_0_all = mean_tracker_0 - mean_tracker_0_0
    
    mean_tracker_1_0 = np.ones(np.shape(mean_tracker_1))*mean_tracker_1[0]
    mean_disp_1_all = mean_tracker_1 - mean_tracker_1_0
    
    disp_abs_mean = (mean_disp_0_all ** 2.0 + mean_disp_1_all ** 2.0) ** 0.5
    return disp_abs_mean, mean_disp_0_all, mean_disp_1_all


def pillar_force_all_steps(pillar_mean_abs_disp: np.ndarray, pillar_mean_disp_row: np.ndarray, 
                           pillar_mean_disp_col: np.ndarray, pillar_stiffnes: float = None, pillar_profile: str = 'rectangular',
                           pillar_modulus: float = 1.61, pillar_width: float = 163, pillar_thickness: float = 33.2, 
                           pillar_diameter: float = 400, pillar_length: float = 199.3, force_location: float = 163,
                           length_scale: float = 1) -> List:
    """Given pillar material Elastic modulus (in MPa), width, thickness, length, force application location 
    in micrometers (um), pillar tracking results in pixels, and a length scale conversion from pixels to 
    micrometers (um). Will compute pillar force in microNewtons (uN) for all steps."""
    
    if pillar_stiffnes is not None:
        pillar_k = pillar_stiffnes
    else:
        pillar_k = compute_pillar_stiffnes(pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)

    pillar_F_row = compute_pillar_force(pillar_k,pillar_mean_disp_row,length_scale)
    pillar_F_col = compute_pillar_force(pillar_k,pillar_mean_disp_col,length_scale)
    pillar_F_abs = compute_pillar_force(pillar_k,pillar_mean_abs_disp,length_scale)

    return pillar_F_abs, pillar_F_row, pillar_F_col
    

def save_pillar_position(*, folder_path: Path, tracker_row_all: List, tracker_col_all: List, 
                         info: np.ndarray = None, split_track: bool = False, fname: str = None) -> List:
    """Given pillar tracking results. Will save as text files."""
    new_path = create_folder(folder_path, "pillar_results")

    saved_paths = []
    if split_track:
        num_beats = info.shape[0]
        for kk in range(0, num_beats):
            tracker_row = tracker_row_all[kk]
            tracker_col = tracker_col_all[kk]
            if fname is not None:
                file_path = new_path.joinpath(fname + "beat%i_row.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_row)
                file_path = new_path.joinpath(fname + "beat%i_col.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_col)
            else:
                file_path = new_path.joinpath("beat%i_row.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_row)
                file_path = new_path.joinpath("beat%i_col.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_col)
    else:
        if fname is not None:
            file_path = new_path.joinpath(fname + "row.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all)
            file_path = new_path.joinpath(fname + "col.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all)
        else:
            file_path = new_path.joinpath("row.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all)
            file_path = new_path.joinpath("col.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all)
        if info is not None:
            file_path = new_path.joinpath("info.txt").resolve()
            np.savetxt(str(file_path), info)
            saved_paths.append(file_path)
    return saved_paths


def save_pillar_force(*, folder_path: Path, pillar_force_abs: np.ndarray, pillar_force_row: np.ndarray, 
                         pillar_force_col: np.ndarray, fname: str = None) -> List:
    """Given pillar force results. Will save as text files."""
    new_path = create_folder(folder_path, "pillar_results")
    saved_paths = [] 
    if fname is not None:
        file_path = new_path.joinpath(fname + "pillar_force_abs.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_abs)   
        file_path = new_path.joinpath(fname + "pillar_force_row.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_row)
        file_path = new_path.joinpath(fname + "pillar_force_col.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_col)
    else:
        file_path = new_path.joinpath("pillar_force_abs.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_abs)
        file_path = new_path.joinpath("pillar_force_row.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_row)
        file_path = new_path.joinpath("pillar_force_col.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_col)
    return saved_paths


def run_pillar_tracking(folder_path: Path, pillar_stiffnes: float = None, pillar_profile: str = 'rectangular', 
                        pillar_modulus: float = 1.61, pillar_width: float = 163, pillar_thickness: float = 33.2, 
                        pillar_diameter: float = 400, pillar_length: float = 199.3, force_location: float = 163, 
                        length_scale: float = 1, split_track: bool = False) -> List:
    """Given a folder path, pillar material Elastic modulus (in MPa), width, thickness, length, force application 
    location in micrometers (um), and a length scale conversion from pixels to micrometers (um).Will perform tracking, 
    compute pillar force in microNewtons (uN) and save results as text files."""
    # read images and mask file
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    img_list_uint8 = uint16_to_uint8_all(tiff_list)
    mask_folder_path = folder_path.joinpath("masks").resolve()
    mask_file_list = glob.glob(str(mask_folder_path) + "/*pillar_mask*.txt")
    
    for ml in range(len(mask_file_list)): 
        # load pillar masks
        mask_file_path = mask_folder_path.joinpath("pillar_mask_%i.txt"%(ml+1)).resolve()
        mask = read_txt_as_mask(mask_file_path)
        # perform tracking
        tracker_0, tracker_1 = track_all_steps_with_adjust_param_dicts(img_list_uint8, mask)
        # perform timeseries analysis
        mean_abs_disp, mean_disp_all_0, mean_disp_all_1 = compute_pillar_position_timeseries(tracker_0,tracker_1)
        info = compute_valleys(mean_abs_disp)
        
        if split_track:
            mean_abs_disp_all = []
            mean_disp_all_0_all = []
            mean_disp_all_1_all = []
            tracker_0_all, tracker_1_all = split_tracking(tracker_0, tracker_1, info)
            num_beats = len(tracker_0_all)
            for nb in range(num_beats):
                tracker_0_beat = tracker_0_all[nb]
                tracker_1_beat = tracker_1_all[nb]
                mean_abs_disp_beat, mean_disp_all_0_beat, mean_disp_all_1_beat = compute_pillar_position_timeseries(tracker_0_beat,tracker_1_beat)
                mean_abs_disp_all.append(mean_abs_disp_beat)
                mean_disp_all_0_all.append(mean_disp_all_0_beat)
                mean_disp_all_1_all.append(mean_disp_all_1_beat)
                # save pillar tracking results
                saved_paths_pos = save_pillar_position(folder_path=folder_path, tracker_col_all=tracker_0_all, tracker_row_all=tracker_1_all, info=info, split_track = True, fname='pillar%i_'%(ml+1))
            
            mean_abs_disp = [disp for disp_lst in mean_abs_disp_all for disp in disp_lst]
            mean_abs_disp = np.asarray(mean_abs_disp)
            mean_disp_all_0 = [disp_0 for disp_0_lst in mean_disp_all_0_all for disp_0 in disp_0_lst]
            mean_disp_all_0 = np.asarray(mean_disp_all_0)
            mean_disp_all_1 = [disp_1 for disp_1_lst in mean_disp_all_1_all for disp_1 in disp_1_lst]
            mean_disp_all_1 = np.asarray(mean_disp_all_1)
        else:  
            # save pillar tracking results
            saved_paths_pos = save_pillar_position(folder_path=folder_path, tracker_col_all=tracker_0, tracker_row_all=tracker_1, info=info, split_track = False, fname='pillar%i_'%(ml+1))
        # compute pillar force 
        pillar_force_all, pillar_row_force_all, pillar_col_force_all = pillar_force_all_steps(mean_abs_disp, mean_disp_all_0, mean_disp_all_1, pillar_stiffnes = pillar_stiffnes, pillar_profile = pillar_profile, pillar_modulus = pillar_modulus, pillar_width = pillar_width, pillar_thickness = pillar_thickness, pillar_diameter = pillar_diameter, pillar_length = pillar_length, force_location = force_location, length_scale = length_scale)
        # save pillar force results
        saved_paths_force = save_pillar_force(folder_path=folder_path, pillar_force_abs=pillar_force_all, pillar_force_row=pillar_row_force_all, pillar_force_col=pillar_col_force_all, fname='pillar%i_'%(ml+1))
    return saved_paths_pos, saved_paths_force


def load_pillar_tracking_results(folder_path: Path, split_track: bool = False, fname: str = "") -> np.ndarray:
    """Given folder path. Will load pillar force results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present -- therefore pillar force results must not be present either")
    file_list = glob.glob(str(res_folder_path) + "/*pillar_force*")
    if len(file_list) == 0:
        raise FileNotFoundError("pillar force results are not present")
        
    if split_track:  
        num_files = len(glob.glob(str(res_folder_path) + "/" + fname + "beat*.txt"))
        num_beats = int((num_files) / 2)
        pillar_row_all = []
        pillar_col_all = []
        for kk in range(0, num_beats):
            pillar_row = np.loadtxt(str(res_folder_path) + "/" + fname + "beat%i_row.txt" % (kk))
            pillar_col = np.loadtxt(str(res_folder_path) + "/" + fname + "beat%i_col.txt" % (kk))
            pillar_row_all.append(pillar_row)
            pillar_col_all.append(pillar_col)
    else:
        pillar_row_all = np.loadtxt(str(res_folder_path) + "/" + fname + "row.txt")
        pillar_col_all = np.loadtxt(str(res_folder_path) + "/" + fname + "col.txt")   
        
    pillar_abs_force_all = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_force_abs.txt")
    pillar_row_force_all = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_force_row.txt")
    pillar_col_force_all = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_force_col.txt")
    return pillar_row_all, pillar_col_all, pillar_abs_force_all, pillar_row_force_all, pillar_col_force_all


def visualize_pillar_tracking(folder_path: Path, split_track: bool = False) -> Path:
    """Given a folder path where tracking has already been run. Will save visualizations."""
    vis_folder_path = create_folder(folder_path, "pillar_visualizations")
    mask_folder_path = folder_path.joinpath("masks").resolve()
    mask_file_list = glob.glob(str(mask_folder_path) + "/*pillar_mask*.txt")
    
    color_lst = ['dodgerblue','firebrick','lightcoral','lightskyblue']
    
    plt.figure()
    for ml in range(len(mask_file_list)): 
        # load pillar force results
        pillar_tracker_row, pillar_tracker_col, all_pillar_force, _, _ = load_pillar_tracking_results(folder_path,split_track,fname='pillar%i_'%(ml+1))
        if split_track:
            num_beats = len(pillar_tracker_row)
            mean_abs_disp_all = []
            for nb in range(num_beats):
                tracker_0_beat = pillar_tracker_row[nb]
                tracker_1_beat = pillar_tracker_col[nb]
                mean_abs_disp_beat, mean_disp_all_0_beat, mean_disp_all_1_beat = compute_pillar_position_timeseries(tracker_0_beat,tracker_1_beat)
                mean_abs_disp_all.append(mean_abs_disp_beat)

            mean_abs_disp = [disp for disp_lst in mean_abs_disp_all for disp in disp_lst]
            mean_abs_disp = np.asarray(mean_abs_disp)
        else:
            mean_abs_disp, _, _ = compute_pillar_position_timeseries(pillar_tracker_row,pillar_tracker_col)
    
        plt.plot(mean_abs_disp, color = color_lst[ml], label='pillar %i'%(ml+1))
    plt.ylabel(r'pillar mean absolute displacement (pixels)')
    plt.xlabel('frame')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(str(vis_folder_path)+'/pillar_mean_absolute_displacement.pdf', format='pdf')
    plt.close()
        
    plt.figure()
    count = 1
    for ml in range(len(mask_file_list)): 
        pillar_tracker_row, pillar_tracker_col, _, _, _ = load_pillar_tracking_results(folder_path,split_track,fname='pillar%i_'%(ml+1))
        
        if split_track:
            num_beats = len(pillar_tracker_row)
            mean_disp_all_0_all = []
            mean_disp_all_1_all = []
            for nb in range(num_beats):
                tracker_0_beat = pillar_tracker_row[nb]
                tracker_1_beat = pillar_tracker_col[nb]
                _, mean_disp_all_0_beat, mean_disp_all_1_beat = compute_pillar_position_timeseries(tracker_0_beat,tracker_1_beat)
                mean_disp_all_0_all.append(mean_disp_all_0_beat)
                mean_disp_all_1_all.append(mean_disp_all_1_beat)
            
            mean_disp_all_row = [disp_0 for disp_0_lst in mean_disp_all_0_all for disp_0 in disp_0_lst]
            mean_disp_all_row = np.asarray(mean_disp_all_row)
            mean_disp_all_col = [disp_1 for disp_1_lst in mean_disp_all_1_all for disp_1 in disp_1_lst]
            mean_disp_all_col = np.asarray(mean_disp_all_col)
        
        else:
            _, mean_disp_all_row, mean_disp_all_col = compute_pillar_position_timeseries(pillar_tracker_row,pillar_tracker_col)
        
        plt.plot(mean_disp_all_row, color = color_lst[-count], label='pillar %i row (vertical)'%(ml+1))
        plt.plot(mean_disp_all_col, color = color_lst[ml], label='pillar %i column (horizontal)'%(ml+1))
        count +=1
        
    plt.ylabel(r'pillar mean displacement (pixels)')
    plt.xlabel('frame')
    # plt.legend(loc='upper right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(str(vis_folder_path)+'/pillar_directional_displacement.pdf', format='pdf')
    plt.close()

    plt.figure()
    for ml in range(len(mask_file_list)): 
        _, _, all_pillar_force, _, _ = load_pillar_tracking_results(folder_path,split_track,fname='pillar%i_'%(ml+1))
        plt.plot(all_pillar_force, color = color_lst[ml], label='pillar %i'%(ml+1))

    plt.ylabel(r'pillar absolute force ($\mu$N)')
    plt.xlabel('frame')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(str(vis_folder_path)+'/pillar_force_absolute.pdf', format='pdf')
    plt.close()
    
    return vis_folder_path
