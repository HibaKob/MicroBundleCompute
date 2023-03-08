import matplotlib.pyplot as plt
from microbundlecompute import image_analysis as ia
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage.filters import threshold_otsu, gabor, sobel
from skimage.measure import label, regionprops
from skimage import morphology
from typing import List, Union

def apply_median_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the median filter applied by scipy"""
    filtered_array = ndimage.median_filter(array, filter_size)
    return filtered_array


def apply_gaussian_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the gaussian filter applied by scipy"""
    filtered_array = ndimage.gaussian_filter(array, filter_size)
    return filtered_array


def apply_sobel_filter(array: np.ndarray) -> np.ndarray:
    """Given an image array. Will return the sobel filter."""
    res = sobel(array)
    return res


def compute_otsu_thresh(array: np.ndarray) -> Union[float, int]:
    """Given an image array. Will return the otsu threshold applied by skimage."""
    thresh = threshold_otsu(array)
    return thresh


def apply_otsu_thresh(array: np.ndarray) -> np.ndarray:
    """Given an image array. Will return a boolean numpy array with an otsu threshold applied."""
    thresh = compute_otsu_thresh(array)
    thresh_img = array > thresh
    return thresh_img


def gabor_filter(array: np.ndarray, theta_range: int = 17, ff_max: int = 11, ff_mult: float = 0.1) -> np.ndarray:
    gabor_all = np.zeros(array.shape)
    for ff in range(0, ff_max):
        frequency = 0.2 + ff * ff_mult
        for tt in range(0, theta_range):
            theta = tt * np.pi / (theta_range - 1)
            filt_real, _ = gabor(array, frequency=frequency, theta=theta)
            gabor_all += filt_real
    return gabor_all


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Given a mask. Will return an inverted mask."""
    invert_mask = mask == 0
    return invert_mask


def get_largest_regions(array: np.ndarray, num_regions: int = 1) -> List:
    """Given a thresholded image. Will return a list of the num_regions largest regions.
    If there are fewer than num_regions regions, will return all regions."""
    label_image = label(array)
    region_props = regionprops(label_image)
    area_list = []
    for region in region_props:
        area_list.append(region.area)
    ranked = np.argsort(area_list)[::-1]
    num_to_return = np.min([len(ranked), num_regions])
    regions_list = []
    for kk in range(0, num_to_return):
        idx = ranked[kk]
        regions_list.append(region_props[idx])
    return regions_list


def region_to_mask(region: object, array: np.ndarray) -> np.ndarray:
    """Given regions. Will return a mask of the coordinates of the region."""
    coords = region.coords
    mask = np.zeros(array.shape)
    for kk in range(0, coords.shape[0]):
        mask[coords[kk, 0], coords[kk, 1]] = 1
    return mask


def close_region(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    closed_array = morphology.binary_closing(array, footprint)
    return closed_array


def segment_mask_1(array: np.ndarray):
    thresh_img = apply_otsu_thresh(array)
    thresh_img_inverted = invert_mask(thresh_img)
    thresh_img_inverted_borders = ia.insert_borders(thresh_img_inverted)
    region = get_largest_regions(thresh_img_inverted_borders)[0]
    mask_region = region_to_mask(region, array)
    mask_region_closed = close_region(mask_region, 3)
    return mask_region_closed


def segment_mask_2(array: np.ndarray):
    sob = apply_sobel_filter(array)
    blurred = apply_gaussian_filter(sob, 5)
    thresh_img = apply_otsu_thresh(blurred)
    region = get_largest_regions(thresh_img)[0]
    mask_region = region_to_mask(region, array)
    # mask_region_closed = close_region(mask_region, 5)
    mask_region_closed = close_region(mask_region, 10)
    return mask_region_closed


def segment_mask_3(tiff_list: List, method:str='minimum'): 
    if method == 'minimum':
        array = np.min(tiff_list,axis=0)
    elif method == 'maximum':
        array = np.max(tiff_list,axis=0)
    median = apply_median_filter(array,2)
    thresh_img = apply_otsu_thresh(median)
    thresh_img_inverted = invert_mask(thresh_img)
    thresh_img_inverted_borders = ia.insert_borders(thresh_img_inverted)
    region = get_largest_regions(thresh_img_inverted_borders)[0]
    mask_region = region_to_mask(region, array)
    mask_region_closed = close_region(mask_region, 25)
    return mask_region_closed


def save_mask(folder_path: Path, mask: np.ndarray, fname: str = "tissue_mask"):
    """Given a folder path and tissue mask. Will save file."""
    new_path = ia.create_folder(folder_path, "masks")
    file_path = new_path.joinpath(fname + ".txt").resolve()
    np.savetxt(str(file_path), mask, fmt="%i")
    img_path = new_path.joinpath(fname + ".png").resolve()
    plt.imsave(img_path, mask)
    return file_path, img_path


def run_create_tissue_mask(folder_path: Path, seg_fcn_num: int = 1, fname: str = "tissue_mask", frame_num: int = 0, method: str = "minimum"):
    """Given a folder and selection of mask segmentation settings. Will segment and save the tissue mask."""
    # load the first image
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img = tiff_list[frame_num]
    # create the tissue mask
    if seg_fcn_num == 1:
        mask = segment_mask_1(img)
    elif seg_fcn_num == 2:
        mask = segment_mask_2(img)
    elif seg_fcn_num == 3:
        mask = segment_mask_3(tiff_list,method)
    # save the tissue mask
    file_path, img_path = save_mask(folder_path, mask, fname)
    return file_path, img_path
