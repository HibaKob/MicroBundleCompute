import glob
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import io
from skimage.filters import threshold_otsu, gabor
from skimage.measure import label, regionprops
from skimage import morphology
from typing import List, Union


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


def apply_median_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the median filter applied by scipy"""
    filtered_array = ndimage.median_filter(array, filter_size)
    return filtered_array


def apply_gaussian_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the gaussian filter applied by scipy"""
    filtered_array = ndimage.gaussian_filter(array, filter_size)
    return filtered_array


def compute_otsu_thresh(array: np.ndarray) -> Union[float, int]:
    """Given an image array. Will return the otsu threshold applied by skimage."""
    thresh = threshold_otsu(array)
    return thresh


def apply_otsu_thresh(array: np.ndarray) -> np.ndarray:
    """Given an image array. Will return a boolean numpy array with an otsu threshold applied."""
    thresh = compute_otsu_thresh(array)
    thresh_img = array > thresh
    return thresh_img


def get_region_props(array: np.ndarray) -> List:
    """Given a binary image. Will return the list of region props."""
    label_image = label(array)
    region_props = regionprops(label_image)
    return region_props


def get_domain_center(array: np.ndarray) -> Union[int, float]:
    """Given an array. Will return center (ix_0, ix_1)"""
    center_0 = array.shape[0] / 2.0
    center_1 = array.shape[1] / 2.0
    return center_0, center_1


def region_to_coords(regions_list: List) -> List:
    """Given regions list. Will return the coordinates of all regions in the list."""
    coords_list = []
    for region in regions_list:
        coords = region.coords
        coords_list.append(coords)
    return coords_list


def coords_to_mask(coords_list: List, array: np.ndarray) -> np.ndarray:
    """Given coordinates and template array. Will turn coordinates into a binary mask."""
    mask = np.zeros(array.shape)
    for coords in coords_list:
        for kk in range(0, coords.shape[0]):
            mask[coords[kk, 0], coords[kk, 1]] = 1
    return mask


def compute_distance(
    a0: Union[int, float],
    a1: Union[int, float],
    b0: Union[int, float],
    b1: Union[int, float]
) -> Union[int, float]:
    """Given two points. Will return distance between them."""
    dist = ((a0 - b0)**2.0 + (a1 - b1)**2.0)**0.5
    return dist


def get_closest_region(
    regions_list: List,
    loc_0: Union[int, float],
    loc_1: Union[int, float]
) -> object:
    """Given a list of region properties. Will return the object closest to location."""
    center_dist = []
    for region in regions_list:
        centroid = region.centroid
        region_0 = centroid[0]
        region_1 = centroid[1]
        dist = compute_distance(region_0, region_1, loc_0, loc_1)
        center_dist.append(dist)
    ix = np.argmin(center_dist)
    return regions_list[ix]


def get_largest_regions(region_props: List, num_regions: int = 3) -> List:
    """Given a list of region properties. Will return a list of the num_regions largest regions.
    If there are fewer than num_regions regions, will return all regions."""
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


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Given a mask. Will return an inverted mask."""
    invert_mask = mask == 0
    return invert_mask


def close_region(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    closed_array = morphology.binary_closing(array, footprint)
    return closed_array


def gabor_filter(array: np.ndarray, theta_range: int = 17, ff_max: int = 11, ff_mult: float = 0.1) -> np.ndarray:
    gabor_all = np.zeros(array.shape)
    for ff in range(0, ff_max):
        frequency = 0.2 + ff * ff_mult
        for tt in range(0, theta_range):
            theta = tt * np.pi / (theta_range - 1)
            filt_real, _ = gabor(array, frequency=frequency, theta=theta)
            gabor_all += filt_real
    return gabor_all


def crop_boundaries(array: np.ndarray, crop: int = 20) -> np.ndarray:
    array[0:crop, :] = 0
    array[-crop:, :] = 0
    array[:, 0:crop] = 0
    array[:, -crop:] = 0
    return array


def microbundle_mask_v1(array: np.ndarray) -> np.ndarray:
    """Given an image. Will segment the central microbundle and return it as a mask."""
    median_filter_size = 5
    array_median = apply_median_filter(array, median_filter_size)
    gaussian_filter_size = 2
    array_gaussian = apply_gaussian_filter(array_median, gaussian_filter_size)
    thresh_img_inverted = apply_otsu_thresh(array_gaussian)
    thresh_img_with_edges = invert_mask(thresh_img_inverted)
    thresh_img = crop_boundaries(thresh_img_with_edges)
    region_props = get_region_props(thresh_img)
    center_0, center_1 = get_domain_center(array)
    num_regions = 3
    region_props_largest = get_largest_regions(region_props, num_regions)
    microbundle_region = get_closest_region(region_props_largest, center_0, center_1)
    mask = coords_to_mask([microbundle_region.coords], array)
    radius = 3
    closed_mask = close_region(mask, radius)
    return closed_mask


# def microbundle_mask_v2(array: np.ndarray) -> np.ndarray:
#     """Given an image. Will segment the central microbundle and return it as a mask."""
#     gabor_all = gabor_filter(array)
#     median_filter_size = 5
#     median_applied = apply_median_filter(gabor_all, median_filter_size)
#     gaussian_filter_size = 2
#     gaussian_applied = apply_gaussian_filter(median_applied, gaussian_filter_size)
#     thresh_img_inverted = apply_otsu_thresh(gaussian_applied)
#     thresh_img = invert_mask(thresh_img_inverted)
#     region_props = get_region_props(thresh_img)
#     center_0, center_1 = get_domain_center(array)
#     num_regions = 3
#     region_props_largest = get_largest_regions(region_props, num_regions)
#     microbundle_region = get_closest_region(region_props_largest, center_0, center_1)
#     mask = coords_to_mask([microbundle_region.coords], array)
#     radius = 3
#     closed_mask = close_region(mask, radius)
#     return closed_mask