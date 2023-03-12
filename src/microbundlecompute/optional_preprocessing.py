from microbundlecompute import image_analysis as ia
from skimage import io
from pathlib import Path
from typing import List
import numpy as np
import cv2
import os


def rename_folder(folder_path: Path, folder_name: str, new_folder_name: str) -> Path:
    """Given a path to a directory, a folder in the given directory, and a new folder name. 
    Will change the name of the folder."""
    original_folder_path = folder_path.joinpath(folder_name).resolve()
    new_folder_path = folder_path.joinpath(new_folder_name).resolve()
    os.rename(original_folder_path,new_folder_path)
    return new_folder_path


def apply_image_kernel(image: np.ndarray, kernel: np.ndarray)-> np.ndarray:
    """ Given an image. Will convolve input kernel with the image. Image depth is preserved."""
    processed_image = cv2.filter2D(image, -1, kernel)
    return processed_image


def filter_all_images(path_list: List, kernel) -> List:
    """ Given a list of image paths. Will return a list of filtered images based on input kernel."""
    filtered_img_list = []
    for img in path_list:
        original_img = io.imread(img)
        filtered_img = apply_image_kernel(original_img, kernel)
        io.imsave(img,filtered_img)
        filtered_img_list.append(filtered_img)
    return filtered_img_list


def run_image_filtering(folder_path: Path, kernel) -> List:
    """ Given a folder path. Will return a list of filtered images based on input kernel."""
    # read images paths
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    # apply kernel to images
    filtered_img_list = filter_all_images(name_list_path, kernel)
    return filtered_img_list


def adjust_first_valley(folder_path: Path, valley_image: int) -> List:
    """ Given a folder path of images. Will remove images prior to the specified "valley_image"."""
    # rename folder to retain original images
    unadjusted_imgs_folder = rename_folder(folder_path,"movie","unadjusted_movie")
    unadjusted_list_path = ia.image_folder_to_path_list(unadjusted_imgs_folder)
    # create a new "movie" folder to save adjusted frames
    adjusted_movie_folder = ia.create_folder(folder_path, "movie")
    number_unadjusted_images = len(unadjusted_list_path)
    # save adjusted images in new folder
    adjusted_img_list = []
    for ff in range(valley_image, number_unadjusted_images):
        img = io.imread(unadjusted_list_path[ff])
        fn = adjusted_movie_folder.joinpath("%04d.TIF"%(ff-valley_image)).resolve()
        io.imsave(fn,img)
        adjusted_img_list.append(fn)
    return adjusted_img_list
