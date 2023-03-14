import argparse
from microbundlecompute import optional_preprocessing as op
from microbundlecompute import create_tissue_mask as ctm
from microbundlecompute import image_analysis as ia
from microbundlecompute import strain_analysis as sa
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", help="the user input folder location")
args = parser.parse_args()
input_folder_str = args.input_folder

# input_folder_str = "files/example_data"

self_path_file = Path(__file__)
self_path = self_path_file.resolve().parent
input_folder = self_path.joinpath(input_folder_str).resolve()

"""Movie parameters: frames per second(fps) and length scale (ls) as micrometers/pixel"""
fps = 1 
ls = 1
    
"""Movie preprocessing: Optional"""
"""1. apply a kernel filter"""
# High-pass sharpening filter
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel = None 
if kernel is not None:
    op.run_image_filtering(input_folder, kernel)
else: 
    pass
"""2. adjust first valley"""
first_valley = 0
if first_valley != 0:
    op.adjust_first_valley(input_folder, first_valley) 
else:
    pass

"""Specify if pillar or tissue tracking"""
track_mode = "tissue" # "pillar" or "tissue"

if track_mode == "pillar":
    pdms_E = 1.61 # Provide this value in MPa
    pillar_width = 163 # Provide this value in micrometer (um)
    pillar_thickness = 33.2 # Provide this value in micrometer (um)
    pillar_length = 199.3 # Provide this value in micrometer (um)
    force_location = 163 # Provide this value in micrometer (um)

    # run pillar tracking
    ia.run_pillar_tracking(input_folder, pdms_E, pillar_width, pillar_thickness, pillar_length, force_location, ls)
    ia.visualize_pillar_tracking(input_folder)

elif track_mode == "tissue":
    # automatically create a tissue mask
    # (a manual mask can also be used -- just name it "tissue_mask.txt" -- 1=tissue, 0=background)
    seg_fcn_num = 3
    fname = "tissue_mask"
    frame_num = 0
    method = "minimum"
    ctm.run_create_tissue_mask(input_folder, seg_fcn_num, fname, frame_num, method)
    
    # run the tracking
    ia.run_tracking(input_folder,fps,ls)
    
    # run the tracking visualization
    automatic_color_constraint = True # Put False if manual limits are to be specified
    col_min = 0
    col_max = 3
    col_map = plt.cm.viridis
    ia.run_visualization(input_folder, automatic_color_constraint, col_min, col_max, col_map)
    
    # rotate and interpolate tracking results
    # rotate the results
    input_mask = True  # this will use the mask to determine the rotation vector.
    ia.run_rotation(input_folder, input_mask)
    
    ia.run_rotation_visualization(input_folder, automatic_color_constraint=automatic_color_constraint, col_min=col_min, col_max=col_max, col_map=col_map)

    # interpolate results
    row_vec = np.linspace(215, 305, 12)
    col_vec = np.linspace(120, 400, 30)
    
    row_grid, col_grid = np.meshgrid(row_vec, col_vec)
    row_sample = row_grid.reshape((-1, 1))
    col_sample = col_grid.reshape((-1, 1))
    row_col_sample = np.hstack((row_sample, col_sample))
    fname = "interpolated_rotated"
    ia.run_interpolate(input_folder, row_col_sample, fname, is_rotated=True)
    
    # visualize interpolated tracking results
    ia.visualize_interpolate(input_folder, automatic_color_constraint=automatic_color_constraint, col_min=col_min, col_max=col_max, col_map=col_map, is_rotated=True, interpolation_fname=fname)
    
    # run the strain analysis (will automatically rotate based on the mask)
    pillar_clip_fraction = 0.5
    clip_columns = True
    clip_rows = False
    shrink_row = 0.1
    shrink_col = 0.1
    tile_dim_pix = 40
    num_tile_row = 3
    num_tile_col = 5
    tile_style = 1 # or 2
    manual_sub = False # or True
    sub_extents = None # if manual_sub = True provide as [r0,r1,c0,c1]
    sa.run_sub_domain_strain_analysis(input_folder, pillar_clip_fraction, shrink_row, shrink_col, tile_dim_pix, num_tile_row, num_tile_col, tile_style, is_rotated = True,clip_columns=clip_columns,clip_rows=clip_rows, manual_sub=manual_sub, sub_extents=sub_extents)
    
    # visualize the strain analysis results
    col_min = -0.025
    col_max = 0.025
    col_map = plt.cm.RdBu
    sa.visualize_sub_domain_strain(input_folder, automatic_color_constraint, col_min, col_max, col_map, is_rotated = True)    

else:
    print("track_mode should be either 'pillar' or 'tissue'")

