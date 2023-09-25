"""
Python code that takes in a BIDS compliant dataset and performs
  parcellation to generate connectivity matrices
Parcellations generated are: AAL, HarvardOxford, Schaefer, kmeans and ward
SETUP -
1. Update output folder path in 'connectivity_matrices_dir' -
Line 113 (Folder location to save connectivity matrices)
2. Update input folder path in 'for' loop -
Line 141 (Main folder location which has preprocessed functional
  nifti files of ABIDE/ADNI/PPMI/TaoWu/Neurocon dataset)
3. Double check lines 153-156 for input file names (remove run tag if not available)
"""

# Nilearn packages:
from nilearn import datasets  # For atlases

from nilearn import plotting  # To plot brain images
from nilearn.input_data import NiftiLabelsMasker  # To mask the data

# Various packages
import glob
import os
import pandas as pd
import numpy as np
import scipy.io
from nilearn.regions import Parcellations
from nilearn.image import (
    mean_img,
    index_img,
    load_img,
    get_data,
    high_variance_confounds,
)
from nilearn.connectome import ConnectivityMeasure


# Download atlas
atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_filename_schaefer = atlas_schaefer.maps
labels_schaefer = atlas_schaefer.labels

atlas_harvard = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_filename_harvard = atlas_harvard.maps
labels_harvard = atlas_harvard.labels

atlas_aal = datasets.fetch_atlas_aal()
atlas_filename_aal = atlas_aal.maps
labels_aal = atlas_aal.labels


def kmeans_parcellation(functional_img, confounds):
    """Returns kmeans parcellation using the functional image and confounds"""
    kmeans = Parcellations(
        method="kmeans",
        n_parcels=100,
        standardize=True,
        smoothing_fwhm=10.0,
        memory="nilearn_cache",
        memory_level=1,
        verbose=1,
    )
    kmeans.fit(functional_img, confounds=confounds)

    return kmeans.labels_img_


def ward_parcellation(functional_img, confounds_np):
    """Returns ward parcellation using the functional image and counfounds"""
    ward = Parcellations(
        method="ward",
        n_parcels=100,
        standardize=False,
        smoothing_fwhm=2.0,
        memory="nilearn_cache",
        memory_level=1,
        verbose=1,
    )
    # Call fit on functional dataset: single subject (less samples).
    ward.fit(functional_img, confounds=confounds_np)

    # Save the file
    # labels_img.to_filename('parcellation.nii.gz')
    return ward.labels_img_


def extract_timeseries(parcellation_img, functional_img, confounds):
    """extracts the timeseries of each defined ROI (from the parcellation) and 
    the input functional image"""
    masker = NiftiLabelsMasker(
        labels_img=parcellation_img, standardize=True, memory="nilearn_cache", verbose=5
    )
    time_series = masker.fit_transform(functional_img, confounds=confounds)
    return time_series


# Calculate connectivity matrices
def compute_matrices(time_series, filename_prefix, output_dir):
    """generates the correlation matrix and the feature matrix (BOLD time series)"""
    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    mean_correlation_matrix = correlation_measure.mean_

    filename = filename_prefix + "_correlation_matrix.mat"
    scipy.io.savemat(os.path.join(output_dir, filename), {"data": correlation_matrix})

    filename = filename_prefix + "_features_timeseries.mat"
    scipy.io.savemat(os.path.join(output_dir, filename), {"data": time_series})

    return correlation_matrix, mean_correlation_matrix


# 1) To Create - Ouput folder
connectivity_matrices_dir = "/home/data_fmriprep/Conn_matrices_ABIDE"

# 2) Creates directory
if not os.path.exists(connectivity_matrices_dir):
    os.makedirs(connectivity_matrices_dir)
    print(f"Created directory:{connectivity_matrices_dir}")
else:
    print(
        f"Directory {connectivity_matrices_dir} already exists. No directory created."
    )


# 3)
session_list = ["1"]  # ['1', '2']
list_confounds = [
    "csf",
    "white_matter",
    "global_signal",
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
]
kind_connectivity = ["correlation"]
atlas = "schaefer"

for subject in glob.glob("/home/data_ABIDE/derivatives/fmriprep/sub-*", recursive=True):
    if os.path.basename(subject).endswith(".html"):
        print("HTML file - not considered: ", os.path.basename(subject))
    else:
        for session in session_list:
            subject_name = os.path.basename(subject)
            print("Only file name: ", subject_name)

            print(f"Subject: {subject_name}")
            print("------------------------------------")
            print(" ")

            pre_processed_fmri_file = f"{subject}/func/{subject_name}_task-resting_\
run-{session}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            full_confound_file_fmriprep = f"{subject}/func/{subject_name}_task-resting\
_run-{session}_desc-confounds_timeseries.tsv"

            functional_img = load_img(pre_processed_fmri_file)
            print("Shape of image: ", functional_img.get_fdata().shape)  # 4D data

            confounds = pd.read_csv(full_confound_file_fmriprep, delimiter="\t")
            final_confounds = confounds[list_confounds]
            confounds_np = final_confounds.to_numpy()

            # print('Creating a directory to save the computed correlation matrices')
            dir_matrices_derivatives = f"{connectivity_matrices_dir}/{subject_name}/"
            if not os.path.exists(dir_matrices_derivatives):
                os.makedirs(dir_matrices_derivatives)
                print(f"Created directory:{dir_matrices_derivatives}")

                # 1)
                # Schaefer ATLAS PARCELLATION
                # Extract the time series using atlas mask
                schaefer_time_series = extract_timeseries(
                    atlas_filename_schaefer, functional_img, confounds_np
                )

                schaefer_filename_prefix = f"{subject_name}_schaefer100"
                correlation_matrix, mean_schaefer = compute_matrices(
                    schaefer_time_series,
                    schaefer_filename_prefix,
                    dir_matrices_derivatives,
                )

                coordinates_schaefer = plotting.find_parcellation_cut_coords(
                    atlas_filename_schaefer
                )
                
                # plot connectome with 80% edge strength in the connectivity
                plotting.plot_connectome(
                    mean_schaefer,
                    coordinates_schaefer,
                    edge_threshold="80%",
                    title="Schaefer Atlas - Connectome",
                )
                plotting.plot_roi(
                    atlas_filename_schaefer,
                    draw_cross=False,
                    cmap=plotting.cm.bwr_r,
                    title="Schaefer Atlas - ROI",
                )

                try:
                    len(labels_schaefer) in np.shape(correlation_matrix)
                    if False:
                        raise ValueError(
                            "Length of labels do not match size of correlation_matrix"
                        )
                except ValueError:
                    exit("The shape of the matrix and labels are not matching")
                print("The shape of the matrix and labels are matching.")

                # 2)
                # Harvard ATLAS PARCELLATION
                harvard_time_series = extract_timeseries(
                    atlas_filename_harvard, functional_img, confounds_np
                )

                harvard_filename_prefix = f"{subject_name}_harvard48"
                correlation_matrix, mean_harvard = compute_matrices(
                    harvard_time_series,
                    harvard_filename_prefix,
                    dir_matrices_derivatives,
                )

                coordinates_harvard = plotting.find_parcellation_cut_coords(
                    atlas_filename_harvard
                )
                
                # plot connectome with 80% edge strength in the connectivity
                plotting.plot_connectome(
                    mean_harvard,
                    coordinates_harvard,
                    edge_threshold="80%",
                    title="Harvard Atlas - Connectome",
                )
                plotting.plot_roi(
                    atlas_filename_harvard,
                    draw_cross=False,
                    cmap=plotting.cm.bwr_r,
                    title="Harvard Atlas - ROI",
                )

                # 3)
                # AAL ATLAS PARCELLATION
                aal_time_series = extract_timeseries(
                    atlas_filename_aal, functional_img, confounds_np
                )

                aal_filename_prefix = f"{subject_name}_AAL116"
                correlation_matrix, mean_aal = compute_matrices(
                    aal_time_series, aal_filename_prefix, dir_matrices_derivatives
                )

                coordinates_aal = plotting.find_parcellation_cut_coords(
                    atlas_filename_aal
                )
                
                # plot connectome with 80% edge strength in the connectivity
                plotting.plot_connectome(
                    mean_aal,
                    coordinates_aal,
                    edge_threshold="80%",
                    title="AAL Atlas - Connectome",
                )
                plotting.plot_roi(
                    atlas_filename_aal,
                    draw_cross=False,
                    cmap=plotting.cm.bwr_r,
                    title="AAL Atlas - ROI",
                )

                # 4)
                # WARD PARCELLATION

                ward_parcellation_img = ward_parcellation(functional_img, confounds_np)
                ward_time_series = extract_timeseries(
                    ward_parcellation_img, functional_img, confounds_np
                )

                ward_filename_prefix = f"{subject_name}_ward100"
                correlation_matrix, mean_ward = compute_matrices(
                    ward_time_series, ward_filename_prefix, dir_matrices_derivatives
                )

                coordinates_ward = plotting.find_parcellation_cut_coords(
                    ward_parcellation_img
                )
                
                # plot connectome with 80% edge strength in the connectivity
                plotting.plot_connectome(
                    mean_ward,
                    coordinates_ward,
                    edge_threshold="80%",
                    title="Ward - Connectome",
                )
                plotting.plot_roi(
                    ward_parcellation_img,
                    draw_cross=False,
                    cmap=plotting.cm.bwr_r,
                    title="Ward - ROI",
                )

                # 5)
                # K-MEANS PARCELLATION

                kmeans_parcellation_img = kmeans_parcellation(
                    functional_img, confounds_np
                )
                kmeans_time_series = extract_timeseries(
                    kmeans_parcellation_img, functional_img, confounds_np
                )

                kmeans_filename_prefix = f"{subject_name}_kmeans100"
                correlation_matrix, mean_kmeans = compute_matrices(
                    kmeans_time_series, kmeans_filename_prefix, dir_matrices_derivatives
                )

                coordinates_k = plotting.find_parcellation_cut_coords(
                    kmeans_parcellation_img
                )
                
                # plot connectome with 80% edge strength in the connectivity
                plotting.plot_connectome(
                    mean_kmeans,
                    coordinates_k,
                    edge_threshold="80%",
                    title="K means - Connectome",
                )
                plotting.plot_roi(
                    kmeans_parcellation_img,
                    draw_cross=False,
                    cmap=plotting.cm.bwr_r,
                    title="K means - ROI",
                )

                print(f"Extracted connectivity matrix for {subject}")
                print("------------------------------------")

            else:
                print(
                    f"Directory {dir_matrices_derivatives} already exists."
                )
print("Done!")
