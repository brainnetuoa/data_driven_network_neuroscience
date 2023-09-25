"""
Calculates summary statistics of the edge weights for each dataset

SETUP -
1. Enter input folder - line 14
"""
import os
import pandas as pd
import numpy as np
import scipy.io

import glob

ip_path = r"C:/python/Data/taowu\*"
print(ip_path)


def statistics_matrix(matrix_stat):
    mat_avg = np.average(matrix_stat["data"])
    mat_min = np.min(matrix_stat["data"])
    mat_max = np.max(matrix_stat["data"])
    mat_std = np.std(matrix_stat["data"])
    mat_sum = np.sum(matrix_stat["data"])
    return mat_avg, mat_min, mat_max, mat_std, mat_sum


(
    overall_mat_avg_aal,
    overall_mat_min_aal,
    overall_mat_max_aal,
    overall_mat_std_aal,
    overall_mat_sum_aal,
) = ([], [], [], [], [])
(
    overall_mat_avg_h,
    overall_mat_min_h,
    overall_mat_max_h,
    overall_mat_std_h,
    overall_mat_sum_h,
) = ([], [], [], [], [])
(
    overall_mat_avg_kmeans,
    overall_mat_min_kmeans,
    overall_mat_max_kmeans,
    overall_mat_std_kmeans,
    overall_mat_sum_kmeans,
) = ([], [], [], [], [])
(
    overall_mat_avg_sch,
    overall_mat_min_sch,
    overall_mat_max_sch,
    overall_mat_std_sch,
    overall_mat_sum_sch,
) = ([], [], [], [], [])
(
    overall_mat_avg_ward,
    overall_mat_min_ward,
    overall_mat_max_ward,
    overall_mat_std_ward,
    overall_mat_sum_ward,
) = ([], [], [], [], [])

for subject in glob.glob(ip_path, recursive=True):  # Input path
    subject_name = os.path.basename(subject)
    # print("Only file name: ", subject_name)

    aal_corr = f"{subject}/{subject_name}_AAL116_correlation_matrix.mat"
    harvard_corr = f"{subject}/{subject_name}_harvard48_correlation_matrix.mat"
    kmeans_corr = f"{subject}/{subject_name}_kmeans100_correlation_matrix.mat"
    schaefer_corr = f"{subject}/{subject_name}_schaefer100_correlation_matrix.mat"
    ward_corr = f"{subject}/{subject_name}_ward100_correlation_matrix.mat"

    # AAL
    matrix_aal = scipy.io.loadmat(aal_corr)
    np.fill_diagonal(matrix_aal["data"], 0)
    mat_avg_aal, mat_min_aal, mat_max_aal, mat_std_aal, mat_sum_aal = statistics_matrix(
        matrix_aal
    )
    overall_mat_avg_aal.append(mat_avg_aal)
    overall_mat_min_aal.append(mat_min_aal)
    overall_mat_max_aal.append(mat_max_aal)
    overall_mat_std_aal.append(mat_std_aal)
    overall_mat_sum_aal.append(mat_sum_aal)
    # harvard
    matrix_harvard = scipy.io.loadmat(harvard_corr)
    np.fill_diagonal(matrix_harvard["data"], 0)
    mat_avg_h, mat_min_h, mat_max_h, mat_std_h, mat_sum_h = statistics_matrix(
        matrix_harvard
    )
    overall_mat_avg_h.append(mat_avg_h)
    overall_mat_min_h.append(mat_min_h)
    overall_mat_max_h.append(mat_max_h)
    overall_mat_std_h.append(mat_std_h)
    overall_mat_sum_h.append(mat_sum_h)
    # kmeans
    matrix_kmeans = scipy.io.loadmat(kmeans_corr)
    np.fill_diagonal(matrix_kmeans["data"], 0)
    (
        mat_avg_kmeans,
        mat_min_kmeans,
        mat_max_kmeans,
        mat_std_kmeans,
        mat_sum_kmeans,
    ) = statistics_matrix(matrix_kmeans)
    overall_mat_avg_kmeans.append(mat_avg_kmeans)
    overall_mat_min_kmeans.append(mat_min_kmeans)
    overall_mat_max_kmeans.append(mat_max_kmeans)
    overall_mat_std_kmeans.append(mat_std_kmeans)
    overall_mat_sum_kmeans.append(mat_sum_kmeans)
    # schaefer
    matrix_schaefer = scipy.io.loadmat(schaefer_corr)
    np.fill_diagonal(matrix_schaefer["data"], 0)
    mat_avg_sch, mat_min_sch, mat_max_sch, mat_std_sch, mat_sum_sch = statistics_matrix(
        matrix_schaefer
    )
    overall_mat_avg_sch.append(mat_avg_sch)
    overall_mat_min_sch.append(mat_min_sch)
    overall_mat_max_sch.append(mat_max_sch)
    overall_mat_std_sch.append(mat_std_sch)
    overall_mat_sum_sch.append(mat_sum_sch)

    # ward
    matrix_ward = scipy.io.loadmat(ward_corr)
    np.fill_diagonal(matrix_ward["data"], 0)
    (
        mat_avg_ward,
        mat_min_ward,
        mat_max_ward,
        mat_std_ward,
        mat_sum_ward,
    ) = statistics_matrix(matrix_ward)
    overall_mat_avg_ward.append(mat_avg_ward)
    overall_mat_min_ward.append(mat_min_ward)
    overall_mat_max_ward.append(mat_max_ward)
    overall_mat_std_ward.append(mat_std_ward)
    overall_mat_sum_ward.append(mat_sum_ward)

# dataframe Name and Age columns
df = pd.DataFrame(
    {
        "Method": ["AAL", "Harvard", "Schaefer", "Kmeans", "Ward"],
        "Average": [
            np.std(overall_mat_avg_aal),
            np.std(overall_mat_avg_h),
            np.std(overall_mat_avg_sch),
            np.std(overall_mat_avg_kmeans),
            np.std(overall_mat_avg_ward),
        ],
        "Minimum": [
            np.std(overall_mat_min_aal),
            np.std(overall_mat_min_h),
            np.std(overall_mat_min_sch),
            np.std(overall_mat_min_kmeans),
            np.std(overall_mat_min_ward),
        ],
        "Maximum": [
            np.std(overall_mat_max_aal),
            np.std(overall_mat_max_h),
            np.std(overall_mat_max_sch),
            np.std(overall_mat_max_kmeans),
            np.std(overall_mat_max_ward),
        ],
        "SD": [
            np.std(overall_mat_std_aal),
            np.std(overall_mat_std_h),
            np.std(overall_mat_std_sch),
            np.std(overall_mat_std_kmeans),
            np.std(overall_mat_std_ward),
        ],
        "Sum": [
            np.std(overall_mat_sum_aal),
            np.std(overall_mat_sum_h),
            np.std(overall_mat_sum_sch),
            np.std(overall_mat_sum_kmeans),
            np.std(overall_mat_sum_ward),
        ],
    }
)

writer = pd.ExcelWriter("Results.xlsx", engine="xlsxwriter")
df.to_excel(writer, sheet_name="Sheet1", index=False)
writer.save()
