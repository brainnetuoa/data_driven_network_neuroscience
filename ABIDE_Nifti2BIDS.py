"""
Python code to convert Nifti files to BIDS format - ABIDE dataset
Download ABIDE dataset from https://ida.loni.usc.edu/home/projectPage.jsp?
                        project=ABIDE&page=HOME&subPage=OVERVIEW_PR

SETUP -
1. Install shutil, os, glob modules
2. The given code is for patient. Rename 'patient' to 'control'
    based on the subjects (Autism or with Normal control)
3. Update input folder path in 'folder' - Line 26
    (The folder must contain all the downloaded subjects from IDA LONI)
    
Sample json files for T1w/anat available in
//sample_json_file/sub-patient50002_T1w.json
(Manually update the metadata information of each subject from IDA LONI)

Sample json file for functional data available in
//sample_json_file/sub-patient50002_task-resting_run-1_bold.json
(Update metadata information of each subject from IDA LONI)
"""

import os
import glob
import shutil

folder = "C:/ABIDE"
for count, filename in enumerate(os.listdir(folder)):

    """ANAT folder name change"""
    anat_nii_destination = f"{folder}/{filename}/MP-RAGE/"
    for anat_nii_source in glob.glob(f"{folder}/{filename}/MP-RAGE/20*/*/*.nii"):
        shutil.move(anat_nii_source, anat_nii_destination)

    shutil.rmtree((f"{folder}/{filename}/MP-RAGE/2000-01-01_00_00_00.0/"))

    anat_json_ref = (
        "//sample_json_file/sub-patient50002_T1w.json"  # Sample anat json file
    )
    # copyfile(anat_json_ref, anat_nii_destination)
    shutil.copy2(anat_json_ref, anat_nii_destination)

    os.rename(
        f"{anat_nii_destination}/sub-patient50002_T1w.json",
        f"{anat_nii_destination}/sub-patient{filename}_T1w.json"
    )
    for anat_nii_source2 in glob.glob(f"{folder}/{filename}/MP-RAGE/*.nii"):
        os.rename(
            (f"{anat_nii_source2}"),
            (f"{anat_nii_destination}/sub-patient{filename}_T1w.nii"),
        )

    os.rename(anat_nii_destination, f"{folder}/{filename}/anat")

    """ FUNC folder name change """
    func_nii_destination = f"{folder}/{filename}/Resting_State_fMRI/"
    for func_nii_source in glob.glob(
        f"{folder}/{filename}/Resting_State_fMRI/20*/*/*.nii"
    ):
        shutil.move(func_nii_source, func_nii_destination)

    shutil.rmtree((f"{folder}/{filename}/Resting_State_fMRI/2000-01-01_00_00_00.0/"))

    func_json_ref = "//sample_json_file/sub-patient50002_task-resting_run-1_bold.json"
    shutil.copy2(func_json_ref, func_nii_destination)

    os.rename(
        (f"{func_nii_destination}/sub-patient50002_task-resting_run-1_bold.json"),
        (f"{func_nii_destination}/sub-patient{filename}_task-resting_run-1_bold.json"),
    )
    for func_nii_source2 in glob.glob(f"{folder}/{filename}/Resting_State_fMRI/*.nii"):
        os.rename(
            (f"{func_nii_source2}"),
            (
                f"{func_nii_destination}/sub-patient{filename}_task-resting_run-1_bold.nii"
            ),
        )

    os.rename(func_nii_destination, f"{folder}/{filename}/func")

    os.rename(f"{folder}/{filename}", f"{folder}/sub-patient{filename}")

    # #Remove specific nii file (Remove the excess folders after conversion)
    # os.remove(f"{folder}/{filename}/anat/{filename}_T1w.nii")
    # os.remove(f"{folder}/{filename}/func/{filename}_task-resting_run-1_bold.nii")
