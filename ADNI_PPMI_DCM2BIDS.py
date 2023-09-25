"""
Python code to convert DICOM images to BIDS format - ADNI dataset
Download ADNI data from https://ida.loni.usc.edu/home/projectPage.jsp?
                         project=ADNI&page=HOME&subPage=OVERVIEW_PR

SETUP -
1. Install dcm2niix, numpy, os, shutil, glob
2. The given code is for subjects with LMCI. Rename 'LMCI' to 'control',
       'patient', 'SMC', 'EMCI', 'MCI' based on the subjects
3. Update output folder path in 'Output main' - Line 22
4. Update input folder path in 'for' loop - Line 29
"""
# importing os module
import os

# from shutil import copyfile
import glob
import shutil
from nipype.interfaces.dcm2nii import Dcm2niix
import numpy as np

Outputmain = "/home/data_fmriprep/OutputADNI/"
if not os.path.exists(Outputmain):
    os.makedirs(Outputmain)
    print(f"Created directory:{Outputmain}")
else:
    print(f"Directory {Outputmain} already exists. None is created.")

for count, subject in enumerate(glob.glob("/home/data_fmriprep/ADNI_LMCI/*")):
    subject_name = os.path.basename(subject)

    print(subject_name)
    print(subject)
    # ************************************************************************
    # DICOM to nii of anat data
    Outputbase = f"{Outputmain}sub-{subject_name}"
    Outputanat = f"{Outputmain}sub-{subject_name}/anat/"

    if not os.path.exists(Outputanat):
        os.makedirs(Outputanat)
        print(f"Created directory:{Outputanat}")
    else:
        print(f"Directory {Outputanat} already exists. None is created.")

    Root_dir = glob.glob(f"{subject}/*MPRAGE/20*/*")
    if np.size(Root_dir) == 0:
        Root_dir = glob.glob(f"{subject}/*SAG*/20*/*")

    if np.size(Root_dir) == 0:
        Root_dir = glob.glob(f"{subject}/*Sag*/20*/*")

    dcmdir = str(Root_dir[0])

    converter = Dcm2niix()
    converter.inputs.source_dir = dcmdir
    converter.inputs.output_dir = Outputanat
    # converter.inputs.compress = 'i'

    converter.cmdline
    "dcm2niix -p y -z y -o Outputanat dcmdir"

    converter.run()
    print("ANAT DATA CONVERSION - DONE")
    # ************************************************************************
    # DICOM to nii of func data
    Outputfunc = f"{Outputmain}sub-{subject_name}/func/"
    if not os.path.exists(Outputfunc):
        os.makedirs(Outputfunc)
        print(f"Created directory:{Outputfunc}")
    else:
        print(f"Directory {Outputfunc} already exists. None is created.")

    Root_dir_func = glob.glob(f"{subject}/*MRI*/20*/*")
    dcmdirfunc = str(Root_dir_func[0])

    converter = Dcm2niix()
    converter.inputs.source_dir = dcmdirfunc
    converter.inputs.output_dir = Outputfunc
    # converter.inputs.compress = 'i'

    converter.cmdline
    "dcm2niix -p y -z y -o Outputfunc dcmdirfunc"

    converter.run()
    print("FUNC DATA CONVERSION - DONE")
    # ************************************************************************
    # Rename anat and func files
    modified_subname = subject_name.replace("_", "")

    niilength = len(os.listdir(f"{Outputbase}/anat/"))
    print(niilength)

    if niilength < 3:
        for anat_json in glob.glob(f"{Outputbase}/anat/*.json"):
            os.rename(anat_json, (f"{Outputanat}sub-LMCI{modified_subname}_T1w.json"))
        for anat_nii in glob.glob(f"{Outputbase}/anat/*.nii.gz"):
            os.rename(anat_nii, (f"{Outputanat}sub-LMCI{modified_subname}_T1w.nii.gz"))
        for func_json in glob.glob(f"{Outputbase}/func/*.json"):
            os.rename(
                func_json,
                (
                    f"{Outputfunc}sub-LMCI{modified_subname}_task-resting_run-1_bold.json"
                ),
            )
        for func_nii in glob.glob(f"{Outputbase}/func/*.nii.gz"):
            os.rename(
                func_nii,
                (
                    f"{Outputfunc}sub-LMCI{modified_subname}_task-resting_run-1_bold.nii.gz"
                ),
            )
    elif niilength > 2:
        # ANAT files
        for anat_extrajson in glob.glob(f"{Outputbase}/anat/*a.json"):
            shutil.move(
                anat_extrajson, (f"{Outputmain}sub-{modified_subname}_T1w.json")
            )
        for anat_extranii in glob.glob(f"{Outputbase}/anat/*a.nii.gz"):
            shutil.move(
                anat_extranii, (f"{Outputmain}sub-{modified_subname}_T1w.nii.gz")
            )
        for anat_json2 in glob.glob(f"{Outputbase}/anat/*.json"):
            os.rename(anat_json2, (f"{Outputanat}sub-LMCI{modified_subname}_T1w.json"))
        for anat_nii2 in glob.glob(f"{Outputbase}/anat/*.nii.gz"):
            os.rename(anat_nii2, (f"{Outputanat}sub-LMCI{modified_subname}_T1w.nii.gz"))

        # FUNC files
        for func_extrajson in glob.glob(f"{Outputbase}/func/*a.json"):
            shutil.move(
                func_extrajson,
                (f"{Outputmain}sub-{modified_subname}_task-resting_run-1_bold.json"),
            )
        for func_extranii in glob.glob(f"{Outputbase}/func/*a.nii.gz"):
            shutil.move(
                func_extranii,
                (f"{Outputmain}sub-{modified_subname}_task-resting_run-1_bold.nii.gz"),
            )
        for func_json2 in glob.glob(f"{Outputbase}/func/*.json"):
            os.rename(
                func_json2,
                (
                    f"{Outputfunc}sub-LMCI{modified_subname}_task-resting_run-1_bold.json"
                ),
            )
        for func_nii2 in glob.glob(f"{Outputbase}/func/*.nii.gz"):
            os.rename(
                func_nii2,
                (
                    f"{Outputfunc}sub-LMCI{modified_subname}_task-resting_run-1_bold.nii.gz"
                ),
            )

    os.rename(Outputbase, f"{Outputmain}sub-LMCI{modified_subname}")

    # #Remove specific nii file (Remove excess folders after conversion)
    # os.remove(f"{folder}/{filename}/anat/{filename}_T1w.nii")
    # os.remove(f"{folder}/{filename}/func/{filename}_task-resting_run-1_bold.nii")
