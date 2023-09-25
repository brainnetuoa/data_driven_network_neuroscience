#fmriprep script
#Update location of fmriprep freesurfer license (line 18) - https://surfer.nmr.mgh.harvard.edu/registration.html.

#User inputs:
bids_root_dir=$HOME/data_fmriprep/ABIDE #Enter Input folder
subj=patient50002 #Enter subject details - {subject_Info}{ID}
nthreads=8 #number of processors
mem=20 #gb
container=docker #docker or singularity

#Begin:

#Convert virtual memory from gb to mb
mem=`echo "${mem//[!0-9]/}"` #remove gb at end
mem_mb=`echo $(((mem*1000)-5000))` #reduce some memory for buffer space during pre-processing

#export TEMPLATEFLOW_HOME=$home/.cache/templateflow
export FS_LICENSE=$HOME/fmriprep_data/flanker/derivatives/license.txt

#Run fmriprep
if [ $container == singularity ]; then
  unset PYTHONPATH; singularity run -B $HOME/.cache/templateflow:/opt/templateflow $HOME/fmriprep.simg \
    $bids_root_dir $bids_root_dir/derivatives \
    participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file $HOME/fmriprep_data/flanker/derivatives/license.txt \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --mem_mb $mem_mb \
else
  fmriprep-docker $bids_root_dir $bids_root_dir/derivatives \
    participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file $HOME/fmriprep_data/flanker/derivatives/license.txt \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --mem_mb $mem_mb
    - $HOME
fi
