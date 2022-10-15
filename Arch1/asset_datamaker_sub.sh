#!/bin/sh
### Set the job name (for your reference)

#PBS -q low
#PBS -N asset_maker
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends, don't change anything on the below line
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=2:ncpus=2:ngpus=1:mem=32G
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=3:00:00
#PBS -l software=PYTHON

# After job starts, must goto working directory.
# $PBS_O_WORKDIR is the directory from where the job is fired.
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module () {
        eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
}

module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load compiler/gcc/7.1.0/compilervars
module load compiler/python/3.6.0/ucs4/gnu/447
module load apps/anaconda/3

# First argument is the path of the model and second argument is the path of the training data
python3 /home/cse/mtech/mcs192554/village_level_project_scripts/dataset_maker.py /home/cse/mtech/mcs192554/Village_Level_Project_Data/extracted_images_final /home/cse/mtech/mcs192554/Literacy_kmeans_with_outlier_removal.pickle /home/cse/mtech/mcs192554/scratch/literacy
