#!/bin/sh
### Set the job name (for your reference)

#PBS -q low
#PBS -N cuts
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends, don't change anything on the below line
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M cs1190673@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=1:ngpus=1:mem=32G
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=10:00:00
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
#module load apps/anaconda/3EnvCreation

# First argument is the path of the model and second argument is the path of the training data

#for 2011
#python3 CNN_regression_predictions.py ~/workingDir/arch4/Models/ ~/adithya_data_folder/Village_Level_Project_Data/extracted_images_final Tele ~/workingDir/arch4/Data/CNN_regression_predictions/allVillage_list.pickle ~/workingDir/arch4/Output/

#for 2001
python3 split_villages.py /home/cse/mtech/mcs202448/scratch/Raw_Data/2018_Data /home/cse/mtech/mcs202448/adithya_data_folder/cutVillages_2019/state_json /home/cse/mtech/mcs202448/scratch/Split_Data/2018_Split
#~/adithya_data_folder/state_150x_new split('@')[3].split('.')[0]
