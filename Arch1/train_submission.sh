#!/bin/sh
### Set the job name (for your reference)
#PBS -q low
#PBS -N asset
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends, don't change anything on the below line 
###PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
###PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=8:ncpus=1:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=168:00:00

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
echo "GONNA RUN FILE NOW!"

module () {
        eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
}

module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load compiler/gcc/7.1.0/compilervars
module load compiler/python/3.6.0/ucs4/gnu/447
module load apps/anaconda/3

echo "ALL MODULES LOADED!"

python3 train_model_asset.py ~/scratch/asset_models/asset_m_new_8 ../scratch/asset1 50
