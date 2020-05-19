#!/usr/bin/env bash

###############################################################################################
## SLURM Configurations
##SBATCH --job-name slurm_launcher_pyrado
##SBATCH --array 0-0
###SBATCH --time 24:00:00
##SBATCH --ntasks 1
### Always leave ntasks value to 1. This is only used for MPI, which is not supported now.
##SBATCH --cpus-per-task 12
### Specify the number of cores. The maximum is 32.
###SBATCH --gres=gpu:rtx2080:1
### Leave this if you want to use a GPU per job. Remove it if you do not need it.
###SBATCH -C avx
###SBATCH --mem-per-cpu=2000
##SBATCH -o /home/muratore/Software/SimuRLacra/remotelaunch/slurm_launcher_pyrado/%A_%a-out.txt
##SBATCH -e /home/muratore/Software/SimuRLacra/remotelaunch/slurm_launcher_pyrado/%A_%a-err.txt
###############################################################################################
#
## Your PROGRAM call starts here
#echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"
#
## Activate the pyrado anaconda environment
#eval "$($HOME/Software/anaconda3/bin/conda shell.bash hook)"
#
#SIMURLACRA_DIR="$HOME/Software/SimuRLacra"
#conda activate pyrado
#
## Move to scripts directory
#SCRIPTS_DIR="$SIMURLACRA_DIR/Pyrado/scripts"
#cd "$SCRIPTS_DIR"
#
## Run python scripts with provided command line arguments
#CMD="$@" # all arguments for the script call starting from PATHTO/SimuRLacra/Pyrado/scripts (excluding "python")
#python $CMD
