#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
#SBATCH --job-name slurm_launcher
#SBATCH --array 0-2
#SBATCH --time 24:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
##SBATCH -C avx
#SBATCH --mem-per-cpu=2000
#SBATCH -o /work/scratch/%u/slurm_launcher/%A_%a-out.txt
#SBATCH -e /work/scratch/%u/slurm_launcher/%A_%a-err.txt
###############################################################################

# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
RESULTS_DIR="/work/scratch/"$USER"/$1"
COMMAND_LINE=${@:2}
python3 test.py \
		$COMMAND_LINE \
		--seed $SLURM_ARRAY_TASK_ID \
		--results-dir $RESULTS_DIR 
