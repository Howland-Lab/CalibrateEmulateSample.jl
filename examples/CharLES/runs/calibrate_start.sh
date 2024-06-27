#!/bin/bash
#SBATCH -n 32                            # Core count
#SBATCH -N 1                             # Node count
#SBATCH -t 10:00:00                      # Wall clock limit
#SBATCH -p cpu                           # Queue 
#SBATCH -o output_%j.txt                 # Redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt                  # Redirect errors to error_JOBID.txt
#SBATCH --mail-type=BEGIN,END            # Mail when job starts and ends
#SBATCH --mail-user=youngin@mit.edu      # Email recipient


script_dir=$(pwd)

date
pwd

# Pass arguments to Julia
julia ../calibrate_start.jl 
