#!/bin/bash

########### WHAT I CHANGE #############################
# Example job_ids array, iteration number, and additional integer input
job_ids=("${@:1:$#-2}")
iteration=${@: -2:1}
n_ens=${@: -1}

#######################################################

# Initialize the dependency string
dependency="aftercorr:"

# Iterate over the array and append each job ID to the dependency string
for job_id in "${job_ids[@]}"
do
    dependency+="$job_id,"
done

# Remove the trailing comma
dependency=${dependency%,}

# Submit the job with the generated dependency string
sbatch --dependency=$dependency calibrate_iterate.sh $iteration $n_ens
