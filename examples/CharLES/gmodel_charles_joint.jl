module gmodel

# Import useful modules
using Random
using Distributions
using LinearAlgebra
using FFTW
using Statistics
using Printf
using Plots
using Distributed
using PyCall

# Functions to be exported to main_template.jl
export run_G_ensemble
export run

# Settings to be inputted into the forward model run
mutable struct Settings
    # Choice of forward model statistics
    statistics::Int64
    # Directory for outputs
    scratch_dir::String
    # Vertical grid points at which we obtain statistics
    points_mask::Vector{Int64}
    # Type of averaging
    avg::String
    # Any settings that are needed and should be easily modifiable inputs for CharLES
end

# Function to transfer an ensemble of parameters into an ensemble of forward model runs
# INPUTS
# params: ensemble of parameters to be run in the forward model
# settings_local: any settings that the mid-fidelity runs may require
# statistics: Total number of output statistics from a single run  (e.g., 64 vector U + 64 vector TKE = 128)
# OUTPUTS
# g_ens: an ensemble of outputs that can be connected back to the input ensemble of parameters
function run_G_ensemble(params::Array{FT, 2}, settings_local::Settings, iter::Int64) where {FT <: AbstractFloat}

    # Initialize ensemble
    N_ens = size(params, 2); # params of size (N_params x N_ens)
    g_ens = zeros(settings_local.statistics, N_ens);    # ensemble of the outputs from the model runs is size (statistics x N_ens)

    # PHASE 1: make changes to input files and create bash_scripts array that connects to input files
    bash_scripts = make_changes_to_inputfiles(params, settings_local, iter);

    # Container to hold jobids
    jobids = Array{Int64,1}(undef, N_ens);
    println(size(jobids));

    # Submit jobs to cluster
    for (i, script) in enumerate(bash_scripts)
        jobid = run_script(script);
        jobids[i] = jobid;
    end
    println("Job ids of submitted runs: ", jobids);

   # To write new bashscript that takes in the iteration number and the jobids (dependencies)
   submit_bash_for_next_iteration(jobids, iter, N_ens, settings_local.avg);
end


# function for making folders for an experiment beforehand (WORKS)
function make_folders(scratch_dir, N_ens, N_iter)
    # Making all folders for data output
    for i in 1:N_iter
        run(`mkdir $scratch_dir/$i`);
        for j in 1:N_ens
            run(`mkdir $scratch_dir/$i/$j`);
        end
    end
end



# Function to run an external script
function run_script(script_path)
    try
        # Run the external script and wait for it to complete
        result = run(pipeline(`sbatch $script_path`; stdout="log.txt"));

        # Find job number
        f = "log.txt";
        job_number = 0;
        for ln in eachline(open(f))
            if (contains(ln,"Submitted batch job "))
                job_number_str = replace(ln, "Submitted batch job "=>"");
                job_number = parse(Int64, job_number_str);
            end
        end

        return job_number;
    catch e
        return "Script $script_path failed with error: $e"
    end
end

# Takes in the settings and parameters for the model and then runs it, then returns post-processed statistics
# Input: settings from .dat input file
# Output: Post-processed statistics from simulation
function make_changes_to_inputfiles(params::Array{FT, 2}, settings::Settings, iter::Int64) where {FT <: AbstractFloat}

    bash_scripts = String[]

    # Make into a loop
    for i in 1:size(params,2)
        # Copy input files and rename to be used
        case_dir = settings.scratch_dir;
	temp_dir = "/home/ctrsp-2024/mjchan/inputfiles_newwm2"
        inputfile_name = "charles.in";
        inputfile_path = string(case_dir,"/",iter,"/",i);
        inputfile = joinpath(inputfile_path, inputfile_name);
        # make copies of input files
        run(`cp $temp_dir/charles.in $inputfile`); 

        # Implement settings to primary and precursor
        resultsdir_string = string("PROBE NAME=", settings.scratch_dir, "/", iter, "/", i, "/ABLpoints  INTERVAL=5300 GEOM=FILE $temp_dir/ABL_points.txt VARS= comp(u,0) comp(u,1) comp(u,2) comp(avg(u),0) comp(avg(u),1) comp(avg(u),2) comp(rms(u),0) comp(rms(u),1) comp(rms(u),2) comp(rey(u),0) comp(rey(u),1) comp(rey(u),2) p avg(p) rms(p) x_cv[0] x_cv[1] x_cv[2]");
        run(`sed -i -E "s+PROBE NAME=.*+$resultsdir_string+" $inputfile`);


        # Assign parameters to the parameters file
        inputfile_name = "myRoughParams.dat";
        paramsfile = joinpath(inputfile_path, inputfile_name);
        println("paramsfile: ", paramsfile);
	println("temp_dir: ", temp_dir);
	# make copies of input files
        run(`cp $temp_dir/myRoughParams.dat $paramsfile`);
        # Change the z0 and d0 values
	z0d0string_st = string("0.0             0.0     0.0", @sprintf("%13.6f",params[1,i][1]), @sprintf("%18.6f",params[2,i][1]));
        z0d0string_en = string("720.0           0.0     0.0", @sprintf("%13.6f",params[1,i][1]), @sprintf("%18.6f",params[2,i][1]));
	println("z0d0string_st: ", z0d0string_st);
        println("z0d0string_en: ", z0d0string_en);
        run(`sed -i -E "s+0.0             0.0     0.0.*+$z0d0string_st+" $paramsfile`);
        run(`sed -i -E "s+720.0           0.0     0.0.*+$z0d0string_en+" $paramsfile`);

        # Modify bash script
        bashfile_name = string("Run",@sprintf("%2.2i", i),".sh");
        bashfile = joinpath(inputfile_path, bashfile_name);
        run(`cp $temp_dir/run.slurm $bashfile`);
        # Only need to change directory of bash file
        problemdir_string = string("export problemDir=\"$inputfile_path\"");
        run(`sed -i -E "s+export problemDir=\".*+$problemdir_string+" $bashfile`);
        rundir_string = string("cd $inputfile_path");
	run(`sed -i -E "s+cd .*+$rundir_string+" $bashfile`);

        push!(bash_scripts, bashfile);
    end

    return bash_scripts;
end


function postprocess(N_ens, statistics, iteration, avg)
    pythonfile = "../postprocess_charles.py";
    run_python_script(pythonfile, N_ens, statistics, iteration, avg);
    println("Post-processing finished.")
end

# Function to run a Python script and capture its output
function run_python_script(script_path, val1, val2, val3, val4)
    # Run using the specific virtual environment
    output = read(`/home/ctrsp-2024/mjchan/miniconda3/envs/myenv/bin/python $script_path $val1 $val2 $val3 $val4`, String)
    return 0;
end


function submit_bash_for_next_iteration(job_ids_str, iter, N_ens, avg);
    # Make bash script
    bashfile_name = string("automate_calibrate_iterate.sh");
    if avg == "t"
        bashfile_path = "/home/ctrsp-2024/mjchan/CalibrateEmulateSample.jl/examples/CharLES/joint_runs_t";
    elseif avg == "txz"
        bashfile_path = "/home/ctrsp-2024/mjchan/CalibrateEmulateSample.jl/examples/CharLES/joint_runs_txz";
    elseif avg == "tz"
        bashfile_path = "/home/ctrsp-2024/mjchan/CalibrateEmulateSample.jl/examples/CharLES/joint_runs_tz";
    end
    bashfile = joinpath(bashfile_path, bashfile_name);

    # Run bash script
    run(`sbatch $bashfile $job_ids_str $iter $N_ens $avg`);
end

end   # end of module
