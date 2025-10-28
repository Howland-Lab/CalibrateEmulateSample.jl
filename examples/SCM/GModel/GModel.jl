# Main code to run the forward model for CES
module GModel

# Export fucntions
export run_G_ensemble

# Include forward model submodules
include("BCs/BCs.jl")
include("FD/FD.jl")
include("Init/Init.jl")
include("IO/IO.jl")
include("Namelists/Namelists.jl")
include("Params/Params.jl")
include("TurbMod/TurbMod.jl")
include("Store/Store.jl")
include("TimeInt/TimeInt.jl")
include("Utils/Utils.jl")
include("Core/Core.jl")
using .Core: scm_run

# Import modules
using Random
using Distributions
using LinearAlgebra
using FFTW
using Statistics
using Printf
using Plots
using Dates
using Distributed


# Settings for SCM forward model using an input file
mutable struct Settings{FT <: AbstractFloat, IT <: Int}
    # Dimensional end time in seconds
    end_time::IT
    # Case name
    casename::String
    # Model statistics
    statistics::IT
    # Averaging time period in seconds
    timeavg::IT
    # Vertical grid points at which we obtain statistics
    points_mask::Vector{IT}
    # Geostrophic wind speed
    geo_U::IT
    # Roughness length
    z0::FT
    # Turbulence model
    turb_model::String
end

# Function to run an ensemble of forward model runs (1D)
function run_G_ensemble(params::Array{FT, 1}, settings_local::Settings, experiment_num::IT, statistics::IT) where {FT <: AbstractFloat, IT <: Int}

    # Initialize ensemble
    N_ens = size(params, 2)   # params is N_params × N_ens
    # Decide output length per ensemble member (specific to predefined combinations)
    nmask = length(settings_local.points_mask)
    n_out = if statistics in (1, 2, 3)
        nmask * statistics
    elseif statistics == 4
        nmask * 2
    else
        error("Invalid statistics option: $statistics")
    end
    g_ens = zeros(n_out, N_ens)

    # Run in parallel? 0. Sequential, 1. Multithread, 2. Parallel map
    # By default, using sequential
    parallel_runs = 0;
    if parallel_runs == 0
        # Option 1. Run the model for each ensemble member sequentially
        for i in 1:N_ens
            println("Run ", i);
            g_ens[:, i] = scm_run(settings_local, params[:,i]);
            GC.gc();   # To prevent running out of memory
        end
    elseif parallel_runs == 1
        # Option 2. run using multithreads
        Threads.@threads for i in 1:N_ens
            # Run model with the current parameters
            g_ens[:, i] = scm_run(settings_local, params[:,i]);
        end
    elseif parallel_runs == 2
        # Option 3. run using parallel map
        # Convert to an array of vectors and then run
        params_ensemble = [params[:, i] for i in 1:size(params, 2)]
        results = pmap(x -> scm_run(settings_local, x), params_ensemble)
        g_ens = hcat(results...)
    end

    return g_ens;
end

# Function to run an ensemble of forward model runs (2D)
function run_G_ensemble(params::Array{FT, 2}, settings_local::Settings, experiment_num::IT, statistics::IT) where {FT <: AbstractFloat, IT <: Int}

    # Initialize ensemble
    N_ens = size(params, 2)   # params is N_params × N_ens
    # Decide output length per ensemble member (specific to predefined combinations)
    nmask = length(settings_local.points_mask)
    n_out = if statistics in (1, 2, 3)
        nmask * statistics
    elseif statistics == 4
        nmask * 2
    else
        error("Invalid statistics option: $statistics")
    end
    g_ens = zeros(n_out, N_ens)

    # Run in parallel? 0. Sequential, 1. Multithread, 2. Parallel map
    # By default, using sequential
    parallel_runs = 0;
    if parallel_runs == 0
        # Option 1. Run the model for each ensemble member sequentially
        for i in 1:N_ens
            println("Run ", i);
            g_ens[:, i] = scm_run(settings_local, params[:,i]);
            GC.gc();   # Essential to not run out of memory
        end
    elseif parallel_runs == 1
        # Option 2. run using multithreads
        Threads.@threads for i in 1:N_ens
            # Run model with the current parameters
            g_ens[:, i] = scm_run(settings_local, params[:,i]);
        end
    elseif parallel_runs == 2
        # Option 3. run using parallel map
        # Convert to an array of vectors and then run
        params_ensemble = [params[:, i] for i in 1:size(params, 2)]
        results = pmap(x -> scm_run(settings_local, x), params_ensemble)
        g_ens = hcat(results...)
    end

    return g_ens;
end

end   # module
