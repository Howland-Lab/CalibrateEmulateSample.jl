module Functions

# Export functions
export scale_statistics, failed_runs, success_run_filter, shrinkage_cov, interpolate_zaxis, get_standardizing_factors, create_prior_dict

# Import modules
using Distributions
using LinearAlgebra
using FFTW
using Interpolations
using Statistics
using Printf
using Distributed
using PyCall
using CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions


#################################################### export functions ############################################################

# Scale statistics
function scale_statistics(matrix::AbstractMatrix{<:AbstractFloat}, l::Real, u::Real)
    T = eltype(matrix)
    l = T(l)
    u = T(u)
    unbounded = (matrix .- l) ./ (u - l);
    return unbounded
end

# Find failed runs
function failed_runs(g_ens::Array{FT, 2}, physical_params::Array{FT, 2}, prior_lower::Array{FT, 1}, prior_upper::Array{FT, 1}) where {FT <: AbstractFloat}
    # Loop over each column to check the conditions
    for j in 1:size(g_ens, 2)
        if all(g_ens[:,j] .== 0) ||
           any(isnan.(g_ens[:,j])) ||
           any(.!isfinite.(g_ens[:,j])) ||
           any(physical_params[:,j] .== prior_upper) ||
           any(physical_params[:,j] .== prior_lower) ||
           any(g_ens[:,j] .>= 5.0) || 
           any(g_ens[:,j] .<= -1.0)
           
           g_ens[:,j] .= NaN;
        end
    end
    return g_ens
end

# Filter only successful runs
function success_run_filter(g_ens::Array{FT, 2}, physical_params::Array{FT, 2}) where {FT <: AbstractFloat}
    # Delete failed forward model statistics and parameter values
    bad = vec(any(isnan.(g_ens), dims=1))
    keep = .!bad

    g_ens = g_ens[:, keep]
    physical_params = physical_params[:, keep]

    # Count number of fails
    n_fails = count(bad)
    println("Number of fails: ", n_fails)

    return g_ens, physical_params
end

# Ledoit-wolf shrinkage
function shrinkage_cov(sample_mat::AA) where {AA <: AbstractMatrix}
    n_out, n_sample = size(sample_mat)
    sample_mat_zeromean = sample_mat .- mean(sample_mat, dims = 2)

    # get sample covariance
    Γ = cov(sample_mat_zeromean, dims = 2)
    # estimate opt shrinkage
    μ_shrink = 1 / n_out * tr(Γ)
    δ_shrink = norm(Γ - μ_shrink * I)^2 / n_out # (scaled) frob norm of Γ_m
    β_shrink = sum([norm(c * c' - -Γ)^2 / n_out for c in eachcol(sample_mat_zeromean)]) / (n_sample - 1)^2
    γ_shrink = min(β_shrink / δ_shrink, 1) # clipping is typically rare
    #  γμI + (1-γ)Γ
    Γ .*= (1 - γ_shrink)
    for i in 1:n_out
        Γ[i, i] += γ_shrink * μ_shrink
    end

    @info "Shrinkage scale: $(γ_shrink), (0 = none, 1 = revert to scaled Identity)\n shrinkage covariance condition number: $(cond(Γ))"
    return Γ
end

# Interpolate to target z-grid
function interpolate_zaxis(g_ens::AbstractMatrix{FT}, points_mask::Vector{Int64}, zCell::Vector{FT}, interp_z::Vector{FT}) where FT<:AbstractFloat
    g_ens = Matrix(g_ens);
    g_ens_interp = zeros(size(interp_z,1), size(g_ens,2))   # current size of g_ens 
    for i in 1:size(g_ens_interp, 2)
        itp = LinearInterpolation(zCell[points_mask], g_ens[:,i])
        g_ens_interp[:,i] = itp(interp_z)
    end
    result = g_ens_interp
end

# Standardizing factors for GP (2D)
function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2)
    return norm_factor
end

# Standardizing factors for GP (1D)
function get_standardizing_factors(data::Array{FT, 1}) where {FT}
    # Input: data size: N_data*N_ensembles (splatted)
    # Ensemble median of the data
    norm_factor = median(data)
    return norm_factor
end

# Make folders for new experiment
function make_folders(scratch_dir::String, N_ens::Int, N_iter::Int)
    # Ensure main folder exists
    mkpath(scratch_dir)

    # Check existing numbered folders and find the highest one
    existing = filter(x -> isdir(joinpath(scratch_dir, x)) && occursin(r"^\d+$", x),
                      readdir(scratch_dir))
    start_idx = isempty(existing) ? 1 : maximum(parse.(Int, existing)) + 1

    # Create numbered folders starting from available index
    for i in start_idx:(start_idx + N_iter - 1)
        iter_dir = joinpath(scratch_dir, string(i))
        mkpath(iter_dir)
        for j in 1:N_ens
            mkpath(joinpath(iter_dir, string(j)))
        end
    end

    return scratch_dir
end

# Create a prior dictionary given the parameter name (standard normal distribution)
function create_prior_dict(param_name::String)
    return Dict(
        "distribution" => PD.Parameterized(Normal(0, 1)),
        "constraint" => PD.no_constraint(),
        "name" => param_name
    )
end

end # module
