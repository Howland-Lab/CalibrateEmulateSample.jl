module Observations

# Export functions
export build_observations

# Import modules
using ..Functions: interpolate_zaxis, scale_statistics
using CSV, DataFrames, NPZ
using LinearAlgebra, Statistics, Random


######################################################### helpers ################################################################

# Map integer code to the order of variables
@inline function stats_order(statistics::Int)
    statistics == 1 && return (:U,)
    statistics == 2 && return (:U, :dir)
    statistics == 3 && return (:U, :dir, :k)
    statistics == 4 && return (:U, :k)
    error("Unsupported statistics = $statistics")
end

# Select statistics for observations
@inline function select_statistics(yt_U, yt_dir, yt_k, statistics::Int)
    order = stats_order(statistics)
    parts = map(order) do s
        s === :U   ? yt_U  :
        s === :dir ? yt_dir :
        s === :k   ? yt_k   :
        error("Unknown stat $s")
    end
    return vcat(parts...)
end

# Return mins/maxs aligned with the selected order
@inline function select_minmax(minU, mind, mink, maxU, maxd, maxk, statistics::Int)
    order = stats_order(statistics)
    min_for(sym) = sym === :U ? minU : sym === :dir ? mind : mink
    max_for(sym) = sym === :U ? maxU : sym === :dir ? maxd : maxk
    mins = [min_for(s) for s in order]
    maxs = [max_for(s) for s in order]
    return mins, maxs
end

function load_case!(out::Dict, data_path::AbstractString, name::String,
                    Ug::Real, mask, zCell, interp_z, sample_num::Int)
    filepath = joinpath(data_path, "data/$(name)_samples_10min.npz")
    samples  = npzread(filepath)

    # Dimensionalize and mask
    yt_U   = Ug   * sqrt.(samples["ubar"].^2 .+ samples["vbar"].^2)'
    yt_dir = atand.(samples["vbar"] ./ samples["ubar"])'
    yt_k   = Ug^2 * samples["k"]'
    @views yt_U = yt_U[mask,:];  yt_dir = yt_dir[mask,:];  yt_k = yt_k[mask,:]

    # Interpolate
    yt_U   = interpolate_zaxis(yt_U,   mask, zCell, interp_z)
    yt_dir = interpolate_zaxis(yt_dir, mask, zCell, interp_z)
    yt_k   = interpolate_zaxis(yt_k,   mask, zCell, interp_z)

    # Compute scaling factors
    @views begin
        minU,maxU = extrema(yt_U[:,sample_num])
        mind,maxd = extrema(yt_dir[:,sample_num])
        mink,maxk = extrema(yt_k[:,sample_num])
    end

    # Scale using scaling factors
    yt_U   = scale_statistics(yt_U,   minU, maxU)
    yt_dir = scale_statistics(yt_dir, mind, maxd)
    yt_k   = scale_statistics(yt_k,   mink, maxk)

    # Output
    out["U"]=yt_U
    out["min_U"]=minU
    out["max_U"]=maxU
    out["dir"]=yt_dir
    out["min_dir"]=mind
    out["max_dir"]=maxd
    out["k"]=yt_k
    out["min_k"]=mink
    out["max_k"]=maxk

    return out
end

# Add nugget for numerical stability
function add_nugget!(yt::AbstractMatrix, statistics::Int, nmask::Int; nugget=1e-3)
    @views block_means = [mean(yt[(i-1)*nmask+1:i*nmask, :]) for i in 1:statistics]
    σ = repeat(nugget .* sqrt.(block_means), inner=nmask)
    @views yt .+= σ .* randn(size(yt,1), size(yt,2))
    yt
end


#################################################### export functions ############################################################

function build_observations(data_path::AbstractString, yt_setting::String;
    statistics::Int,
    points_mask::Vector{Int64},
    casename::String,
    interp_z::Vector{Float64},
    zCell::Vector{Float64},
    sample_num::Int=1,
    nugget::Real=1e-3)

    # Choose perfect or imperfect model setting for observations
    if yt_setting == "perfect"
        # Load synthetic data
        yt = Matrix(CSV.read(joinpath(data_path, "data/synthetic_yt.csv"), DataFrame; header=false))

        # Use points mask
        @views yt_U   = yt[points_mask, :]

        # Interpolate to target grid
        yt_U   = interpolate_zaxis(yt_U,   points_mask, zCell, interp_z)

        # Scale per variable
        @views begin
            minU,  maxU  = extrema(yt_U[:,  sample_num])
        end
        yt_mat   = scale_statistics(yt_U,   minU, maxU)

        
        # Select observed variable
        mins, maxs = minU, maxU
        if !iszero(nugget)
            add_nugget!(yt_mat, statistics, points_mask; nugget)
        end

        return yt_mat, mins, maxs 

    # LES data
    elseif yt_setting == "imperfect"

        # Choose case + geostrophic scale
        Ug = casename == "tnbl" ? 12 :
             casename == "sblw" ?  8 :
             casename == "sblm" ?  8 :
             error("Unknown casename = $casename")

        filepath = joinpath(data_path, "data/$(casename)_samples_10min.npz")
        samples  = npzread(filepath)

        # Dimensionalize
        yt_U   = Ug   .* sqrt.(samples["ubar"].^2 .+ samples["vbar"].^2)'
        yt_dir = atand.(samples["vbar"] ./ samples["ubar"])'
        yt_k   = (Ug^2) .* samples["k"]'

        # Use points mask
        @views yt_U   = yt_U[points_mask, :]
        @views yt_dir = yt_dir[points_mask, :]
        @views yt_k   = yt_k[points_mask, :]

        # Interpolate to target grid
        yt_U   = interpolate_zaxis(yt_U,   points_mask, zCell, interp_z)
        yt_dir = interpolate_zaxis(yt_dir, points_mask, zCell, interp_z)
        yt_k   = interpolate_zaxis(yt_k,   points_mask, zCell, interp_z)

        # Scale per variable
        @views begin
            minU,  maxU  = extrema(yt_U[:,  sample_num])
            mind,  maxd  = extrema(yt_dir[:,sample_num])
            mink,  maxk  = extrema(yt_k[:,  sample_num])
        end
        yt_U   = scale_statistics(yt_U,   minU, maxU)
        yt_dir = scale_statistics(yt_dir, mind, maxd)
        yt_k   = scale_statistics(yt_k,   mink, maxk)

        # Select observed variable(s)
        yt_mat = select_statistics(yt_U, yt_dir, yt_k, statistics)
        mins, maxs = select_minmax(minU, mind, mink, maxU, maxd, maxk, statistics)
        if !iszero(nugget)
            add_nugget!(yt_mat, statistics, points_mask; nugget)
        end

        return yt_mat, mins, maxs 
    else
        error("Unknown yt_setting: $yt_setting")
    end
end

end # module