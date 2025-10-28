module ProblemSettings

# Export functions
export problem_settings

# Import modules
using Statistics
using ..Functions: create_prior_dict


######################################################### helpers ################################################################

@inline function interp_nz_for(statistics::Int)
    statistics == 1 && return 11
    error("Unsupported statistics = $statistics")
end

@inline function cell_centers(Lz::Real, nz::Int)
    Δ = Lz / nz
    return collect(range(Δ/2, step=Δ, length=nz))
end

@inline function upper_edges(Lz::Real, n_edges::Int)
    Δe = Lz / (n_edges - 1)
    return collect(range(Δe, step=Δe, length=n_edges-1))
end


#################################################### export functions ############################################################

# Sets up specific ABL problem
function problem_settings(statistics::Int, casename::AbstractString, regime::AbstractString;
    
    # Masks
    tnbl_npoints::Int=180,
    sbl_npoints::Int=48,

    # Base grids
    tnbl_Lz::Real=5000, tnbl_nz::Int=360,
    sbl_Lz::Real=400,  sbl_nz::Int=64,

    # ABL heights from LES
    tnbl_interp_Lz::Real=2420,
    sblw_interp_Lz::Real=234,
    sblm_interp_Lz::Real=181,
)
    interp_nz = interp_nz_for(statistics)

    # Points masks
    tnbl_points_mask = 1:tnbl_npoints
    sbl_points_mask  = 1:sbl_npoints

    # LES grid
    zCell_tnbl    = cell_centers(tnbl_Lz, tnbl_nz)
    zCell_sbl     = cell_centers(sbl_Lz,  sbl_nz)

    # Interpolated grid
    tnbl_interp_z = upper_edges(tnbl_interp_Lz, interp_nz)
    sblw_interp_z = upper_edges(sblw_interp_Lz, interp_nz)
    sblm_interp_z = upper_edges(sblm_interp_Lz, interp_nz)

    # Dictionary to collect masks
    masks = Dict(
        "tnbl" => tnbl_points_mask,
        "sblw" => sbl_points_mask,
        "sblm" => sbl_points_mask,
    )

    # Dictionary to collect zcells
    zCells = Dict(
        "tnbl" => zCell_tnbl,
        "sblw" => zCell_sbl,
        "sblm" => zCell_sbl,
    )

    # Dictionary to collect interpolated z-axes
    interp = Dict(
        "tnbl" => tnbl_interp_z,
        "sblw" => sblw_interp_z,
        "sblm" => sblm_interp_z,
    )

    # Choose specific mask for problem
    haskey(masks, casename) || error("Unknown casename: $casename")

    return (
        points_mask        = masks[casename],
        zCell              = zCells[casename],
        interp_z           = interp[casename],
        interp_nz          = interp_nz,
    )
end

end # module
