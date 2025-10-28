module ParameterTransforms

# Export functions
export comp2phys, phys2comp

# Import modules
using Distributions
using CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions


######################################################### HELPERS ################################################################

# TransformMethod types and factory for dispatch
abstract type TransformMethod end

# Logarithmic transform
struct LogTransform <: TransformMethod end

# Logit-normal transform
struct LogitNormalTransform <: TransformMethod end

# Probability integral transform
struct PITransform <: TransformMethod end

# Apply bounding to [lo, hi] range
struct BoundingTransform{T<:TransformMethod} <: TransformMethod
    inner::T
end

# TransformMethod: (1) Logarithmic transform, (2) Logit-normal transform, (3) Probability integral transform
function TransformMethod(m::Int; bounded::Bool=true)
    if m == 0
        return NoTransform()
    else
        base = m == 1 ? LogTransform() :
               m == 2 ? LogitNormalTransform() :
               m == 3 ? PITransform() :
               error("Unknown method $m")
        return bounded ? BoundingTransform(base) : base
    end
end

logit(x)  = log.(x ./ (1 .- x))
ilogit(x) = exp.(x) ./ (1 .+ exp.(x))
const stdnorm = Normal(0,1)
const unif01  = Uniform(0,1)

# Computational → physical
function _to_phys(x::AbstractVector, ::LogTransform)
    exp.(x)
end
function _to_phys(x::AbstractVector, ::LogitNormalTransform)
    ilogit(x)
end
function _to_phys(x::AbstractVector, ::PITransform)
    u = cdf.(stdnorm, x)
    quantile.(unif01, u)
end
function _to_phys(x::AbstractVector, t::BoundingTransform, lo::Real, hi::Real)
    y = _to_phys(x, t.inner)
    (hi - lo) .* y .+ lo
end
function _to_phys(x::AbstractVector, t::TransformMethod, lo::Real, hi::Real)
    if t isa BoundingTransform
        _to_phys(x, t, lo, hi)
    else
        _to_phys(x, t)
    end
end

# Physical → computational 
function _to_comp(x::AbstractVector, ::LogTransform)
    log.(x)
end
function _to_comp(x::AbstractVector, ::LogitNormalTransform)
    logit(x)
end
function _to_comp(x::AbstractVector, ::PITransform)
    u = cdf.(unif01, x)
    quantile.(stdnorm, u)
end
function _to_comp(x::AbstractVector, t::BoundingTransform, lo::Real, hi::Real)
    xb = (x .- lo) ./ (hi - lo)
    _to_comp(xb, t.inner)
end
function _to_comp(x::AbstractVector, t::TransformMethod, lo::Real, hi::Real)
    if t isa BoundingTransform
        _to_comp(x, t, lo, hi)
    else
        _to_comp(x, t)
    end
end


#################################################### EXPORT FUNCTIONS ############################################################

# Map each row from computational to physical space using transformation method
function comp2phys(comp::AbstractMatrix, method::Int, lower::AbstractVector, upper::AbstractVector)
    t = TransformMethod(method; bounded=true)
    out = similar(comp)
    for i in 1:size(comp,1)
        out[i, :] = _to_phys(view(comp,i,:), t, lower[i], upper[i])
    end
    return out
end

# Map each row from physical to computational space using inverse transformation method
function phys2comp(phys::AbstractMatrix, method::Int, lower::AbstractVector, upper::AbstractVector)
    t = TransformMethod(method; bounded=true)
    out = similar(phys)
    for i in 1:size(phys,1)
        out[i, :] = _to_comp(view(phys,i,:), t, lower[i], upper[i])
    end
    return out
end

end # module