module gmodel

# Import libraries
using Random
using Distributions
using LinearAlgebra
using FFTW
using Statistics
using Printf
using Plots
using Dates
using Distributed


# Functions to be exported to scm_example_script.jl
export run_G_ensemble
export scm_run


# Settings for SCM forward model using an input file
mutable struct Settings
    # Dimensional end time
    end_time::Int64
    # Case name
    casename::String
    # Model statistics
    statistics::Int64
    # Averaging time period in hrs
    timeavg::Float64
    # bottom-top height bounds for model statistics
    points_mask::Vector{Int64}
    # Turbulence model
    turb_model::String
    # averaging method
    avg::String
end


# Function to run an ensemble of forward model runs
function run_G_ensemble(params::Array{FT, 2}, settings_local::Settings, experiment_num::Int64, statistics) where {FT <: AbstractFloat}

    # Define the base path
    path = @__DIR__
    
    # Check the case name and include the appropriate namelist file
    if settings_local.casename in ["tnbl", "cnbl", "sbl", "channelflow"]
        include(path * "/namelist_" * settings_local.casename * ".jl")
    else
        error("Invalid casename: " * settings_local.casename)
    end

    # Initialize ensemble
    N_ens = size(params, 2) # params is N_params x N_ens
    g_ens = zeros(length(settings_local.points_mask)*statistics, N_ens);

    # Do I run in parallel? 0. Sequential, 1. Multithread, 2. Parallel map
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


# Takes in the settings and parameters for the model
# Input: settings, parameters
# Output: model statistics
function scm_run(settings_local::Settings, params::Array{FT, 1}) where {FT <: AbstractFloat}

    # Make accessible
    global settings = settings;
    
    # Update general settings
    settings["casename"] = settings_local.casename;
    settings["end_time"] = settings_local.end_time * 3600;     # into seconds
    settings["timeavg"] = Int(settings_local.timeavg * 3600);
    record_time = Int((settings["end_time"] - settings["timeavg"])/settings["dt"]);
    
    # Update roughness length value from ensemble 
    settings["z0"] = params[1];
    settings["z0H"] = params[1];
    settings["zd"] = params[2];
    settings["C_mu"] = params[3];
    settings["C_1"] = params[4];
    settings["C_2"] = params[5];
    settings["C_3"] = params[6];
    settings["sig_k"] = params[7];
    settings["sig_e"] = params[8];

    # Set ustar to dimensionalize runs
    if settings_local.avg == "t"
        settings["ustar"] = 0.4756
    elseif settings_local.avg == "tz"
        settings["ustar"] = 0.4756
    elseif settings_local.avg == "txz"
        settings["ustar"] = 0.4756
    end
    settings["dPfdx"] = -(settings["ustar"]^2)/settings["Lz"]



    ###
    ### Run simulation 
    ###
    # Time settings
    timesteps = 0:settings["dt"]:settings["end_time"]     # discretize time
    convcriteria = 1e-8

    # Make storage variables for time averaging
    u_store,v_store,k_store,e_store,ustar_store = make_storage(settings);
    # Initialize profiles
    settings, u, v, T, k, e = initialize_profiles(settings);
    
    # For comparison in convergence
    prev_u = zeros(Float64, settings["nz"]);
    prev_v = zeros(Float64, settings["nz"]);


    try

        # Numerical timestepping scheme
        for i = 1:length(timesteps)-1 

            # Boundary conditions
            settings, u, v, T, k, e, ustar, uw_s, vw_s, wT_s = bc(settings, u, v, T, k, e, i);
        
            # Store the updated values for quantities of interest
            settings, uw, vw, wT, nu_t, alpha_t, ablh = compute_for_storage(settings, u, v, T, k, e, ustar, uw_s, vw_s, wT_s)
        
            # If in averaging time period
            if i > record_time
                ustar_store[i-record_time] = ustar;
                u_store[:, i-record_time] = u;
                v_store[:, i-record_time] = v;
                k_store[:, i-record_time] = k;
                e_store[:, i-record_time] = e;
            end

            # Numerical timestepping scheme
            if settings["timestep"] == "rk4"
                settings, u, v, T, k, e = rk4(settings, u, v, T, k, e, uw_s, vw_s, wT_s);
            end

            # Check convergence of velocity profiles
            if isnan(mean(u))
                @warn "Simulation fail. Algorithm has computed a NaN."
                break;
            elseif i > 1 && i < settings["end_time"] / settings["dt"]
                diffu = maximum(abs.(u .- prev_u))
                diffv = maximum(abs.(v .- prev_v))
                if diffu < convcriteria && diffv < convcriteria
                    println("Simulation finished. Algorithm has converged.")
                    break;
                else
                    # Update previous values
                    prev_u = copy(u);
                    prev_v = copy(v);
                end
            elseif i == settings["end_time"] / settings["dt"]
                @warn "Simulation success."
                break;
            end    
        end
 
        # Make function to postprocess
        postprocess(settings_local, u, v, k, e);

    catch e
        if isa(e, DomainError)
            println("A DomainError occurred: ", e)
            return fill(NaN, settings["nz"]); 
        else
            rethrow(e)
        end
    end

end


##
## Functions for the forward model
##
function make_storage(settings)
    u_store = zeros(Float64, settings["nz"], div(settings["timeavg"], settings["dt"]));
    v_store = zeros(Float64, settings["nz"], div(settings["timeavg"], settings["dt"]));
    k_store = zeros(Float64, settings["nz"], div(settings["timeavg"], settings["dt"]));
    e_store = zeros(Float64, settings["nz"], div(settings["timeavg"], settings["dt"]));
    ustar_store = zeros(Float64, 1, div(settings["timeavg"], settings["dt"]));   

    return u_store,v_store,k_store,e_store,ustar_store;
end


function initialize_profiles(settings)
    if settings["casename"] == "channelflow"
        # Initial conditions
        u = (1/settings["kappa"]) * log.(settings["zCell"]/settings["z0"])   # initial u profile 
        v = zeros(Float64,settings["nz"]);                              # initial v profile
        T = 300 * ones(Float64,settings["nz"]);                             # initial temperature profile
        k = 0.4 .* (1 .- settings["zCell"] ./ (settings["zCell"][end]/4)) .^ 3; k[settings["zCell"] .> (settings["zCell"][end]/4)] .= 0      # initial TKE profile
    elseif settings["casename"] == "tnbl"
        u = settings["u_G"] .* ones(Float64, settings["nz"])    # initial u profile 
        v = settings["v_G"] .* ones(Float64, settings["nz"])    # initial v profile
    	T = 300 * ones(Float64,settings["nz"]);   # neutral ABL temperature profile
        k = 0.4 .* (1 .- settings["zCell"] ./ 250) .^ 3; k[settings["zCell"] .> 250] .= 0      # initial TKE profile
    elseif settings["casename"] == "cnbl"    
        u = settings["u_G"] .* ones(Float64, settings["nz"])    # initial u profile 
        v = settings["v_G"] .* ones(Float64, settings["nz"])    # initial v profile
        T = similar(settings["zCell"]); 
        T .= 300 .+ 0.003 .* (settings["zCell"] .- 700);    # CNBL temperature profile
        T[settings["zCell"] .< 700] .= 300;        
        T .= T .+ 0.0001 .* settings["zCell"];       # nugget added for stability
        k = 0.4 .* (1 .- settings["zCell"] ./ 250) .^ 3; k[settings["zCell"] .> 250] .= 0      # initial TKE profile
    elseif settings["casename"] == "sbl"
        u = settings["u_G"] .* ones(Float64, settings["nz"])    # initial u profile 
        v = settings["v_G"] .* ones(Float64, settings["nz"])    # initial v profile
        T = similar(settings["zCell"]);
        T .= 265 .+ 0.01 .* (settings["zCell"] .- 100);    # CNBL temperature profile
        T[settings["zCell"] .< 100] .= 265; 
        T .= T .+ 0.0001 .* settings["zCell"];       # nugget added for stability
        k = 0.4 .* (1 .- settings["zCell"] ./ 50) .^ 3; k[settings["zCell"] .> 50] .= 0      # initial TKE profile
    end    
    
    if settings["turb_model"] == "kl"
        e = zeros(Float64, 1, settings["nz"]);      # dummy variable
    elseif settings["turb_model"] == "stdke"
        e = settings["C_mu"]^(3/4) * k.^(3/2)./settings["l_m"];              # initial TKE dissipation profile 
    else
        throw("Unknown turbulence model");
    end

    settings["psi_M"] = 0;
    settings["psi_H"] = 0;
    
    return settings, u, v, T, k, e;
end


function rk4(settings, u, v, T, k, e, uw_s, vw_s, wT_s)
    # Find rk4 coefficients
    k1_u, k1_v, k1_T, k1_k, k1_e = fun(settings, u, v, T, k, e, uw_s, vw_s, wT_s);    
    k2_u, k2_v, k2_T, k2_k, k2_e = fun(settings, u .+ k1_u .* settings["dt"] ./ 2, v .+ k1_v .* settings["dt"] ./ 2, T .+ k1_T .* settings["dt"] ./ 2, k .+ k1_k .* settings["dt"] / 2, e .+ k1_e .* settings["dt"] ./ 2, uw_s, vw_s, wT_s);
    k3_u, k3_v, k3_T, k3_k, k3_e = fun(settings, u .+ k2_u .* settings["dt"] ./ 2, v .+ k2_v .* settings["dt"] ./ 2, T .+ k2_T .* settings["dt"] ./ 2, k .+ k2_k .* settings["dt"] / 2, e .+ k2_e .* settings["dt"] ./ 2, uw_s, vw_s, wT_s);
    k4_u, k4_v, k4_T, k4_k, k4_e = fun(settings, u .+ k3_u .* settings["dt"], v .+ k3_v .* settings["dt"], T .+ k3_T .* settings["dt"], k .+ k3_k .* settings["dt"], e .+ k3_e .* settings["dt"], uw_s, vw_s, wT_s);
   
    # Update u, v, T
    u .+= (1/6) * (k1_u .+ 2*k2_u .+ 2*k3_u .+ k4_u) * settings["dt"];
    v .+= (1/6) * (k1_v .+ 2*k2_v .+ 2*k3_v .+ k4_v) * settings["dt"];
    T .+= (1/6) * (k1_T .+ 2*k2_T .+ 2*k3_T .+ k4_T) * settings["dt"];
    k .+= (1/6) * (k1_k .+ 2*k2_k .+ 2*k3_k .+ k4_k) * settings["dt"]; 
    e .+= (1/6) * (k1_e .+ 2*k2_e .+ 2*k3_e .+ k4_e) * settings["dt"];
    # Regularize k and e
    k[k .< 0] .= 0
    e[e .< 0] .= 0

    return settings, u, v, T, k, e;
end


# Boundary conditions
function bc(settings, u, v, T, k, e, i)
    if settings["casename"] == "tnbl" || settings["casename"] == "cnbl" || settings["casename"] == "channelflow"
        
        # SURFFLUX1 (Basu, 2008)
        U = sqrt(u[settings["matching_idx"]]^2 + v[settings["matching_idx"]]^2);
        ustar = (U*settings["kappa"])/(log((settings["zCell"][settings["matching_idx"]] - settings["zd"])/settings["z0"]) - settings["psi_M"][1]);

        wT_s = 0;
        settings["L"] = Inf;
        settings["zeta"] = settings["zCell"] / settings["L"];
        settings["psi_M"] = - settings["A"] * settings["zeta"];
        settings["psi_H"] = settings["psi_M"];
        settings["phi_M"] = 1 .+ settings["A"] * settings["zeta"];
        settings["phi_H"] = settings["phi_M"];
        settings["phi_E"] = settings["phi_M"] - settings["zeta"];
    elseif settings["casename"] == "sbl"
        T[1] = 265 + settings["coolingrate"]*i*settings["dt"]/3600;
        # SURFFLUX2, Basu 2008
        U = sqrt(u[1]^2 + v[1]^2);
        ustar = (U*settings["kappa"])/(log(settings["zCell"][1]/settings["z0"]) - settings["psi_M"][1]);
        delta_T = T[2] - T[1];
        wT_s = - delta_T * ustar * settings["kappa"] / (log(settings["zCell"][1]/settings["z0H"])  - settings["psi_H"][1]);
        settings["L"] = - settings["T0"] * ustar^3 / (settings["kappa"] * settings["g"] * wT_s);
        settings["zeta"] = settings["zCell"] / settings["L"];
        settings["psi_M"] = - settings["A"] * settings["zeta"];
        settings["psi_H"] = settings["psi_M"];
        settings["phi_M"] = 1 .+ settings["A"] * settings["zeta"];
        settings["phi_H"] = settings["phi_M"];
        settings["phi_E"] = settings["phi_M"] - settings["zeta"];
    end
    
    # Boundary conditions
    uw_s = -(ustar^2)*(u[settings["matching_idx"]]/U);
    vw_s = -(ustar^2)*(v[settings["matching_idx"]]/U);
    
    if settings["casename"] == "tnbl" || settings["casename"] == "cnbl" || settings["casename"] == "sbl"
        u[end] = settings["u_G"];
        v[end] = settings["v_G"];
    end

    if settings["turb_model"] == "kl"
        k[1] = ustar^2 / sqrt(0.09);
        e[1] = 0;
        k[end] = 0;
        e[end] = 0;
    elseif settings["turb_model"] == "stdke"
        k[1] = ustar^2 / sqrt(settings["C_mu"]);
        e[1] = ustar^3 / (settings["kappa"] * (settings["zCell"][1] - settings["zd"]));     # zd has to be smaller than the first half grid scell
        # e[1] = ustar^3 * settings["zCell"][settings["matching_idx"]] / (settings["kappa"] * (settings["zCell"][settings["matching_idx"]] - settings["zd"])^2);
        k[end] = 0;
        e[end] = 0;
    else
        throw("Unknown turbulence model");
    end

    return settings, u, v, T, k, e, ustar, uw_s, vw_s, wT_s;
end


# Function to integrate through timestepping scheme
function fun(settings, u, v, T, k, e, uw_s, vw_s, wT_s)
    # Regularize k and e
    k[k .< 0] .= 0
    e[e .< 0] .= 0
    
    # Formulate eddy viscosity from chosen turbulence model
    nu_t, alpha_t = eddyviscosity(settings, k, e, u, v, T);

    # Compute derivatives
    dudz = CDS2_1(settings["zCell"], u);
    dvdz = CDS2_1(settings["zCell"], v);
    dTdz = CDS2_1(settings["zCell"], T);
    dkdz = CDS2_1(settings["zCell"], k);
    dedz = CDS2_1(settings["zCell"], e);
    # Get gradients
    uw = - nu_t .* dudz; uw[1] = uw_s; 
    vw = - nu_t .* dvdz; vw[1] = vw_s;
    wT = - alpha_t .* dTdz; wT[1] = wT_s;    
    
    # Get flux gradients 
    gradRx = CDS2_1(settings["zCell"], uw);
    gradRy = CDS2_1(settings["zCell"], vw);
    gradT = CDS2_1(settings["zCell"], wT);

    # Governing equations for u, v, T
    dudt = settings["f"] .* (v .- settings["v_G"]) - gradRx .- settings["dPfdx"];
    dvdt = settings["f"] .* (settings["u_G"] .- u) - gradRy .- settings["dPfdy"];
    dTdt = - gradT;    

    # Equations for k, e
    P_s = - uw .* dudz - vw .* dvdz;
    P_b = (settings["g"]/settings["T0"]) .* wT;

    # Turbulence model-specific components
    if settings["turb_model"] == "kl"
        T_k = CDS2_1(settings["zCell"], nu_t/settings["sig_k"]) .* dkdz + (nu_t/settings["sig_k"]) .* CDS2_2(settings["zCell"], k);
        # alpha = 4*(1 .+ 2.5*settings["zCell"]/settings["L"]).^(1/3);
        proxy_e = settings["alpha"].^(3/2) .* (k.^(3/2)) ./ settings["l_m"];

        dkdt = T_k + P_s + P_b - proxy_e;
        dedt = zeros(Float64, 1, settings["nz"]);
    elseif settings["turb_model"] == "stdke"
        T_k = CDS2_1(settings["zCell"], nu_t/settings["sig_k"]) .* dkdz + (nu_t/settings["sig_k"]) .* CDS2_2(settings["zCell"], k);
        T_e = CDS2_1(settings["zCell"], nu_t/settings["sig_e"]) .* dedz + (nu_t/settings["sig_e"]) .* CDS2_2(settings["zCell"], e);
        shear_term = settings["C_1"]*(e./k).*P_s;

        dkdt = T_k + P_s + P_b - e;
        dedt = T_e + shear_term + settings["C_3"]*(e./k).*P_b - settings["C_2"]*(e./k).*e; dedt[isnan.(dedt) .| isinf.(dedt)] .= 0;
    end
    
    return dudt, dvdt, dTdt, dkdt, dedt;
end


# Formulate eddy viscosity from chosen turbulence model
function eddyviscosity(settings, k, e, u, v, T)
    if settings["turb_model"] == "kl"
        Ri = (settings["g"]/settings["T0"])*CDS2_1(settings["zCell"], T)./(CDS2_1(settings["zCell"], u).^2 + CDS2_1(settings["zCell"], v).^2); 
        Ri[isnan.(Ri)] .= 0;
        # alpha = 4*(1 .+ 2.5*settings["zCell"]/settings["L"]).^(1/3);
        if settings["stability"] == 1
            # short tail stability function with critical Richardson number = 0.25
            idx = (Ri .< 0.25);
            fmRi = zeros(size(Ri)); # Initialize fmRi
            fmRi[idx] .= (1 .- Ri[idx] ./ 0.25) .^ 2;
            fmRi[.!idx] .= 0;
        elseif settings["stability"] == 2
            # long tail stability function
            fmRi = (1 + 12*Ri).^(-1);
        else
            throw("Not a valid stability function choice. Not using fm(Ri).");
        end
        nu_t = sqrt.(k.*settings["alpha"]) .* settings["l_m"] .* fmRi;
        nu_t[isnan.(nu_t) .| isinf.(nu_t)] .= 0  
    elseif settings["turb_model"] == "stdke"
        nu_t = settings["C_mu"] * k.^2 ./ e; 
        nu_t[isnan.(nu_t) .| isinf.(nu_t)] .= 0
    end
    alpha_t = nu_t / settings["Pr_t"];
    
    return nu_t, alpha_t;
end


# Compute QoI for storage
function compute_for_storage(settings, u, v, T, k, e, ustar, uw_s, vw_s, wT_s)
    nu_t, alpha_t = eddyviscosity(settings,k,e,u,v,T);
    uw = - nu_t .* CDS2_1(settings["zCell"], u); 
    vw = - nu_t .* CDS2_1(settings["zCell"], v);
    wT = - alpha_t .* CDS2_1(settings["zCell"], T);
    uw[1] = uw_s
    vw[1] = vw_s
    wT[1] = wT_s     
    idx = findfirst(sqrt.(uw.^2 .+ vw.^2) .< 0.05 * (ustar^2))
    ablh = 0; # settings["zCell"][idx];

    return settings, uw, vw, wT, nu_t, alpha_t, ablh;
end


# Central differencing for first derivatives
function CDS2_1(dir, val)
    n = length(dir)
    dvalddir = zeros(n)
    dir = dir[:]
    dirn = zeros(length(dir))
    dirn[1] = dir[2] - dir[1]
    dirn[end] = dir[end] - dir[end-1]
    val = val[:]
    dvalddir[1] = (val[2] - val[1]) / dirn[1]
    dvalddir[2:n-1] = (val[3:end] .- val[1:end-2]) ./ (dir[3:end] .- dir[1:end-2])
    dvalddir[n] = (val[n] - val[n-1]) / dirn[end]
    return dvalddir
end


# Central differencing for second derivatives
function CDS2_2(dir, val)
    n = length(dir)
    dvalddir = zeros(n)
    dir = dir[:]
    dirn = zeros(length(dir))
    dirn[1] = dir[2] - dir[1]
    dirn[end] = dir[end] - dir[end-1]
    val = val[:]
    dvalddir[1] = (val[3] - 2*val[2] + val[1]) / (dir[2] - dir[1])^2
    dvalddir[2:n-1] = (val[3:end] .- 2*val[2:end-1] .+ val[1:end-2]) ./ (dir[2:end-1] .- dir[1:end-2]).^2
    dvalddir[n] = (val[n-2] - 2*val[n-1] + val[n]) / (dir[n] - dir[n-1])^2
    return dvalddir
end



# Postprocess for model statistics
function postprocess(settings_local, u, v, k, e)
    
    result = u[settings_local.points_mask] / u[maximum(settings_local.points_mask)];
    return result;
end

end   # end of module



