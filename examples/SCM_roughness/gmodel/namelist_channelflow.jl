# Input case to be used for the SCM
# Listed below are the settings and variables

# A dictionary of parameters
settings = Dict()

# Case name (options: neutral, stable, unstable)
settings["casename"] = "channelflow"

# Numerical timestepping method input
settings["timestep"] = "rk4"
settings["dt"] = 1  # seconds
settings["hours"] = 25;
settings["max_time"] = settings["hours"]*3600

# Vertical domain settings
settings["Lz"] = 600    # height [m]   (FIXED FOR NOW; THIS IS USED TO NONDIMENSIONALIZE z0 AND dPfdx)
settings["nz"] = 36     # input for number of grid points
settings["zEdge"] = range(0, stop=settings["Lz"], length=settings["nz"]+1)
settings["zCell"] = 0.5 .* (settings["zEdge"][2:end] .+ settings["zEdge"][1:end-1])
settings["dz"] = settings["zCell"][2] - settings["zCell"][1]

# Physics variables
settings["ustar"] = 0.4756   # m/s
settings["u_G"] = 0  # m/s
settings["v_G"] = 0  # m/s
settings["dPfdx"] = -(settings["ustar"]^2)/settings["Lz"]  # m/s
settings["dPfdy"] = 0
settings["phi"] = 0  # deg
settings["lambda"] = 40    # m
settings["kappa"] = 0.4  # von Karman constant
settings["l_m"] = (1 ./ (settings["kappa"]*settings["zCell"]) .+ 1/settings["lambda"]).^(-1);   # mixing length (m)
settings["Omega"] = 7.29E-5     # rad / s
settings["f"] = 2 * settings["Omega"] * sin(settings["phi"]*(π /180))     # rad / s

settings["z0"] = 6.8E-5 * settings["Lz"]          # m    surface roughness length for momentum
settings["z0H"] =  6.8E-5 * settings["Lz"]        # m    surface roughness length for heat
settings["zd"] = 0;            # displacement length
settings["matching_idx"] = 3;     # WM matching location

settings["Pr_t"] = 0.6  # turbulent Prandtl number
settings["g"] = 9.81  # m/s^2
settings["T0"] = 300  # reference potential temperature

# Stability function coefficients
settings["A"] = 5;
settings["L"] = Inf;
settings["zeta"] = settings["zCell"] / settings["L"];
settings["psi_M"] = - settings["A"] * settings["zeta"];
settings["psi_H"] = settings["psi_M"];
settings["phi_M"] = 1 .+ settings["A"] * settings["zeta"];
settings["phi_H"] = settings["phi_M"];
settings["phi_E"] = settings["phi_M"] - settings["zeta"];

# Turbulence model options
# "kl" : k-l model (Costa et al., 2020)
# "stdke" : standard k-epsilon model
settings["turb_model"] = "stdke";

if settings["turb_model"] == "kl"
    # the k-l model
    settings["stability"] = 1;
    settings["sig_k"] = 1.0;
    settings["lambda"] = 40;
    settings["alpha"]
elseif settings["turb_model"] == "stdke"
    # the standard k-e model
    settings["C_mu"]= 0.09;
    settings["C_1"] = 1.44;
    settings["C_2"] = 1.92;
    settings["C_3"] = 1.0;
    settings["sig_k"] = 1.0;
    settings["sig_e"] = 1.3;
end

