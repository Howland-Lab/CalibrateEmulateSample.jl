# emulate_sample.jl — Emulation and sampling step in CES
# author: Ethan Shin


# Include forward model and utils
include(joinpath(dirname(@__DIR__), "GModel", "GModel.jl"))
include(joinpath(dirname(@__DIR__), "utils", "utils.jl"))
using .utils: construct_priors, phys2comp, success_run_filter

# Import modules
using Distributions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics
using CSV, Tables, DataFrames
using NPZ, MAT
using Interpolations
using DelimitedFiles
using CalibrateEmulateSample
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.DataContainers
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions


################################################## Input settings ################################################################

# Check if enough arguments are passed
if length(ARGS) < 3
    println("Please provide the required inputs.")
else
    # Input arguments
    script_dir = ARGS[1]
    turb_model = ARGS[2]
    svd = parse(Float64, ARGS[3])

    # Print the received arguments
    println("Received script directory path: $script_dir")
    println("Received turbulence model: $turb_model")
    println("Received svd: $svd");
end


################################################### Construct parameter priors ###################################################

params_true, param_names, n_param, prior_lower, prior_upper, priors = construct_priors(turb_model)

# Transformation settings
constrain = true   # Need to transform parameters from computational to physical?
constrain_method = 3   # 1. log transformation, 2. logitnormal transformation, 3. inverse probability integral transformation


################################################### Prepare training data ########################################################

# Load data from results file
datafile = joinpath(script_dir, "calibrate_results.jld2")
inputs = load(datafile)["inputs"]
outputs = load(datafile)["outputs"]
eki_params = load(datafile)["eki_params"]
truth_sample = load(datafile)["truth_sample"]
params_true = load(datafile)["truth_input_constrained"]

# Read in CSV files
yt_csv = joinpath(script_dir, "yt.csv")
yt = CSV.File(yt_csv; header=false); yt = DataFrame(yt); yt = Matrix(yt)
Gy_csv = joinpath(script_dir, "Gy.csv")
Gy = CSV.File(Gy_csv; header=false); Gy = DataFrame(Gy); Gy = Matrix(Gy)

# Change parameters from physical to computational space
comp_params_true = constrain ?
        phys2comp(params_true, constrain_method, prior_lower, prior_upper) :
        omp_params_true = params_true;
comp_inputs = constrain ?
        phys2comp(inputs, constrain_method, prior_lower, prior_upper) :
        comp_inputs = inputs;
comp_eki_params = constrain ?
        phys2comp(eki_params, constrain_method, prior_lower, prior_upper) :
        comp_eki_params = eki_params;

# Filter out all outputs and inputs of NaNs
outputs_success, comp_inputs_success = success_run_filter(outputs, comp_inputs)
# Input-output pairs
input_output_pairs_train = PairedDataContainer(comp_inputs_success, outputs_success, data_are_columns = true)
# Save input-output pairs
@save joinpath(script_dir, "input_output_pairs_train.jld2") input_output_pairs_train


########################################### Emulate: Gaussian Process Regression #################################################

# Emulator settings
gppackage = Emulators.SKLJL()
pred_type = Emulators.YType()
gauss_proc = GaussianProcess(
    gppackage;
    kernel = nothing,
    prediction_type = pred_type,
    noise_learn = false,
)

# Emulator settings
standardize = false;                   # False if already normalized
retained_svd_frac = svd;               # Retaining n% of svd
normalize = true;                      # Normalize inputs
decorrelate = true;                    # Decorrelate output space

# Train emulator
emulator = Emulator(
    gauss_proc,
    input_output_pairs_train;
    obs_noise_cov = Gy,
    normalize_inputs = normalize,
    standardize_outputs = standardize,
    retained_svd_frac = retained_svd_frac,
    decorrelate = decorrelate,
);
optimize_hyperparameters!(emulator)

# Check Gaussian Process regression prediction on true parameters
y_mean, y_var = Emulators.predict(emulator, reshape(comp_params_true, :, 1), transform_to_real = true)
println("ML prediction on true parameters: ")
println(vec(y_mean))
println("true data: ")
println(truth_sample)


############################################## Sample: Markov Chain Monte Carlo ##################################################

# Fix random seed
rng_seed = 413
rng = Random.MersenneTwister(rng_seed)

# Initial values: EKI-calibrated parameters
u0 = vec(mean(comp_inputs_success[:,end-99:end], dims = 2))
println("Initial parameters in computational space: ", u0)
# Run a short chain to determine a good step size
mcmc = MCMCWrapper(RWMHSampling(), truth_sample[:], priors, emulator; init_params = u0)
new_step = optimize_stepsize(rng, mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)
# Begin actual MCMC
println("Begin MCMC - with step size ", new_step)
chain = MarkovChainMonteCarlo.sample(rng, mcmc, 200_000; stepsize = new_step, discard_initial = 2_000)
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

# Save results in jld2 file
save(
    joinpath(script_dir, "posterior.jld2"),
    "posterior",
    posterior,                           
)
