############################
# Latent SDE – STABLE VERSION
# Forward-only, no adjoint
############################

using Random
using LinearAlgebra
using Lux
using ComponentArrays
using Optimisers
using DifferentialEquations
using StochasticDiffEq
using SciMLBase
using Zygote
using SciMLSensitivity
using ProgressMeter
using Statistics

############################
# Zygote-safe Latent SDE
############################

using Random, LinearAlgebra, Statistics
using Lux, ComponentArrays, Optimisers
using DifferentialEquations, StochasticDiffEq
using Zygote

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

λ_diff = 1f-2
obs_dim    = 150
latent_dim = 16
T          = 3652
ratioTrain = 0.7
dt         = 1.0f0
σ_max      = 0.05f0
H = 10  # horizon court


# --------------------------------------------------
# DATA (replace with your PCs)
# --------------------------------------------------

Random.seed!(42)
X = randn(Float32, obs_dim, T)

Ttrain = Int(floor(ratioTrain * T))
Xtrain = X[:, 1:Ttrain]
Xval   = X[:, Ttrain+1:end]

# --------------------------------------------------
# MODELS
# --------------------------------------------------

encoder = Chain(Dense(obs_dim, 64, relu), Dense(64, latent_dim))
decoder = Chain(Dense(latent_dim, 64, relu), Dense(64, obs_dim))

# Linear stable drift (OU prior)
A = -0.1f0 * I(latent_dim)

drift_net = Chain(
    Dense(latent_dim, 32, tanh),
    Dense(32, latent_dim)
)

diff_net  = Chain(Dense(latent_dim, latent_dim), x -> σ_max .* tanh.(x))

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

rng = Random.default_rng()

ps_enc, st_enc = Lux.setup(rng, encoder)
ps_dec, st_dec = Lux.setup(rng, decoder)
ps_f,   st_f   = Lux.setup(rng, drift_net)
ps_g,   st_g   = Lux.setup(rng, diff_net)

ps = ComponentVector(
    encoder = ps_enc,
    decoder = ps_dec,
    drift   = ps_f,
    diff    = ps_g
)

# --------------------------------------------------
# INITIAL STATES
# --------------------------------------------------

z0_train, _ = Lux.apply(encoder, Xtrain[:, 1], ps.encoder, st_enc)
z0_val,   _ = Lux.apply(encoder, Xval[:, 1],   ps.encoder, st_enc)

z0_train = Array(z0_train)
z0_val   = Array(z0_val)

# --------------------------------------------------
# LATENT SDE
# --------------------------------------------------

function f!(du, u, p, t)
    nn = first(Lux.apply(drift_net, u, p.drift, st_f))
    du .= A * u .+ nn
end

function g!(du, u, p, t)
    du .= first(Lux.apply(diff_net, u, p.diff, st_g))
end

# --------------------------------------------------
# PURE ROLLOUT (NO MUTATION)
# --------------------------------------------------

function rollout(z0, T, ps; ntraj=4)
    tspan = (0f0, Float32(T - 1))
    prob = SDEProblem(f!, g!, z0, tspan, ps)

    sols = map(1:ntraj) do _
        Array(
            solve(
                prob,
                EM(),
                dt = dt,
                adaptive = false,
                saveat = dt,
                sensealg = Nothing()
            )
        )
    end

    mean(reduce(hcat, sols), dims=2)
end

# --------------------------------------------------
# DECODE (PURE)
# --------------------------------------------------

decode(Z, ps) =
    reduce(hcat, map(t -> first(Lux.apply(decoder, Z[:, t], ps.decoder, st_dec)),
                      axes(Z, 2)))

# --------------------------------------------------
# LOSS (TRAIN ONLY)
# --------------------------------------------------

function loss(ps)
    Z = rollout(z0_train, H, ps; ntraj=2)
    X̂ = decode(Z, ps)
    Xref = Xtrain[:, 1:H]

    recon = mean((X̂ .- Xref).^2)

    # diffusion penalty (keeps σ small)
    σ = first(Lux.apply(diff_net, Z[:, 1], ps.diff, st_g))
    diff_penalty = mean(σ.^2)

    recon + λ_diff * diff_penalty
end


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

opt = Optimisers.Adam(1f-3)
opt_state = Optimisers.setup(opt, ps)

@showprogress for epoch in 1:50
    global opt_state, ps, loss
    l, back = Zygote.pullback(loss, ps)
    grads = first(back(1f0))
    opt_state, ps = Optimisers.update(opt_state, ps, grads)
    epoch % 5 == 0 && println("Epoch $epoch | Train loss = $l")
end

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

Zval = rollout(z0_val, size(Xval, 2), ps; ntraj=32)
X̂val = decode(Zval, ps)

rmse = sqrt(mean((X̂val .- Xval).^2))
baseline = sqrt(mean(Xval.^2))

println("\n===== VALIDATION =====")
println("RMSE (normalized) = ", rmse)
println("Baseline RMSE     = ", baseline)
println("Relative RMSE     = ", rmse / baseline)

Tuse = min(size(predTest, 2), size(zTest, 1))

corrModes = [
    cor(predTest[i, 1:Tuse], zTest[1:Tuse, i])
    for i in 1:min(10, nMods)
]

println("Correlation modes 1–10 = ", corr_modes)
