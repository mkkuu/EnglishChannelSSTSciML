# Stable version of latent SDE (forward-only, no adjoint)
using Random
using LinearAlgebra
using Lux
using ComponentArrays
using Optimisers
using DifferentialEquations
using StochasticDiffEq
using Zygote
using ProgressMeter
using Statistics

# We preconfigure some parameters
λDiff = 1f-2
obsDim = 150
latentDim = 16
T = 3652
ratioTrain = 0.7
dt = 1.0f0
σMax = 0.05f0
H = 10

# Place real data here
Random.seed!(42)
X = randn(Float32, obs_dim, T)
Ttrain = Int(floor(ratioTrain * T))
Xtrain = X[:, 1:Ttrain]
Xval = X[:, Ttrain+1:end]

encoder = Chain(Dense(obsDim, 64, relu), Dense(64, latentDim))
decoder = Chain(Dense(latentDim, 64, relu), Dense(64, obsDim))

# linear stable drift
A = -0.1f0 * I(latentDim)

driftNet = Chain(
    Dense(latentDim, 32, tanh),
    Dense(32, latentDim)
)
diffNet  = Chain(Dense(latentDim, latentDim), x -> σMax .* tanh.(x))

rng = Random.default_rng()
psEnc, stEnc = Lux.setup(rng, encoder)
psDec, stDec = Lux.setup(rng, decoder)
psF, stF   = Lux.setup(rng, driftNet)
psG, stG   = Lux.setup(rng, diffNet)

ps = ComponentVector(
    encoder = psRnc,
    decoder = psDec,
    drift = psF,
    diff = psG
)

# We set initial states
z0Train, _ = Lux.apply(encoder, Xtrain[:, 1], ps.encoder, stEnc)
z0Val, _ = Lux.apply(encoder, Xval[:, 1],   ps.encoder, stEnc)

z0Train = Array(z0Train)
z0Val = Array(z0Val)

# Latent SDE function
function f!(du, u, p, t)
    nn = first(Lux.apply(driftNet, u, p.drift, stF))
    du .= A * u .+ nn
end

function g!(du, u, p, t)
    du .= first(Lux.apply(diffNet, u, p.diff, stG))
end

# We implement a pure rollout (without mutation)
Zygote.@nograd solve # We precise to Julia
Zygote.@nograd solve!

function rollout(z0, T, ps; ntraj=4)
    tspan = (0f0, Float32(T - 1))
    prob = SDEProblem(f!, g!, z0, tspan, ps)

    sols = map(1:ntraj) do _
        sol = solve(
            prob,
            EM(),
            dt = dt,
            adaptive = false,
            saveat = dt,
            sensealg = Nothing
        )
        Array(sol)
    end

    mean(reduce(hcat, sols), dims=2)
end

# Pure decoding
decode(Z, ps) = reduce(hcat, map(t -> first(Lux.apply(decoder, Z[:, t], ps.decoder, stDec)), axes(Z, 2)))

# Loss function implementation
function loss(ps)
    Z = Zygote.ignore() do
        rollout(z0Train, size(Xtrain, 2), ps; ntraj=2)
    end
    
    X̂ = decode(Z, ps)
    Xref = Xtrain[:, 1:H]

    recon = mean((X̂ .- Xref).^2)

    # diffusion penalty (keeps σ small)
    σ = first(Lux.apply(diffNet, Z[:, 1], ps.diff, stG))
    diffPenalty = mean(σ.^2)

    recon + λDiff * diffPenalty
end

# Then we just train
opt = Optimisers.Adam(1f-3)
optState = Optimisers.setup(opt, ps)

@showprogress for epoch in 1:50
    global optState, ps, loss
    l, back = Zygote.pullback(loss, ps)
    grads = first(back(1f0))
    optState, ps = Optimisers.update(optState, ps, grads)
    epoch % 5 == 0 && println("Epoch $epoch | Train loss = $l")
end

# We test on validation set
Zval = rollout(z0Val, size(Xval, 2), ps; ntraj=32)
X̂val = decode(Zval, ps)

rmse = sqrt(mean((X̂val .- Xval).^2))
baseline = sqrt(mean(Xval.^2))

println("RMSE (normalized) = ", rmse)
println("Baseline RMSE     = ", baseline)
println("Relative RMSE     = ", rmse / baseline)

Tuse = min(size(predTest, 2), size(zTest, 1))

corrModes = [
    cor(predTest[i, 1:Tuse], zTest[1:Tuse, i])
    for i in 1:min(10, nMods)
]

println("Correlation modes 1–10 = ", corrModes)
