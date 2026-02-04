# Stable version 2 of latent SDE (forward-only, no adjoint)

using Random, LinearAlgebra, Statistics
using Lux, ComponentArrays, Optimisers
using DifferentialEquations, StochasticDiffEq
using Zygote
using ChainRulesCore
using ProgressMeter

# Same config but we different horizon windows to test
obsDim = 150
latentDim = 16
T = 3652
ratioTrain = 0.7
dt = 1.0f0
σMax = 0.05f0
Hs = [1, 5, 10, 20, 50]
H = 10
λ_diff = 1f-2

# Real data to placehere
Random.seed!(42)
X = randn(Float32, obs_dim, T)

Ttrain = Int(floor(ratioTrain * T))
Xtrain = X[:, 1:Ttrain]
Xval   = X[:, Ttrain+1:end]

# Models
encoder = Chain(Dense(obsDim, 64, relu), Dense(64, latentDim))
decoder = Chain(Dense(latentDim, 64, relu), Dense(64, obsDim))

# Linear drift
A = -0.1f0 * I(latentDim)

driftNet = Chain(Dense(latentDim, 32, tanh), Dense(32, latentDim))

diffNet  = Chain(Dense(latentDim, latentDim), x -> σMax .* tanh.(x))

# Parameters
rng = Random.default_rng()

psEnc, stEnc = Lux.setup(rng, encoder)
psDec, stDec = Lux.setup(rng, decoder)
psF, stF = Lux.setup(rng, driftNet)
psG, stG = Lux.setup(rng, diffNet)

ps = ComponentVector(
    encoder = psEnc,
    decoder = psDec,
    drift = psF,
    diff = psG
)

# After parameter setup, ensure z0_train is properly shaped
z0TrainRaw, _ = Lux.apply(encoder, Xtrain[:, 1], ps.encoder, stEnc)
z0Train = vec(Array(z0TrainRaw))  # Ensure it's a vector, not matrix

println("z0Train shape: ", size(z0Train))  # Should be (16,)

# Latent SDE function
function f!(du, u, p, t)
    nn = first(Lux.apply(driftNet, u, p.drift, stF))
    du .= A * u .+ nn
end

function g!(du, u, p, t)
    du .= first(Lux.apply(diffNet, u, p.diff, stG))
end

# Nuclear no AD (nuclear option)
function rolloutShort(z0, H, ps; ntraj=2)
    tspan = (0f0, Float32(H) * dt)
    prob = SDEProblem(f!, g!, z0, tspan, ps)

    sols = map(1:ntraj) do _
        Array(solve(
            prob,
            EM(),
            dt = dt,
            adaptive = false,
            saveat = dt,
            save_start = true,
            sensealg = nothing
        ))
    end

    Z = cat(sols...; dims=3) # (latent, time, traj)
    Z̄ = mean(Z; dims=3) # (latent, time)

    @assert size(Z̄,2) == H+1 "rolloutShort broken: $(size(Z̄))"
    Z̄
end

# Use ChainRulesCore instead of Zygote.@nograd
ChainRulesCore.@non_differentiable rolloutShort(::Any, ::Any, ::Any)

function rollout(z0, T, ps; ntraj=4)
    tspan = (0f0, Float32((T-1) * dt))
    saveat = Float32.(collect(0:dt:((T-1)*dt)))
    
    prob = SDEProblem(f!, g!, z0, tspan, ps)

    sols = map(1:ntraj) do _
        sol = solve(prob, EM();
            dt = dt,
            adaptive = false,
            saveat = saveat,
            sensealg = nothing)
        Array(sol)
    end

    mean(reduce(hcat, sols), dims=2)
end

ChainRulesCore.@non_differentiable rollout(::Any, ::Any, ::Any)

# Decoding
function decode(Z, ps)
    if size(Z, 2) == 0
        @warn "decode called with empty trajectory"
        return zeros(Float32, obsDim, 0)
    end
    reduce(hcat, map(t -> first(Lux.apply(decoder, Z[:, t], ps.decoder, stDec)), axes(Z, 2)))
end


############################
# LOSS (HORIZON COURT)
############################

function loss(ps)
    totalLoss = 0f0

    for h in Hs
        Z = rolloutShort(z0Train, h, ps; ntraj=2)

        if size(Z, 2) < h + 1
            return 1f10
        end

        Zpred = Z[:, 2:end] # H steps
        X̂ = decode(Zpred, ps)
        Xref = Xtrain[:, 1:h]

        totalLoss += mean((X̂ .- Xref).^2)
    end

    totalLoss /= length(Hs)

    # diffusion regularization
    σ = first(Lux.apply(diffNet, z0Train, ps.diff, stG))
    diffPenalty = mean(σ.^2)

    totalLoss + λDiff * diffPenalty
end


# Add this before training to verify f! and g!
function verifySdeFunctions()
    zTest = randn(Float32, latentDim)
    du = similar(zTest)
    
    # Test drift
    f!(du, zTest, ps, 0f0)
    # println("Drift output shape: ", size(du))
    # println("Drift output: ", du[1:3])
    
    # Test diffusion
    g!(du, z_test, ps, 0f0)
    # println("Diffusion output shape: ", size(du))
    # println("Diffusion output: ", du[1:3])
end

verifySdeFunctions()

# Test the SDE functions directly
function testSdeSolver()
    zTest = randn(Float32, latentDim)
    tspan = (0f0, 10f0)
    saveat = 0f0:1f0:10f0
    
    prob = SDEProblem(f!, g!, zTest, tspan, ps)
    sol = solve(prob, EM(); dt=1f0, adaptive=false, saveat=saveat)
    
    # println("Solution size: ", size(Array(sol)))
    # println("Time points: ", sol.t)
    # println("Expected: 11 points from t=0 to t=10")
    return sol
end

# Run before training
solTest = testSdeSolver()

# Then we train
opt = Optimisers.Adam(1f-3)
optState = Optimisers.setup(opt, ps)

@showprogress for epoch in 1:50
    global ps, optState
    l, back = Zygote.pullback(loss, ps)
    grads = first(back(1f0))
    optState, ps = Optimisers.update(optState, ps, grads)

    epoch % 5 == 0 && println("Epoch $epoch | Train loss = $l")
end

# Validation part
z0Val, _ = Lux.apply(encoder, Xval[:, 1], ps.encoder, stEnc)
z0Val = vec(Array(z0Val))  # Also fix validation z0

Zval = rollout(z0Val, size(Xval, 2), ps; ntraj=32)
X̂val = decode(Zval, ps)

rmse = sqrt(mean((X̂val .- Xval).^2))S
baseline = sqrt(mean(Xval.^2))

println("RMSE (normalized) = ", rmse)
println("Baseline RMSE     = ", baseline)
println("Relative RMSE     = ", rmse / baseline)