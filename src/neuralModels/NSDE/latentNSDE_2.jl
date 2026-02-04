# Stable version 3 of latent SDE (forward-only, no adjoint, with noise annealing)
using Random, LinearAlgebra, Statistics
using Lux, ComponentArrays, Optimisers
using DifferentialEquations, StochasticDiffEq
using Zygote
using ChainRulesCore
using ProgressMeter

# Config
obsDim = 150
latentDim = 16
T = 3652
ratioTrain = 0.7
dt = 1.0f0
σ0 = 0.05f0
τ = 50f0
σEpoch = Ref(σ0)
epochRef = Ref(0f0) # Float32
Hs = [1, 5, 10, 20, 50]
H = 10
λDiff = 1f-2

# Place real data
Random.seed!(42)
X = randn(Float32, obsDim, T)

Ttrain = Int(floor(ratioTrain * T))
Xtrain = X[:, 1:Ttrain]
Xval   = X[:, Ttrain+1:end]

# Models
encoder = Chain(Dense(obsDim, 64, relu), Dense(64, latentDim))
decoder = Chain(
    Dense(2 * latentDim, 64, relu),
    Dense(64, obsDim)
)

# Stable linear drift
A = -0.1f0 * I(latentDim)

driftNet = Chain(Dense(latentDim, 32, tanh),
                  Dense(32, latentDim))

diffNet = Chain(
                Dense(latentDim, latentDim),
                x -> tanh.(x)
                )

# We set encoding/decoding parameters
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
z0Train = vec(Array(z0TrainRaw)) # Ensure it's a vector, not matrix

println("z0Train shape: ", size(z0Train))  # Should be (16,)

# Latent SDE functions + explicit drift definition
function f!(du, u, p, t)
    nn = first(Lux.apply(driftNet, u, p.drift, stF))
    du .= A * u .+ nn
end

function driftExplicit(u, ps)
    nn = first(Lux.apply(driftNet, u, ps.drift, stF))
    return A * u .+ nn
end

function g!(du, u, p, t)
    σ = σEpoch[]
    du .= σ .* first(Lux.apply(diffNet, u, p.diff, stG))
end

# Rollout with no AD
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

# decoding function
function decode(Z, ps)
    reduce(hcat,
        map(t -> begin
            z  = Z[:, t]
            γ = clamp(epochRef[] / 60f0, 0f0, 1f0)
            dz = driftExplicit(z, ps)
            dz = dz ./ (norm(dz) + 1f-4)
            zin = vcat(z, γ .* dz)

            first(Lux.apply(decoder, zin, ps.decoder, stDec))
        end,
        axes(Z, 2)))
end


function encodeTraj(X, ps)
    reduce(hcat,
        map(t -> first(Lux.apply(encoder, X[:, t], ps.encoder, stEnc)),
            axes(X, 2)))
end

# Loss management
Hmax = maximum(Hs)
β = 0.5f0

window_weight(h) = (Float32(h) / Float32(Hmax))^β

function horizonGate(h, epoch)
    # h : horizon
    # epoch : Float32
    t0 = 15f0 * log(Float32(h) + 1f0)
    return clamp((epoch - t0) / 20f0, 0f0, 1f0)
end

function loss(ps)
    epoch = epochRef[]::Float32
    α = clamp(1f0 - epoch / 80f0, 0.2f0, 1f0)

    totalLoss = 0f0

    for h in Hs
        w = window_weight(h) * horizongGate(h, epoch)
        w < 1f-3 && continue
    
        Z = rolloutShort(z0Train, h, ps; ntraj=2)
    
        kl = 0f0
        for t in 1:(size(Z,2)-1)
            pred = A * Z[:,t]
            kl += mean((Z[:,t+1] .- pred).^2)
        end
        kl /= (size(Z,2)-1)
    
        Ztrue = encode_traj(Xtrain[:, 1:(h+1)], ps)
        Zmixed = α .* Z[:, 2:end] .+ (1f0 - α) .* Ztrue[:, 2:end]
    
        X̂ = decode(Zmixed, ps)
        Xref = Xtrain[:, 1:h]
    
        totalLoss += w * (mean((X̂ .- Xref).^2) + 1f-3 * kl)
    end
    
    # Free running decoder stabilization
    Zfree = rollout_short(z0Train, 20, ps; ntraj=2)
    Xfree = decode(Zfree[:, 2:end], ps)
    XrefFree = Xtrain[:, 1:20]

    freeLoss = mean((Xfree .- XrefFree).^2)

    totalLoss += 0.2f0 * freeLoss
    
    totalLoss /= sum(window_weight.(Hs))

    # Autoencoder stabilization
    Zae = encodeTraj(Xtrain[:, 1:50], ps)
    Xae = decode(Zae, ps)
    aeLoss = mean((Xae .- Xtrain[:, 1:50]).^2)

    totalLoss += 0.1f0 * aeLoss

    # Diffusion penalty
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
    g!(du, zTest, ps, 0f0)
    # println("Diffusion output shape: ", size(du))
    # println("Diffusion output: ", du[1:3])
end

verify_sde_functions()

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

# Training part
opt = Optimisers.Adam(1f-3)
optState = Optimisers.setup(opt, ps)

@showprogress for epoch in 1:100
    global ps, optState

    # Noise annealing
    epochRef[] = Float32(epoch) 
    σEpoch[] = σ0 * exp(-epoch / τ)

    l, back = Zygote.pullback(loss, ps)
    grads = first(back(1f0))
    optState, ps = Optimisers.update(optState, ps, grads)

    epoch % 5 == 0 && println("Epoch $epoch | Train loss = $l | σ = $(σEpoch[])")
end

# Then we test on validation set
z0Val, _ = Lux.apply(encoder, Xval[:, 1], ps.encoder, stEnc)
z0Val = vec(Array(z0Val))  # Also fix validation z0

σEpoch[] = 0.0f0
Zval = rollout(z0Val, size(Xval, 2), ps; ntraj=32)
X̂val = decode(Zval, ps)

rmse = sqrt(mean((X̂val .- Xval).^2))
baseline = sqrt(mean(Xval.^2))

println("RMSE (normalized) = ", rmse)
println("Baseline RMSE     = ", baseline)
println("Relative RMSE     = ", rmse / baseline)