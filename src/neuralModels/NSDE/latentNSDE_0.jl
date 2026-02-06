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
using NPZ
using Functors

# We preconfigure some parameters
λDiff = 1f-3  # Reduced diffusion penalty
λDrift = 1f-4  # Add drift regularization
latentDim = 8
ratioTrain = 0.7
dt = 1.0f0
σMax = 0.02f0  # Reduced noise
H = 50

# Load data
data = npzread("data/processed/sstReducedState2COPERNICUS20102019.npz")

Z_raw = Float32.(data["Z"])
dZ_raw = Float32.(data["dZ"])

# Transpose to get (modes, time)
X = permutedims(Z_raw)
dX = permutedims(dZ_raw)

X_mean = mean(X, dims=2)
X_std = std(X, dims=2) .+ 1f-8  # Avoid division by zero
X = (X .- X_mean) ./ X_std

obsDim, T = size(X)
println("Data shape after transpose: ", size(X))
println("obsDim = $obsDim, T = $T")
println("Data stats - mean: $(mean(X)), std: $(std(X))")
println("Data range: $(extrema(X))")

# Split data
Ttrain = Int(floor(ratioTrain * T))
Xtrain = X[:, 1:Ttrain]
Xval = X[:, Ttrain+1:end]

println("\nTrain shape: ", size(Xtrain))
println("Val shape: ", size(Xval))

# Networks
encoder = Chain(
    Dense(obsDim, 64, tanh),
    Dense(64, 32, tanh),
    Dense(32, latentDim)
)
decoder = Chain(
    Dense(latentDim, 32, tanh),
    Dense(32, 64, tanh),
    Dense(64, obsDim)
)

# Linear stable drift with stronger damping
A = -0.5f0 * I(latentDim)

driftNet = Chain(
    Dense(latentDim, 32, tanh),
    Dense(32, latentDim)
)
diffNet = Chain(
    Dense(latentDim, 16, tanh),
    Dense(16, latentDim),
    x -> σMax .* tanh.(x)
)

# Initialize parameters
rng = Random.default_rng()
psEnc, stEnc = Lux.setup(rng, encoder)
psDec, stDec = Lux.setup(rng, decoder)
psF, stF   = Lux.setup(rng, driftNet)
psG, stG   = Lux.setup(rng, diffNet)

ps = ComponentVector(
    encoder = psEnc,
    decoder = psDec,
    drift = psF,
    diff = psG
)

# Set initial states
z0Train, _ = Lux.apply(encoder, Xtrain[:, 1], ps.encoder, stEnc)
z0Val, _ = Lux.apply(encoder, Xval[:, 1], ps.encoder, stEnc)

z0Train = Array(z0Train)
z0Val = Array(z0Val)

println("\nLatent dim z0Train: ", size(z0Train))
println("z0 stats - mean: $(mean(z0Train)), std: $(std(z0Train))")

# Latent SDE functions
function f!(du, u, p, t)
    nn = first(Lux.apply(driftNet, u, p.drift, stF))
    du .= A * u .+ nn
end

function g!(du, u, p, t)
    σ = first(Lux.apply(diffNet, u, p.diff, stG))
    du .= σ
end

# Mark solve as non-differentiable
Zygote.@nograd solve
Zygote.@nograd solve!

function rollout(z0, T_steps, ps; ntraj=4)
    tspan = (0f0, Float32(T_steps - 1) * dt)
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

    # Average trajectories
    Z_mean = mean(reduce(hcat, sols), dims=2)
    
    return Z_mean
end

# Decoding function
decode(Z, ps) = reduce(hcat, map(t -> first(Lux.apply(decoder, Z[:, t], ps.decoder, stDec)), axes(Z, 2)))

# Loss function with multiple components
function loss(ps)
    Z = Zygote.ignore() do
        rollout(z0Train, H, ps; ntraj=2)
    end
    
    X̂ = decode(Z, ps)
    Xref = Xtrain[:, 1:min(H, size(Xtrain, 2))]

    # Reconstruction loss
    recon = mean((X̂ .- Xref).^2)

    # Diffusion penalty (keep noise small)
    σ = first(Lux.apply(diffNet, z0Train, ps.diff, stG))
    diffPenalty = mean(σ.^2)
    
    # Drift penalty (regularize drift network)
    drift = first(Lux.apply(driftNet, z0Train, ps.drift, stF))
    driftPenalty = mean(drift.^2)

    total = recon + λDiff * diffPenalty + λDrift * driftPenalty
    
    return total
end

# Diagnostic: test one forward pass
println("\n=== Testing forward pass ===")
try
    l_test = loss(ps)
    println("Initial loss: ", l_test)
    
    # Test rollout
    Z_test = rollout(z0Train, 10, ps; ntraj=2)
    println("Test rollout shape: ", size(Z_test))
    println("Test rollout stats - mean: $(mean(Z_test)), std: $(std(Z_test))")
    println("Test rollout range: $(extrema(Z_test))")
    
    X̂_test = decode(Z_test, ps)
    println("Test decode shape: ", size(X̂_test))
    println("Test decode stats - mean: $(mean(X̂_test)), std: $(std(X̂_test))")
catch e
    println("ERROR in forward pass: ", e)
    rethrow(e)
end

# Training
opt = Optimisers.Adam(5f-4)  # Smaller learning rate
optState = Optimisers.setup(opt, ps)

println("\n=== Starting training ===")
losses = Float32[]

@showprogress for epoch in 1:200
    global loss, ps, optState
    l, back = Zygote.pullback(loss, ps)
    grads = first(back(1f0))
    
    # Gradient clipping
    grads = fmap(grads) do x
        if x isa AbstractArray
            clamp.(x, -1f0, 1f0)
        else
            x
        end
    end
    
    optState, ps = Optimisers.update(optState, ps, grads)
    push!(losses, l)
    
    if epoch % 20 == 0
        println("Epoch $epoch | Train loss = $l")
        
        # Check rollout health
        Z_check = rollout(z0Train, 10, ps; ntraj=2)
        if any(isnan.(Z_check)) || any(isinf.(Z_check))
            println("WARNING: NaN/Inf detected in rollout!")
            break
        end
    end
end

# Validation
println("\n=== Validation ===")

# First check a short rollout
Z_short = rollout(z0Val, 50, ps; ntraj=8)
println("Short rollout stats - mean: $(mean(Z_short)), std: $(std(Z_short))")
println("Short rollout range: $(extrema(Z_short))")

# Full validation
Zval = rollout(z0Val, size(Xval, 2), ps; ntraj=16)
X̂val = decode(Zval, ps)

println("\nPrediction stats:")
println("  X̂val mean: $(mean(X̂val)), std: $(std(X̂val))")
println("  X̂val range: $(extrema(X̂val))")
println("\nTrue stats:")
println("  Xval mean: $(mean(Xval)), std: $(std(Xval))")
println("  Xval range: $(extrema(Xval))")

rmse = sqrt(mean((X̂val .- Xval).^2))
baseline = sqrt(mean(Xval.^2))

println("\nRMSE (normalized) = ", rmse)
println("Baseline RMSE     = ", baseline)
println("Relative RMSE     = ", rmse / baseline)

# Mode-wise correlation with diagnostics
Tuse = min(size(X̂val, 2), size(Xval, 2))
println("\nMode-wise analysis:")

corrModes = Float32[]
for i in 1:obsDim
    pred = X̂val[i, 1:Tuse]
    true_val = Xval[i, 1:Tuse]
    
    # Check variance
    var_pred = var(pred)
    var_true = var(true_val)
    
    if var_pred < 1f-6
        println("  Mode $i: constant prediction (var=$(var_pred))")
        push!(corrModes, 0f0)
    else
        c = cor(pred, true_val)
        push!(corrModes, c)
        println("  Mode $i: cor=$(round(c, digits=3)), var_pred=$(round(var_pred, digits=4)), var_true=$(round(var_true, digits=4))")
    end
end

println("\nMean correlation: ", round(mean(corrModes), digits=3))
println("Median correlation: ", round(median(corrModes), digits=3))

# Plot loss curve
println("\nFinal 10 losses: ", losses[end-9:end])