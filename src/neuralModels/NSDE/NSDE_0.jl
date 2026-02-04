using NPZ
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqFlux
using Lux
using Random
using Optimisers
using Zygote
using Statistics
using ProgressMeter
using ComponentArrays

# ============================================================
# 1. LOAD DATA
# ============================================================

reducedState = npzread("data/processed/sstReducedStateCOPERNICUS20102019Prepared.npz")

struct InputVars
    PCsTrain
    PCsVal
    tTrain
    tVal
end

iV = InputVars(
    reducedState["PCsTrain"],
    reducedState["PCsVal"],
    reducedState["tTrain"],
    reducedState["tVal"]
)

PCs = cat(iV.PCsTrain, iV.PCsVal, dims=1)
spanT = size(PCs, 1)

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

ratioTrain = 0.7
Ttrain = Int(round(ratioTrain * spanT))

zTrain = PCs[1:Ttrain, :]
zTest  = PCs[Ttrain+1:end, :]

# ============================================================
# 3. NORMALISATION (TRAIN ONLY)
# ============================================================

μ = mean(zTrain, dims=1)
σ = std(zTrain, dims=1) .+ 1f-6

zTrain = Float32.((zTrain .- μ) ./ σ)
zTest  = Float32.((zTest  .- μ) ./ σ)

nMods = size(zTrain, 2)

z0     = zTrain[1, :]
z0Test = zTest[1, :]

# ============================================================
# 4. NEURAL SDE ARCHITECTURE
# ============================================================

rng = Xoshiro(0)

drift = Lux.Chain(
    Lux.Dense(nMods, 64, tanh),
    Lux.Dense(64, nMods)
)

diffusion = Lux.Chain(
    Lux.Dense(nMods, 64, tanh),
    Lux.Dense(64, nMods),
    Lux.Dense(nMods, nMods, x -> abs.(x))
)

ps_drift, st_drift = Lux.setup(rng, drift)
ps_diff,  st_diff  = Lux.setup(rng, diffusion)

ps = ComponentArray(drift = ps_drift, diffusion = ps_diff)
st = (drift = st_drift, diffusion = st_diff)

# ============================================================
# 5. SDE DEFINITION (DIAGONAL NOISE)
# ============================================================

function f!(du, u, p, t)
    y, _ = drift(u, p.drift, st.drift)
    du .= y
end

function g!(du, u, p, t)
    σ̂, _ = diffusion(u, p.diffusion, st.diffusion)
    du .= 0f0
    @inbounds for i in 1:length(u)
        du[i, i] = 0.05f0 * σ̂[i]
    end
end

tspanTrain = (0f0, Float32(size(zTrain, 1) - 1))

sdeprob = SDEProblem(
    f!,
    g!,
    z0,
    tspanTrain,
    ps
)

# ============================================================
# 6. PREDICTION (MONTE CARLO MEAN)
# ============================================================

function predictSDE(z0, ps; ntraj = 4)
    sols = map(1:ntraj) do _
        sol = solve(
            sdeprob,
            EM(),
            p = ps,
            dt = 1f0,
            adaptive = false,
            saveat = 1f0,
            sensealg = InterpolatingAdjoint()
        )
        Array(sol)
    end

    Tuse = minimum(size.(sols, 2))
    sols = map(Y -> Y[:, 1:Tuse], sols)

    mean(reduce(hcat, sols), dims = 2)
end

# ============================================================
# 7. LOSS FUNCTION
# ============================================================

function statsLoss(pred, truth)
    # pred, truth : (modes × time)

    # Variance par mode
    var_loss = mean((var(pred, dims=2) .- var(truth, dims=2)).^2)

    # Autocorrélation lag-1
    function acf1(x)
        mean(x[:, 1:end-1] .* x[:, 2:end], dims=2)
    end

    acf_loss = mean((acf1(pred) .- acf1(truth)).^2)

    return var_loss + acf_loss
end


function lossSDE(ps)
    pred = predictSDE(z0, ps; ntraj=8)
    statsLoss(pred, zTrain')
end


# ============================================================
# 8. TRAINING
# ============================================================

opt = Optimisers.Adam(1e-3)
optState = Optimisers.setup(opt, ps)

function train!(ps, optState; nEpochs = 100)
    @showprogress for epoch in 1:nEpochs
        loss, back = Zygote.pullback(lossSDE, ps)
        grads = back(1f0)[1]
        optState, ps = Optimisers.update(optState, ps, grads)
        epoch % 10 == 0 && println("Epoch $epoch | Loss = $loss")
    end
    return ps, optState
end

ps, optState = train!(ps, optState)

# ============================================================
# 9. VALIDATION
# ============================================================

Ttest = size(zTest, 1)
tspanTest = (0f0, Float32(Ttest - 1))

sdeprob_test = SDEProblem(
    f!,
    g!,
    z0Test,
    tspanTest,
    ps
)

predTest = zeros(Float32, nMods, Ttest)

for _ in 1:16
    sol = solve(
        sdeprob_test,
        EM(),
        p = ps,
        dt = 1f0,
        adaptive = false,
        saveat = 1f0
    )
    predTest .+= Array(sol)
end

predTest ./= 16

# ============================================================
# 10. METRICS
# ============================================================

mseTest  = mean((predTest .- zTest').^2)
rmseTest = sqrt(mseTest)

println("Validation RMSE (normalized) = ", rmseTest)

errT = vec(mean((predTest .- zTest').^2, dims = 1))
println("Mean error at final time = ", errT[end])

corrModes = [cor(predTest[i, :], zTest[:, i]) for i in 1:min(10, nMods)]
println("Correlation modes 1–10 = ", corrModes)

baselineRmse = sqrt(mean(zTest.^2))
println("Baseline RMSE = ", baselineRmse)
println("Relative RMSE = ", rmseTest / baselineRmse)
