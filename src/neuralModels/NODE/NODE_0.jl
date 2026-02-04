using NPZ
using OrdinaryDiffEq
using DiffEqFlux
using Lux
using Random
using Optimisers
using Zygote
using DifferentialEquations
using Flux
using Statistics
using ProgressMeter
using ComponentArrays


# We open the reduced state previously prepared in Python
reducedState = npzread("data/processed/sstReducedStateCOPERNICUS20102019Prepared.npz")

# for key in keys(reducedState) 
#     println(key)
# end

# println(reducedState["tVal"])

# We set a shape of struct adapted to the input reduced state
struct inputVars
    PCsTrain
    PCsVal
    tTrain
    tVal
end

# We instance the retrieve .npz reduced state
iV = inputVars(reducedState["PCsTrain"], reducedState["PCsVal"], reducedState["tTrain"], reducedState["tVal"])

# println(size(iV.PCsTrain)) # (2922,150)
# println(size(iV.PCsVal)) # (730, 150)

# We remerge initial unsplitted PCs state
PCs = cat(iV.PCsTrain, iV.PCsVal, dims=1)
dT = iV.tTrain[2]-iV.tTrain[1] # We assume time step as already been normalized
spanT = size(PCs)[1]

# println(spanT) # 3652
# println(dT) # 1.0
# println(size(PCs)) # (3652, 150) -> correct because we compute 150 mods and we have 10 years of data with one point for each day

# From now on, we desire to implement Neural ODE network

# Before we are splitting dataset between a train set and a validation set
ratioTrain = 0.7 # We set the ratio of the set we want to train (so we can easily modify it later)
zTrain, zTest = PCs[1:Int32.(round(ratioTrain*spanT)), :], PCs[Int32.(round(ratioTrain*spanT)):spanT, :] # Here we separete train and validation dataset

# println(size(zTrain))
# println(size(zTest))

mu = mean(zTrain, dims=1)
sigma = std(zTrain, dims=1) .+ 1f-6

zTrain = (zTrain .- mu) ./ sigma
zTest  = (zTest  .- mu) ./ sigma


zTrain = Float32.(zTrain)
zTest  = Float32.(zTest)

# Now we can start to set the NN and its parameters
z0 = Float32.(zTrain[1, :])
tSpan = (0, Int32.(round(ratioTrain*spanT)))
nMods = size(zTrain)[2]
# println(z0)
# println(tSpan)
# println(nMods)

# We follow the exemplage of Lux.jl to construct a NN layer
rng = Xoshiro(0)

dZdT = Lux.Chain(
    Lux.Dense(nMods, 64, tanh),
    Lux.Dense(64, nMods)
)

ps, st = Lux.setup(rng, dZdT)
ps = ComponentArray(ps)

Ttrain = size(zTrain, 1)
tspan = (0f0, Float32(Ttrain - 1))

NODE = NeuralODE(
    dZdT,
    tspan,
    Tsit5(),
    saveat = 1f0
)

function predictNODE(z0, ps, st)
    sol, _ = NODE(z0, ps, st)
    Array(sol)
end

function lossNODE(ps)
    pred = predictNODE(z0, ps, st)
    return mean((pred .- zTrain').^2)
end

# To review (on how an optimiser really works)
opt = Optimisers.Adam(1e-3)
optState = Optimisers.setup(opt, ps)

function train!(ps, st, optState; nEpochs=200) 
    @showprogress for epoch in 1:nEpochs 
        loss, back = Zygote.pullback(lossNODE, ps) 
        grads = back(1f0)[1] 
        optState, ps = Optimisers.update(optState, ps, grads) 
        epoch % 10 == 0 && println("Epoch $epoch | Loss = $loss") 
    end 
    return ps, optState 
end


ps, optState = train!(ps, st, optState)

z0Test = zTest[1, :]

Ttest = size(zTest, 1)
tspanTest = (0f0, Float32(Ttest - 1))

NODETest = NeuralODE(
    dZdT,
    tspanTest,
    Tsit5(),
    saveat = 1f0
)

function predictNODETest(z0, ps, st)
    sol, _ = NODETest(z0, ps, st)
    Array(sol)
end

predTest = predictNODETest(z0Test, ps, st)

mseTest  = mean((predTest .- zTest').^2)
rmseTest = sqrt(mseTest)

println("Validation RMSE (normalized) = ", rmseTest)

errT = vec(mean((predTest .- zTest').^2, dims=1))

println("Mean error at final time = ", errT[end])

corrModes = [cor(predTest[i, :], zTest[:, i]) for i in 1:min(10, nMods)]

println("Correlation modes 1–10 = ", corrModes)

baselineRmse = sqrt(mean(zTest.^2))
println("Baseline RMSE = ", baselineRmse)
println("Relative RMSE = ", rmseTest / baselineRmse)

# Validation RMSE (normalized) = 617.4908
# Mean error at final time = 1.1852432e6
# Correlation modes 1–10 = Float32[-0.069957174, -0.12170579, -0.12053006, -0.03875882, -0.15872005, -0.09965332, -0.11037414, 0.0079764705, 0.07323617, -0.0043647634]
# Baseline RMSE = 1.1774399
# Relative RMSE = 524.43506




