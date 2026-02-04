# ROM version of NODE

using NPZ, LinearAlgebra, Random, Statistics
using Lux, ComponentArrays
using OrdinaryDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers

reducedState = npzread("data/processed/sstReducedStateCOPERNICUS20102019Prepared.npz")

PCsTrainRaw = Float32.(reducedState["PCsTrain"])
tTrainRaw = Float32.(reducedState["tTrain"])

# println("Raw PCsTrain size = ", size(PCsTrain_raw))
# println("Raw tTrain length = ", length(tTrain_raw))

# Auto-detect orientation → want Z[mode, time]
if size(PCsTrainRaw, 2) == length(tTrainRaw)
    Zfull = PCsTrainRaw
    println("PCsTrain already (modes × time)")
elseif size(PCsTrainRaw, 1) == length(tTrainRaw)
    Zfull = PCsTrainRaw'
    println("PCsTrain transposed to (modes × time)")
else
    error("Cannot infer orientation of PCsTrain")
end

tfull = tTrainRaw

@assert size(Zfull, 2) == length(tfull)

println("Final Zfull size = ", size(Zfull), " (modes × time)")

# We split as 80/20
T = size(Zfull, 2)
splitIdx = Int(floor(0.8 * T))

ZtrainFull = Zfull[:, 1:splitIdx]
Zval = Zfull[:, splitIdx+1:end]

tTrain_full = tfull[1:splitIdx]
tVal = tfull[splitIdx+1:end]

# println("Ztrain_full size = ", size(ZtrainFull))
# println("Zval size = ", size(Zval))

# We select an horizon to train NODE on
Tshort = 400

Ztrain = ZtrainFull[:, 1:Tshort]
tTrain = tTrainFull[1:Tshort]

# We normalize
stdMode = vec(std(Ztrain; dims=2))

Ztrain .= Ztrain ./ reshape(std_mode, :, 1)
Zval .= Zval ./ reshape(std_mode, :, 1)

# We define our neural network
r = size(Ztrain, 1) # number of modes

z0Train = Ztrain[:, 1]
z0Val = Zval[:, 1]

tspanTrain = (tTrain[1], tTrain[end])
tspanVal = (tVal[1], tVal[end])

nn = Chain(
    Dense(r, 32, tanh),
    Dense(32, r)
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, nn)
ps = ComponentArray(ps)

λ = 0.1f0  # dissipation coefficient we can adjust

function f!(dz, z, p, t)
    y, _ = nn(z, p, st)
    @. dz = y - λ * z
end

probTrainTemplate = ODEProblem(f!, z0Train, tspanTrain, nothing)
probValTemplate = ODEProblem(f!, z0Val,   tspanVal,   nothing)

# We set our predict and loss function
function predict(ps, probTemplate, tSave)
    prob = remake(probTemplate; p=ps)
    sol = solve(
        prob,
        Tsit5(),
        saveat = tSave,
        dense = false, # precise to avoid an error
        sensealg = InterpolatingAdjoint(autodiff=ZygoteVJP()),
        verbose = false
    )
    return Array(sol)
end

function loss(ps)
    Ẑ = predict(ps, probTrainTemplate, tTrain)
    return sum(abs2, Ẑ .- Ztrain) / length(Ztrain)
end

# println("Warm-up loss (compile):")
# @time println("Initial loss = ", loss(ps))

# We optimize thanks to Optimization library
optf = OptimizationFunction((x, p) -> loss(x), Optimization.AutoZygote())
optprob = OptimizationProblem(optf, ps)

res = Optimization.solve(
    optprob,
    ADAM(1e-3),
    maxiters = 20
)

ps = res.u

# Then we test on the validation set and see results
ẐVal = predict(ps, probValTemplate, tVal)
@assert size(ẐVal) == size(Zval)

valError = norm(ẐVal .- Zval) / norm(Zval)
println("Relative validation error = ", valError)

errMode = vec(norm.(eachrow(ẐVal .- Zval)) ./ norm.(eachrow(Zval)))

println("Per-mode relative error (first 10 modes):")
for i in 1:min(10, r)
    println("  Mode $i : ", errMode[i])
end

ẐTrain = predict(ps, probTrainTemplate, tTrain)

trainError = norm(ẐTrain .- Ztrain) / norm(Ztrain)
println("Train relative error (Tshort) = ", trainError)
