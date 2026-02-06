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

tTrainFull = tfull[1:splitIdx]
tVal = tfull[splitIdx+1:end]

# println("Ztrain_full size = ", size(ZtrainFull))
# println("Zval size = ", size(Zval))

# We select an horizon to train NODE on
Tshort = 400

Ztrain = ZtrainFull[:, 1:Tshort]
tTrain = tTrainFull[1:Tshort]

# We normalize
stdMode = vec(std(Ztrain; dims=2))

Ztrain .= Ztrain ./ reshape(stdMode, :, 1)
Zval .= Zval ./ reshape(stdMode, :, 1)

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


# Results :
# julia src/neuralModels/NODE/NODE_1.jl 
# PCsTrain transposed to (modes × time)
# Final Zfull size = (150, 2922) (modes × time)

# ┌ Warning: Lux.apply(m::AbstractLuxLayer, x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st) input was corrected to Lux.apply(m::AbstractLuxLayer, x::ReverseDiff.TrackedArray}, ps, st).
# │ 
# │ 1. If this was not the desired behavior overload the dispatch on `m`.
# │ 
# │ 2. This might have performance implications. Check which layer was causing this problem using `Lux.Experimental.@debug_mode`.
# └ @ ArrayInterfaceReverseDiffExt ~/.julia/packages/LuxCore/kQC9S/ext/ArrayInterfaceReverseDiffExt.jl:9

# Relative validation error = 8.950518
# Per-mode relative error (first 10 modes):
#   Mode 1 : 22.394346
#   Mode 2 : 7.443269
#   Mode 3 : 11.339866
#   Mode 4 : 3.359684
#   Mode 5 : 4.056679
#   Mode 6 : 5.1049404
#   Mode 7 : 3.0732152
#   Mode 8 : 4.8116536
#   Mode 9 : 6.005008
#   Mode 10 : 11.166377
# Train relative error (Tshort) = 8.099191