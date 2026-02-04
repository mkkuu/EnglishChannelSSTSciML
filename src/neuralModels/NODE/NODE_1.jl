versioninfo()
using Pkg
Pkg.instantiate()
Pkg.precompile()

using NPZ, LinearAlgebra, Statistics
using DifferentialEquations, SciMLSensitivity
using Lux, Optimisers, ComponentArrays, Zygote
using Random
using Optimization
using OptimizationOptimisers

reducedState = npzread("data/processed/sstReducedStateCOPERNICUS20102019Prepared.npz")
# println("Available keys in reducedState:", keys(reducedState))

PCsTrainRaw = Float32.(reducedState["PCsTrain"])
tTrainRaw   = Float32.(reducedState["tTrain"])
PCsValRaw   = Float32.(reducedState["PCsVal"])
tValRaw     = Float32.(reducedState["tVal"])
stdData     = Float32.(reducedState["std"])

# println("\nSize of PCsTrain_raw: ", size(PCsTrain_raw))
# println("Size of tTrain_raw: ", size(tTrain_raw))
# println("Size of PCsVal_raw: ", size(PCsVal_raw))
# println("Size of tVal_raw: ", size(tVal_raw))

if size(PCsTrainRaw, 1) == 150 && size(PCsTrainRaw, 2) != 150
    PCsTrainFull = PCsTrainRaw'
    println("Données transposées")
else
    PCsTrainFull = PCsTrainRaw
end

if length(tTrainRaw) == size(PCsTrainFull, 1)
    tTrain = Float32.(0:size(PCsTrainFull, 2)-1)
else
    tTrain = tTrain_raw
end

# println("\nDimensions finales:")
# println("PCsTrain_full: ", size(PCsTrain_full), " (modes × temps)")
# println("tTrain: ", length(tTrain), " pas de temps")

splitIdx = Int(floor(0.8 * size(PCsTrainFull, 2)))
PCsTrain = PCsTrainFull[:, 1:splitIdx]
PCsVal = PCsTrainFull[:, splitIdx+1:end]

tTrainSplit = tTrain[1:splitIdx]
tValSplit = tTrain[splitIdx+1:end]

z0 = PCsTrain[:, 1]
tspan = (tTrain[1], tTrain[end])
r = size(PCsTrain, 1)

nn = Chain(
    Dense(r, 8, tanh),
    Dense(8, r)
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, nn)
ps = ComponentArray(ps)

function f!(dz, z, p, t)
    y, _ = nn(z, p, st)
    @assert length(y) == length(dz) "Dimension mismatch: length(y)=$(length(y)), length(dz)=$(length(dz))"
    dz .= y
end

dz = similar(z0)
y, _ = nn(z0, ps, st)

@show length(z0)
@show length(dz)
@show length(y)

f!(dz, z0, ps, tTrain[1])

prob = ODEProblem(f!, z0, tspan, ps)