# SINDy implementation to reduce number of terms in library -> we use: canonical base + hierarchy

using NPZ
using LinearAlgebra
using Statistics
using Printf

# basic configuration
filePath = "data/processed/sstReducedState2COPERNICUS20102019.npz"

polyDegree = 2
kDominant  = 3 # dominants mods autorised in interactions
lambdaList = [0.05, 0.1, 0.2, 0.5]
tauPost    = 0.1 # threshold post-STLSQ

data = npzread(filePath)

Z  = Float64.(data["Z"]) #(time, state)
dZ = Float64.(data["dZ"])

Z  = permutedims(Z) #(state, time)
dZ = permutedims(dZ)

nState, T = size(Z)

# we build the canonical lib Θ

basisLabels = String[]
# constant
push!(basisLabels, "1")

# linear
for i in 1:nState
    push!(basisLabels, "x[$i]")
end

# quadratic (hierarchy + canonical)
for i in 1:nState
    for j in i:min(nState, kDominant)
        push!(basisLabels, "x[$i]*x[$j]")
    end
end

nBasis = length(basisLabels)

Θ = zeros(Float64, T, nBasis)

for t in 1:T
    col = 1
    Θ[t, col] = 1.0
    col += 1

    # linear
    for i in 1:nState
        Θ[t, col] = Z[i, t]
        col += 1
    end

    # quadratic
    for i in 1:nState
        for j in i:min(nState, kDominant)
            Θ[t, col] = Z[i, t] * Z[j, t]
            col += 1
        end
    end
end

# STLSQ
function stlsq(Θ, y; λ=0.1, nIter=10)
    ξ = Θ \ y
    for _ in 1:nIter
        small = abs.(ξ) .< λ
        ξ[small] .= 0.0
        if any(.!small)
            ξ[.!small] = Θ[:, .!small] \ y
        end
    end
    ξ
end

function hardThreshold!(Ξ, τ)
    Ξ[abs.(Ξ) .< τ] .= 0.0
end

# We train and select model
bestRmse = Inf
bestΞ = nothing
bestλ = nothing

for λ in lambdaList
    Ξ = zeros(Float64, nBasis, nState)

    for i in 1:nState
        Ξ[:, i] = stlsq(Θ, dZ[i, :]; λ=λ)
    end

    hardThreshold!(Ξ, tauPost)

    dZPred = (Θ * Ξ)' #(state, time)
    rmseB = sqrt(mean((dZPred .- dZ).^2))
    sparsity = 100 * count(iszero, Ξ) / length(Ξ)

    @printf "λ = %.2f | RMSE = %.4f | Sparsity = %.1f%%\n" λ rmseB sparsity

    if rmseB < bestRmse && sparsity < 100
        bestRmse = rmseB
        bestΞ = copy(Ξ)
        bestλ = λ
    end
end

# we return results 
println("\nBEST λ = ", bestλ)
@printf "RMSE = %.4f\n" bestRmse

println("\nDiscovered equations (first 5 states):")

for i in 1:min(5, nState)
    println("\ndx$i/dt =")
    for j in findall(!iszero, bestΞ[:, i])
        @printf "  %+0.4f * %s\n" bestΞ[j, i] basisLabels[j]
    end
end
