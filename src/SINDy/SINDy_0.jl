# SINDy -> STLSQ MANUEL

using NPZ
using LinearAlgebra
using Statistics
using ModelingToolkit
using Printf

data = npzread("data/processed/sstReducedState2COPERNICUS20102019.npz")

Z  = Float64.(data["Z"]) # (time, state)
dZ = Float64.(data["dZ"])

Z  = permutedims(Z) # (state, time)
dZ = permutedims(dZ)

nState, T = size(Z)

# We build library 0
polyDegree = 2

@variables x[1:nState]
basis = polynomial_basis(x, polyDegree)

ΘFun = ModelingToolkit.build_function(
    basis, x, expression=Val(false)
)[1]

Θ = zeros(Float64, T, length(basis))
for t in 1:T
    Θ[t, :] .= ΘFun(Z[:, t])
end

# STLSQ IMPLEMENTATION
function stlsq(Θ, y; λ=1e-3, nIter=10)
    ξ = Θ \ y                # least squares
    for _ in 1:nIter
        small = abs.(ξ) .< λ
        ξ[small] .= 0.0
        big = .!small
        ξ[big] = Θ[:, big] \ y
    end
    ξ
end

# then we run SINDy
λ = 1e-3
Ξ = zeros(Float64, length(basis), nState)

for i in 1:nState
    println("Learning equation $i / $nState")
    Ξ[:, i] = stlsq(Θ, dZ[i, :]; λ=λ)
end

# We compute RMSE to compare
dZPred = (Θ * Ξ)'# (state, time)

rmseTotal = sqrt(mean((dZPred .- dZ).^2))
println("\nRMSE total = ", rmseTotal)

# DISPLAY FIRST EQUATIONS

println("\nDiscovered equations:")
for i in 1:min(5, nState)
    println("\ndx$i/dt =")
    for j in findall(!iszero, Ξ[:, i])
        println("  + $(Ξ[j,i]) * $(basis[j])")
    end
end
