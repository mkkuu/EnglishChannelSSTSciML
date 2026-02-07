# final v1 SINDy

using NPZ
using LinearAlgebra
using Statistics
using ModelingToolkit
using OrdinaryDiffEq
using Printf
using Statistics
using Plots

data = npzread("data/processed/sstReducedState2COPERNICUS20102019.npz")

Z  = Float64.(data["Z"]) #(time, state)
dZ = Float64.(data["dZ"])
split = Int.(data["split"])

Z  = permutedims(Z) #(state, time)
dZ = permutedims(dZ)

nState, T = size(Z)

trainIdx = findall(split .== 0)
testIdx  = findall(split .== 1)

ZTrain  = Z[:, trainIdx]
dZTrain = dZ[:, trainIdx]

ZTest   = Z[:, testIdx]
dZTest  = dZ[:, testIdx]

# we normalize (as state-wise)

μ = mean(ZTrain; dims=2)
σ = std(ZTrain; dims=2) .+ 1e-8

ZTrainN = (ZTrain .- μ) ./ σ
ZTestN  = (ZTest  .- μ) ./ σ

dZTrainN = dZTrain ./ σ
dZTestN  = dZTest  ./ σ

# We declare symbolic variables
@variables x[1:nState]

# We build the local polynomial lib
basis = Num[]
push!(basis, 1)

# linear
for i in 1:nState
    push!(basis, x[i])
end

# quadratic local
for i in 1:nState
    push!(basis, x[i]^2)
    if i < nState
        push!(basis, x[i] * x[i+1])
    end
end

nbasis = length(basis)
println("Reduced library size = ", nbasis)

# we build Θ function

ΘFun = ModelingToolkit.build_function(
    basis, x;
    expression = Val(false)
)[1]

# Then we build Θ matrix

function buildΘ(Zmat, ΘFun, nbasis)
    Tloc = size(Zmat, 2)
    Θ = zeros(Float64, Tloc, nbasis)
    for t in 1:Tloc
        Θ[t, :] .= ΘFun(Zmat[:, t])
    end
    return Θ
end

ΘTrain = buildΘ(ZTrainN, ΘFun, nbasis)
ΘTest  = buildΘ(ZTestN,  ΘFun, nbasis)

# STLSQ (MANUAL)
function stlsq(Θ, y; λ=0.1, nIter=10)
    ξ = Θ \ y
    for _ in 1:nIter
        small = abs.(ξ) .< λ
        ξ[small] .= 0.0
        big = .!small
        if any(big)
            ξ[big] = Θ[:, big] \ y
        end
    end
    return ξ
end

# λ Sweep
λList = [0.05, 0.1, 0.2, 0.5]
results = Dict{Float64,Any}()

for λ in λList
    Ξ = zeros(Float64, nbasis, nState)

    for i in 1:nState
        Ξ[:, i] = stlsq(ΘTrain, dZTrainN[i, :]; λ=λ)
    end

    dZPred = (ΘTrain * Ξ)'

    rmseVal = sqrt(mean((dZPred .- dZTrainN).^2))
    sparsity = 100 * count(abs.(Ξ) .< 1e-12) / length(Ξ)

    results[λ] = (Ξ=Ξ, rmse=rmseVal, sparsity=sparsity)

    @printf "λ = %.2f | RMSE = %.4f | Sparsity = %.1f%%\n" λ rmseVal sparsity
end

# we seek for the best λ
valid = filter(kv -> kv[2].sparsity < 99.9, results)
bestλ = argmin(k -> valid[k].rmse, keys(valid))
best = valid[bestλ]

println("\nBEST λ = ", bestλ)
println("RMSE = ", best.rmse)

Ξ = best.Ξ

# Then we display equations to observe shape
println("\nDiscovered equations (first 5 states):")

for i in 1:min(5, nState)
    println("\ndx$i/dt =")
    for j in findall(abs.(Ξ[:, i]) .> 1e-8)
        println(@sprintf("  %+0.4f * %s", Ξ[j,i], basis[j]))
    end
end

# We print RMSE
dZPredTest = (ΘTest * Ξ)'
rmseTest = sqrt(mean((dZPredTest .- dZTestN).^2))
println("\nTEST RMSE = ", rmseTest)

# ODE rollout validation
function sindyRhs!(du, u, p, t)
    du .= (ΘFun(u)' * Ξ)'[:]
end

u0 = ZTestN[:, 1]
tspan = (0.0, size(ZTestN,2)-1)

prob = ODEProblem(sindy_rhs!, u0, tspan)
sol = solve(prob, Tsit5(), saveat=1.0)

ZRollout = hcat(sol.u...)
rolloutRmse = sqrt(mean((ZRollout .- ZTestN[:,1:size(ZRollout,2)]).^2))

println("\nROLL-OUT RMSE = ", rolloutRmse)

# Stabilized SINDy A solution

# Dissipation parameter
γ = 0.25 # we can try by adjusting (0.05–0.3)

# Stabilized RHS

function sindyRhsStable!(du, u, p, t)
    Θu = ΘFun(u) # library evaluation
    du .= (Θu' * Ξ)'[:] # SINDy prediction
    du .-= γ .* u # global dissipation
end

# validation with rollout
u0 = ZTestN[:, 1]
tspan = (0.0, length(testIdx)-1)

prob = ODEProblem(
    sindyRhsStable!,
    u0,
    tspan
)

sol = solve(
    prob,
    Tsit5(),
    saveat = 1.0,
    abstol = 1e-8,
    reltol = 1e-8
)

ZRollout = hcat(sol.u...)

Tcmp = min(size(ZRollout,2), size(ZTestN,2))
rolloutRmse = sqrt(mean(
    (ZRollout[:,1:Tcmp] .- ZTestN[:,1:Tcmp]).^2
))

println("\nSTABILIZED ROLL-OUT RMSE = ", rolloutRmse)

# 1st plot
gr()
modesToPlot = 1:5
t = 1:size(ZRollout, 2)

p = plot(layout=(length(modesToPlot), 1), size=(900, 900))

for (k, i) in enumerate(modesToPlot)
    plot!(
        p[k],
        t, ZTestN[i, 1:length(t)],
        label="True",
        lw=2
    )
    plot!(
        p[k],
        t, ZRollout[i, :],
        label="SINDy rollout",
        lw=2,
        ls=:dash
    )
    ylabel!(p[k], "x$i")
end

xlabel!(p[end], "time")
plot!(p, title="SINDy stabilized roll-out vs truth")
display(p)

# 2nd plot
pairs = [(1,2), (1,3), (2,3)]

p = plot(layout=(1,3), size=(1200,400))

for (k, (i,j)) in enumerate(pairs)
    scatter!(
        p[k],
        ZTestN[i, :], ZTestN[j, :],
        ms=1,
        alpha=0.4,
        label="True"
    )
    plot!(
        p[k],
        ZRollout[i, :], ZRollout[j, :],
        lw=2,
        label="SINDy"
    )
    xlabel!(p[k], "x$i")
    ylabel!(p[k], "x$j")
end

plot!(p, title="Phase portraits: true vs SINDy")
display(p)

#3rd plot
energyTrue = var(ZTestN; dims=2)[:]
energySindy = var(ZRollout; dims=2)[:]

p = plot(
    energyTrue,
    label="True",
    lw=3,
    yscale=:log10
)

plot!(
    p,
    energySindy,
    label="SINDy rollout",
    lw=3,
    ls=:dash
)

xlabel!("Mode index")
ylabel!("Variance (log)")
title!("Energy spectrum: true vs SINDy")
display(p)

