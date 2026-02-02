# lv_plot.jl

using Pkg
# À faire une fois dans ce dossier, ensuite tu peux commenter
Pkg.activate(".")
Pkg.add(["DifferentialEquations", "Plots"])

using DifferentialEquations
using Plots

println("== Début du script avec plot ==")

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, γ, δ = p

    du[1] = α * x - β * x * y
    du[2] = -γ * y + δ * x * y
end

u0 = [1.0, 1.0]
p  = (1.5, 1.0, 3.0, 1.0)
tspan = (0.0, 10.0)

prob = ODEProblem(lotka_volterra!, u0, tspan, p)
println("prob OK")

sol = solve(prob)
println("sol OK, type = ", typeof(sol))

# Tracé simple
plot(sol, title = "Rabbits vs Wolves")

println("== Script terminé (un graph devrait être affiché) ==")

p = plot(sol, title = "Rabbits vs Wolves")
savefig(p, "lotka_volterra.png")
println("Figure enregistrée dans lotka_volterra.png")
