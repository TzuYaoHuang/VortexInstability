using Plots, DelimitedFiles
cd(@__DIR__)

include("util.jl")

# --- RUN AND PLOT ---
# Flow parameters
q_test = 0.85
rv_test = 1.12
λρ = 2

# Perturbation parameters
α_test = 0.35
n_test = -1

# gridding
N1g = 100
N2g = 100


r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ, N1=N1g, N2=N2g)

println("Most Unstable Eigenvalue (σ) = ", best_val)
println("ζ₀ = ", ζ₀)

# Plotting
p1 = plot(r_grid, [abs.(F_mode) abs.(G_mode) abs.(H_mode)], 
              labels=["|R|" "|Θ|" "|Z|" "ωₑ"], 
              lw=2, 
              title="q=$q_test, rv=$rv_test, λρ=$λρ, α=$α_test, n=$n_test, N1,2=$N1g,$N2g",
              xlabel="Radius (r)", 
              xlims=(0, 5))
hline!(p1,[0],ls=:dash,c=:gray,label="")

p2 = scatter(real.(all_sigmas), imag.(all_sigmas), 
             title="σ ($best_val)", 
             xlabel="Re(σ)", ylabel="Im(σ)", 
             marker=:circle, label=false, markersize=3)

plot(p1, p2, layout=(2,1), size=(650, 800))

