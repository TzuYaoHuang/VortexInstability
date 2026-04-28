using Plots, DelimitedFiles
cd(@__DIR__)

include("util.jl")

# --- RUN AND PLOT ---
# Parameters
α_test = 0.3
q_test = 0.3
n_test = -1
rv_test = 2
λρ = 2
N1g = 100
N2g = 100

r_grid, best_val, F_mode, G_mode, H_mode, P_mode, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ, N1=N1g, N2=N2g)

println("Most Unstable Eigenvalue (σ) = ", best_val)

# Plotting
p1 = plot(r_grid, [abs.(F_mode) abs.(G_mode) abs.(H_mode)], 
              labels=["|R|" "|Θ|" "|Z|" "ωₑ"], 
              lw=2, 
              title="α=$α_test, q=$q_test, n=$n_test, rv=$rv_test, N1g=$N1g, N2g=$N2g",
              xlabel="Radius (r)", 
              xlims=(0, 5))
hline!(p1,[0],ls=:dash,c=:gray,label="")

p2 = scatter(real.(all_sigmas), imag.(all_sigmas), 
             title="σ ($best_val)", 
             xlabel="Re(σ)", ylabel="Im(σ)", 
             marker=:circle, label=false, markersize=3)

plot(p1, p2, layout=(2,1), size=(650, 800))


# TEST WITH KHORRAMI (Re=141.4, α=1.34, n=-2)
# qList = 0.2:0.1:1.2
# ciList = zero(qList)
# for (iq,q) ∈ enumerate(qList)
#     r_grid, best_val, F_mode, G_mode, H_mode, P_mode, all_sigmas, all_vecs, sort_idx, enconFreq = solve_khorrami_qvortex(1.34, -2, q, N=200, Re=141.4, L=100)
#     ciList[iq] = imag(best_val)
# end

# Khorrami_data = readdlm("../Dataset/Khorrami1989_JCP_viscousStability.csv", ',')

# p3 = plot(qList, ciList, lw=2, label="Calculated")
# plot!(p3, Khorrami_data[:,1], Khorrami_data[:,2], lw=2, label="Khorrami 1989")
