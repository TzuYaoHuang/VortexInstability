using Plots, DelimitedFiles
cd(@__DIR__)

include("util.jl")

# --- RUN AND PLOT ---
# Flow parameters
q_test = -0.8
rv_test = 1.12
λρ = 0.1

# Perturbation parameters
α_test = 0.35
n_test = 8

# gridding
N1g = 100
N2g = 100


r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx, D_r₁, D_r₂ = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ, N1=N1g, N2=N2g)

println("Most Unstable Eigenvalue (σ) = ", best_val)
println("ζ₀ = ", ζ₀)

Dᵣ = zeros((N1g+N2g+2,N1g+N2g+2))
Dᵣ[1:N1g+1,1:N1g+1] .= D_r₁
Dᵣ[N1g+2:end,N1g+2:end] .= D_r₂

ωz = (Dᵣ*(r_grid.*G_mode) .- im*n_test*F_mode) ./ r_grid
ωz[1] =0


# Define polar grid
r = r_grid[r_grid .< 2.1]
# r[1] = r[2]/10
# r[N1g+2] = (9r[N1g+2]+r[N1g+3])/10 
θ = range(0, 2π, length=360)

x = [ri * cos(θj) for ri in r, θj in θ]
y = [ri * sin(θj) for ri in r, θj in θ]

z = [real(ωz[iri]*exp(im*n_test*θj)) for (iri,ri) in enumerate(r), θj in θ]

heatmap(θ, r, z; projection = :polar, color = :PRGn)
plot!(
    grid     = false,
    ticks    = nothing,
    showaxis = false,
    legend   = false,
    colorbar = false,
    size = (600,600)
)
hline!([1.12], lw=4)

savefig("PerturbationShape_q$(q_test)_n$(n_test).png")