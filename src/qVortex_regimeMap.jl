using Plots, DelimitedFiles
cd(@__DIR__)

include("util.jl")

qList = -1.5:0.15:1.5
ρList = 10 .^ (-1:0.1:1)
log₁₀ρ = log10.(ρList)
α_test = 1.4
n_test = -1
rv_test = 1 #1.12

σMat = zeros((length(qList), length(ρList)))

for iq∈eachindex(qList), iρ∈eachindex(ρList)
    q_test = qList[iq]
    λρ_test = ρList[iρ]
    r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ_test, N1=50, N2=50)
    σMat[iq, iρ] = imag(best_val)
end

plot(size=(700,500))
contourf!(qList, log₁₀ρ, σMat', xlimit=(-1.5,1.5), ylimit=(-1,1), aspect_ratio=:equal, color=:PuBuGn)
plot!(xlabel="q", ylabel="log10(ρout/ρin)", title="α=$α_test, n=$n_test, rv=$rv_test")

