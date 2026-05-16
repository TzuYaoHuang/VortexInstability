using Plots, DelimitedFiles
cd(@__DIR__)

include("util.jl")

# qList = -1.5:0.15:1.5
# ρList = 10 .^ (-1:0.2:1)
# log₁₀ρ = log10.(ρList)
# α_test = 0.35
# n_test = -1
# rv_test = 1.12 #1.12

# σMat = zeros((length(qList), length(ρList)))

# for iq∈eachindex(qList), iρ∈eachindex(ρList)
#     q_test = qList[iq]
#     λρ_test = ρList[iρ]
#     r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ_test, N1=50, N2=50)
#     σMat[iq, iρ] = imag(best_val)
# end

# plot(size=(700,500))
# contourf!(qList, log₁₀ρ, σMat', xlimit=(-1.5,1.5), ylimit=(-1,1), aspect_ratio=:equal, color=:PuBuGn)
# plot!(xlabel="q", ylabel="log10(ρout/ρin)", title="α=$α_test, n=$n_test, rv=$rv_test")

# q_test = 0.
# λρ_test = 1
# # log₁₀ρ = log10.(ρList)
# αList = 0:0.1:2
# nList = -1:1
# rv_test = 1.12 #1.12

# σMat = zeros((length(αList), length(nList)))

# for ik∈eachindex(αList), im∈eachindex(nList)
#     α_test = αList[ik]
#     n_test = nList[im]
#     r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ_test, N1=50, N2=50)
#     σMat[ik, im] = imag(best_val)
# end

# plot(size=(700,500))
# contourf!(αList, nList, σMat', levels=5, xlimit=(0,2.0), ylimit=(-1,1), color=:PuBuGn)
# plot!(xlabel="α", ylabel="n", title="q=$q_test, λρ=$λρ_test, rv=$rv_test")


qList = -1.5:0.05:0
λρ_test = .1
# log₁₀ρ = log10.(ρList)
αList = 0:0.05:2
n_test = 1
rv_test = 1.12 #1.12

σiMat = zeros((length(qList), length(αList)))
σrMat = zeros((length(qList), length(αList)))

for ik∈eachindex(αList), iq∈eachindex(qList)
    α_test = αList[ik]
    q_test = qList[iq]
    r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ_test, N1=50, N2=50)
    σiMat[iq, ik] = imag(best_val)
    σrMat[iq, ik] = real(best_val)
end

plot(size=(700,500))
heatmap!(qList, αList, σiMat', levels=0:0.01:0.24, xlimit=(-1.5,0), ylimit=(0,2), color=:PuBuGn)
plot!(xlabel="q", ylabel="α", title="n=$n_test, λρ=$λρ_test, rv=$rv_test")

