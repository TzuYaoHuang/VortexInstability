using Plots, DelimitedFiles
gr()
using Plots.PlotMeasures
using LaTeXStrings


default()
Plots.scalefontsizes()
default(fontfamily="Arial",linewidth=2, framestyle=:axes, label=nothing, grid=false, tick_dir=:out, size=(900,600),right_margin=3mm,left_margin=5mm,top_margin=0mm,bottom_margin=1mm,markerstrokewidth=0,markersize=8,dpi=600, guidefont = (20, :black), tickfont = (12, :black), colorbar_titlefontsize = 20)
# Plots.scalefontsizes(3.1)
cd(@__DIR__)

include("util.jl")


qList = -1.5:0.1:0
ρList = 10. .^ (-2:0.5:0)
log₁₀ρ = log10.(ρList)
α_test = 1.05
n_test = 4
rv_test = 1.12 #1.12

σiMat = zeros((length(qList), length(ρList)))
σrMat = zeros((length(qList), length(ρList)))

for iq∈eachindex(qList), iρ∈eachindex(ρList)
    q_test = qList[iq]
    λρ_test = ρList[iρ]
    r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx, _ = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ_test, N1=50, N2=50)
    σiMat[iq, iρ] = imag(best_val)
    σrMat[iq, iρ] = real(best_val)
end

p1 = heatmap(log₁₀ρ, qList, σiMat, color=:tempo, clim=(0,0.6))
plot!(p1, ylimit=(-1.5,0), xlimit=(-2,0), xlabel=L"\log10(\lambda_\rho)", ylabel=L"q", colorbar_title="Growth rate")
plot!(p1, xmirror = true, size=(700, 600))
savefig(p1, "qrho_$(n_test)_k$(αList).png")
display(p1)
