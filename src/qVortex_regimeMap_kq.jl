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


qList = -1.5:0.05:0
λρ_test = .1
αList = 0:0.05:2
n_test = 1
rv_test = 1.12 #1.12

σiMat = zeros((length(qList), length(αList)))
σrMat = zeros((length(qList), length(αList)))

for ik∈eachindex(αList), iq∈eachindex(qList)
    α_test = αList[ik]
    q_test = qList[iq]
    r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx, _ = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ_test, N1=50, N2=50)
    σiMat[iq, ik] = imag(best_val)
    σrMat[iq, ik] = real(best_val)
end

p1 = heatmap(αList, qList, σiMat, ylimit=(-1.5,0), xlimit=(0,2), color=:tempo, xlabel=L"k", ylabel=L"q", colorbar_title="Growth rate", clim=(0,0.6))

plot!(p1, xmirror = true, size=(700, 600))
savefig(p1, "qk_$(n_test)_l$(λρ_test).png")
display(p1)