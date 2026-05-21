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


qList = -1.5:0.1:1.5
λρ_test = 0.1
α_test = 0.35
nList = 0:3
rv_test = 1.12 #1.12

σiMat = zeros((length(qList), length(nList)))
σrMat = zeros((length(qList), length(nList)))

for im∈eachindex(nList), iq∈eachindex(qList)
    n_test = nList[im]
    q_test = qList[iq]
    r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ_test, N1=100, N2=50)
    σiMat[iq, im] = imag(best_val)
    σrMat[iq, im] = real(best_val)
end

# Plotting
p1 = heatmap(nList, qList, σiMat, xlimit=(-0.5,3.5), ylimit=(-1.5,1.5), color=:tempo, xlabel=L"n", ylabel=L"q", colorbar_title="Growth rate", clim=(0,0.6)) #linear_worb_100_25_c53_n256
hline!(p1, [0], color=:black, label="", ls=:dash, lw=1.5)
# p2 = heatmap(qList, nList, σrMat', xlimit=(0,1.5), ylimit=(-3,3), color=:PRGn_5, xlabel="q", ylabel="n", colorbar_title="Phase speed",clim=(-π/2,π/2))

# plot(p1, p2, layout=(2,1), size=(650, 900), title="Density ratio (out/in) = $(λρ_test)")

plot!(p1, xmirror = true, size=(350, 700))
savefig(p1, "qn_l$(λρ_test).png")