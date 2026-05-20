using Plots
cd(@__DIR__)

include("util.jl")

r_vec = 0.001:0.01:10
q = 0.1
n = -2
k = 0.35

W, dW, V, V_over_r, dV_plus_V_over_r, r⁻¹, r⁻², ωₑ = get_khorrami_arrays(r_vec, q, n, k)
dV_rdr = @. (dV_plus_V_over_r-2V_over_r)/r_vec

Φ = ωₑ
dΦdr = @. k*dW + n*dV_rdr
Vort = @. r_vec*dΦdr/(k^2*r_vec^2+n^2)

plot(r_vec, Φ, label="Φ")
plot!(r_vec, Vort, label="dΦ/dr")
plot!(xlimit=(0,5))