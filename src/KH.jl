using LinearAlgebra
using Plots

include("util.jl")

N = 1024
k = 0.5 # need to be <1 to be unstable.
L_domain = 20.0 # Mapping from xi [-1, 1] to z [-10, 10]

# 1. Base Flow
U_func(z) = tanh(z)
D2U_func(z) = -2 * tanh(z) * sech(z)^2

# 2. Differentiation Matrices
ξ, D_mat = cheb_diff(N) # Assuming your cheb_diff function is defined
z = ξ .* L_domain
Dz = D_mat ./ L_domain
D2z = Dz^2

# 3. Operators
# Your derivation: (σ + kU)(D²/k² - 1)Z - (D²U/k)Z = 0
# Let B_op = -(D2z/k^2 - I) 
# Then: σ*B*Z = -k*U*B*Z - (D2U/k)*Z
B = -(D2z ./ k^2 - I)
UMat = diagm(U_func.(z))
D2UMat = diagm(D2U_func.(z))

A = -k .* UMat * B .- D2UMat ./ k

# 4. Corrected Boundary Conditions
for i in (1, N+1)
    A[i, :] .= 0.0
    B[i, :] .= 0.0
    A[i, i] = 1.0 
    # B[i,i] remains 0, pushing boundary eigenvalues to infinity
end

# 5. Solve
vals, vecs = eigen(A, B)

# 6. Filter out Infinite/NaN eigenvalues from BCs
# We want the most unstable mode (min imag for e^{i*sigma*t})
mask = .!isnan.(vals) .& .!isinf.(vals)
valid_vals = vals[mask]
valid_vecs = vecs[:, mask]

growth_rates = imag.(valid_vals)
idx_unstable = argmin(growth_rates)
# idx_unstable = argmin(abs.(real.(valid_vals)))
sigma_best = valid_vals[idx_unstable]
u_mode = valid_vecs[:, idx_unstable]

println("Eigenvalue: ", sigma_best)
println("Growth Rate: ", -imag(sigma_best))

# 7. Plotting
u_mode ./= u_mode[N÷2+1] # shift with middle phase. exp(i*phase)
maxReal = max(abs.(real.(u_mode))...)
p = plot(size=(500,700))
plot!(p, U_func.(z), z,color=:red, label="U(z) = tanh(z)", lw=2)
plot!(p, abs.(u_mode) ./ maxReal, z,color=:black, label="|Z|", lw=2)
plot!(p, real.(u_mode)./maxReal, z,color=:blue,label="real")
plot!(p, imag.(u_mode)./maxReal, z,color=:green,label="imaginary")
# plot!(sech.(z), z, label="Analytical (sech)", ls=:dash, color=:black)
plot!(xlabel="Z(z), U(z)",ylabel="z")
display(p)