using LinearAlgebra, Plots

# 1. Standard Chebyshev Differentiation Matrix
function cheb_diff(N)
    x = cos.(pi * (0:N) / N)
    c = [2; ones(N-1); 2] .* (-1).^(0:N)
    X = repeat(x, 1, N+1)
    dX = X - X'
    D = (c * (1 ./ c)') ./ (dX + I) 
    D -= diagm(sum(D, dims=2)[:])
    return x, D
end

# 2. L'Hôpital-Safe Base Flow Functions for the q-Vortex
function q_vortex_base_flow(r_vec, q, m, k)
    N = length(r_vec)
    W = zeros(N); dW = zeros(N)
    V = zeros(N); dV = zeros(N)
    V_over_r = zeros(N)
    SQ_over_r = zeros(N)
    S = zeros(N)
    
    for i in 1:N
        ri = r_vec[i]
        
        # Axial Flow
        W[i]  = exp(-ri^2)
        dW[i] = -2 * ri * exp(-ri^2)
        
        # Swirl Flow and S(r)
        S[i]  = ri^2 / (m^2 + k^2 * ri^2)
        
        if abs(ri) < 1e-8
            # L'Hôpital Exact Limits at r -> 0
            V[i] = 0.0
            dV[i] = q
            V_over_r[i] = q
            if m==0
                S[i] = 1/k^2
            end
            
            # The exact limit of S*Q/r at the origin
            SQ_over_r[i] = (m == 0) ? -2.0/k : 2.0*q/m
        else
            # Normal Evaluation
            V[i] = (q / ri) * (1 - exp(-ri^2))
            dV[i] = (q / ri^2) * (2 * ri^2 * exp(-ri^2) - 1 + exp(-ri^2))
            V_over_r[i] = V[i] / ri
            
            # Q = kW' + (m/r)(V' + V/r)
            Qi = k * dW[i] + (m / ri) * (dV[i] + V_over_r[i])
            SQ_over_r[i] = S[i] * Qi / ri
        end
    end
    
    return W, dW, V, dV, V_over_r, S, SQ_over_r
end

# 3. The Main QEP Solver
function solve_q_vortex(k, m, q; N=150, L=20.0, ε=1e-6)
    # Generate Grid (Shifted to avoid r=0 exactly)
    ξ, D_ξ = cheb_diff(N)
    
    # Map from ξ ∈ [1, -1] to r ∈ [L, ε]
    r = L*ξ #(L - ε)/2 .* ξ .+ (L + ε)/2
    dr_dξ = L # (L - ε)/2
    r = (L - ε)/2 .* ξ .+ (L + ε)/2
    dr_dξ =  (L - ε)/2
    D_r = D_ξ ./ dr_dξ
    
    # Get Safe Base Flow Profiles
    W, dW, V, dV, V_over_r, S, SQ_over_r = q_vortex_base_flow(r, q, m, k)
    
    # Local Doppler Frequency: ω(r) = m(V/r) + kW
    ω = m .* V_over_r .+ k .* W
    
    # Differential Operator Matrix: L = D_r * diag(S/r) * D_r
    L_op = D_r * diagm(S ./ r) * D_r
    
    # The H(r) term
    d_SQ_r = D_r * SQ_over_r
    H = r .* d_SQ_r
    
    # The Centrifugal/Shear C(r) term
    C = (2 .* k .* V .* S ./ r.^2) .* (k .* r .* (dV .+ V_over_r) .- m .* dW)
    
    # Build the Quadratic Matrices (M2 * σ^2 + M1 * σ + M0) * G = 0
    R_mat = diagm(r)
    Ω_mat = diagm(ω)
    H_mat = diagm(H)
    C_mat = diagm(C)
    I_mat = I(N+1)
    
    M2 = R_mat * L_op - I_mat
    M1 = 2 .* R_mat * Ω_mat * L_op - (2 .* Ω_mat + H_mat)
    M0 = R_mat * (Ω_mat^2) * L_op - (Ω_mat^2 + Ω_mat * H_mat - C_mat)
    
    # Boundary Conditions: G(L) = 0 (Index 1) and G(ε) = 0 (Index N+1)
    for idx in [1, N+1]
        M2[idx, :] .= 0; M2[idx, idx] = 1.0  # σ^2 * G_boundary = 0
        M1[idx, :] .= 0
        M0[idx, :] .= 0
    end
    
    # Linearize into a 2N x 2N Generalized Eigenvalue Problem
    Zero_mat = zeros(N+1, N+1)
    
    A_GEP = [Zero_mat  I_mat;
            -M0       -M1]
            
    B_GEP = [I_mat     Zero_mat;
             Zero_mat  M2]
             
    # Solve the system
    vals, vecs = eigen(A_GEP, B_GEP)
    
    # Filter out spurious infinite/NaN eigenvalues
    valid_idx = isfinite.(vals)
    vals = vals[valid_idx]
    vecs = vecs[:, valid_idx]
    
    # Find the most unstable mode (Maximum negative imaginary part)
    sort_idx = sortperm(imag.(vals), rev=false) 
    
    best_val = vals[sort_idx[1]]
    best_vec = vecs[1:N+1, sort_idx[1]] # Extract the G vector (top half)
    
    return r, best_val, best_vec, vals, vecs, sort_idx, D_r
end

# --- RUN AND PLOT ---
# Parameters
k_test = 0.5
q_test = 0.1
m_test = 0 # Try changing this to 1 or 2!

r_grid, sigma, G_mode, all_sigmas, all_vecs, idx, D_r = solve_q_vortex(k_test, m_test, q_test, N=152, L=20.0, ε=1e-6)

println("Most Unstable Eigenvalue (σ) = ", sigma)

# Convert G back to Radial Velocity Amplitude: R(r) = G(r)/r
dG_dr = D_r * G_mode

# 2. Reconstruct R_amp safely
R_amp = zeros(ComplexF64, length(r_grid))

for i in 1:length(r_grid)
    if r_grid[i] < 0.0001 # For the noisy points near the origin
        R_amp[i] = dG_dr[i] 
    else
        R_amp[i] = G_mode[i] / r_grid[i]
    end
end

# Plot the Eigenfunction Shape (Magnitude of R)
plot(r_grid, abs.(G_mode), 
    lw=2, color=:black, 
    label="|u_r(r)|", 
    title="Radial Velocity Mode (m=$m_test, q=$q_test)",
    xlabel="Radius (r)", ylabel="Amplitude",
    xlims=(0,5))

scatter(real.(all_sigmas), imag.(all_sigmas), 
    title="Eigenvalue Spectrum (k=$k_test, q=$q_test, m=$m_test)", 
    xlabel="Re(σ) = -ck", ylabel="Im(σ)=-growth rate", 
    label="Modes", marker=:circle)

plot(r_grid, abs.(R_amp), 
    lw=2, color=:black, 
    label="|u_r(r)|", 
    title="Radial Velocity Mode (m=$m_test, q=$q_test)",
    xlabel="Radius (r)", ylabel="Amplitude",
    xlims=(0, 5))