using LinearAlgebra, Plots
import LambertW: lambertw

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

# 2. Purely Analytical Base Flow (No D_r matrix used here!)
function q_vortex_base_flow_analytical(r_vec, q, m, k)
    N = length(r_vec)
    W = zeros(ComplexF64,N); V_over_r = zeros(ComplexF64,N); S = zeros(ComplexF64,N); S_over_r = zeros(ComplexF64,N)
    H = zeros(ComplexF64,N); C = zeros(ComplexF64,N); ω = zeros(ComplexF64,N)

    W₋₁ = lambertw(-1/(2sqrt(exp(1))), -1)
    rᵥ= sqrt(-W₋₁-0.5)
    
    for i in 1:N
        r = r_vec[i]
        E = exp(-r^2)
        W[i] = E #- exp(-rᵥ^2)
        
        if abs(r) < 1e-9
            V_over_r[i] = q
            S[i] = (m == 0) ? 1.0/k^2 : 0.0
            S_over_r[i] = (m == 0) ? Inf : 0.0 # Bypassed by ε-offset anyway
            H[i] = 0.0
            C[i] = (m == 0) ? 4.0 * q^2 : 0.0
            ω[i] = m * q + k  
        else
            V_over_r[i] = q * (-expm1(-r^2)) / r^2
            S[i] = r^2 / (m^2 + k^2 * r^2)
            S_over_r[i] = r / (m^2 + k^2 * r^2) # The analytically smoothed term
            
            Q  = 2 * E * (m * q / r - k * r)
            dQ = 2 * E * (-2 * m * q + 2 * k * r^2 - m * q / r^2 - k)
            
            term1 = (r * (m^2 - k^2 * r^2)) / ((m^2 + k^2 * r^2)^2)
            H[i] = term1 * Q + S[i] * dQ
            C[i] = 4 * k * E * V_over_r[i] * S[i] * (k * q + m)
            ω[i] = m * V_over_r[i] + k * W[i]
        end
    end
    
    return W, V_over_r, S, S_over_r, H, C, ω
end

# 3. The Main QEP Solver for R(r)
function solve_q_vortex(k, m, q; N=150, L=20.0, ε=1e-5)
    # Generate Grid
    ξ, D_ξ = cheb_diff(N)
    
    # Map from ξ ∈ [1, -1] to r ∈ [L, ε]
    r = (L - ε)/2 .* ξ .+ (L + ε)/2
    dr_dξ = (L - ε)/2
    D_r = D_ξ ./ dr_dξ

    # with damping
    # r_c = 0.5  # Estimate of critical layer location
    # depth = 0.022
    # width = 0.2
    # r_real = (L - ε)/2 .* ξ .+ (L + ε)/2
    # r = r_real .- 1im .* depth .* exp.(-((r_real .- r_c)./width).^2)
    # dr_dξ = (L - ε)/2 .* (1 .+ 1im .* (2 .* depth ./ width^2) .* (r_real .- r_c) .* exp.(-((r_real .- r_c)./width).^2))
    # D_r = D_ξ ./ dr_dξ
    
    # Get Analytical Profiles
    W, V_over_r, S, S_over_r, H, C, ω = q_vortex_base_flow_analytical(r, q, m, k)
    
    # The New Distributive Operator
    # L_op = D_r * (S * D_r + S/r)
    L_op = D_r * (diagm(S) * D_r + diagm(S_over_r))
    
    # Build the Matrices (No 'r' multiplication needed)
    Ω_mat = diagm(ω)
    H_mat = diagm(H)
    C_mat = diagm(C)
    I_mat = I(N+1)
    
    M2 = L_op - I_mat
    M1 = 2 .* Ω_mat * L_op - (2 .* Ω_mat + H_mat)
    M0 = (Ω_mat^2) * L_op - (Ω_mat^2 + Ω_mat * H_mat - C_mat)
    
    # --- Physical Boundary Conditions for R(r) ---
    
    # 1. Far-field R(L) = 0 (Index 1)
    M2[1, :] .= 0; M2[1, 1] = 1.0  
    M1[1, :] .= 0
    M0[1, :] .= 0
    
    # 2. Origin Physics (Index N+1)
    if m == 1
        # Bending mode: Velocity is finite, but its slope is zero -> R'(0) = 0
        M2[N+1, :] .= 0
        M1[N+1, :] .= 0
        M0[N+1, :] .= D_r[N+1, :]  
    else
        # Axisymmetric & Fluting modes: Velocity must vanish -> R(0) = 0
        M2[N+1, :] .= 0; M2[N+1, N+1] = 1.0  
        M1[N+1, :] .= 0
        M0[N+1, :] .= 0
    end

    # Linearize into a 2N x 2N GEP (Companion Matrix)
    Zero_mat = zeros(N+1, N+1)
    
    A_GEP = [Zero_mat  I_mat;
            -M0       -M1]
            
    B_GEP = [I_mat     Zero_mat;
             Zero_mat  M2]
             
    # Solve the system
    vals, vecs = eigen(A_GEP, B_GEP)
    
    # Filter valid eigenvalues
    valid_idx = isfinite.(vals) .& (abs.(vals) .< 1e5)
    vals = vals[valid_idx]
    vecs = vecs[:, valid_idx]
    
    # Sort by maximum imaginary part (most unstable)
    sort_idx = sortperm(imag.(vals), rev=false) 
    
    best_val = vals[sort_idx[1]]
    best_vec = vecs[1:N+1, sort_idx[1]] # This is now R(r) directly!
    
    return r, best_val, best_vec, vals, L_op, D_r
end

function extract_full_mode(r_vec, R_mode, sigma, q, m, k, D_r;ε=1e-2)
    N = length(r_vec)
    Z_mode = zeros(ComplexF64, N)
    Theta_mode = zeros(ComplexF64, N)
    Pi_mode = zeros(ComplexF64, N)
    
    # Calculate the spectral derivative of R exactly once
    R_prime = D_r * R_mode
    
    for i in 1:N
        r = r_vec[i]
        E = exp(-r^2)
        
        # Base flow analytical terms
        dW = -2 * r * E
        V_sum = 2 * q * E  # This is V' + V/r
        
        if abs(r) < 1e-9
            # L'Hôpital Limits at the Origin
            V_over_r = q
            ω = m * q + k
            S = (m == 0) ? 1.0/k^2 : 0.0
            S_over_r = 0.0
            S_over_r2 = (m == 0) ? 0.0 : 1.0/m^2 # (m=0 is safe because it gets multiplied by m)
            SQ_over_r = (m == 0) ? -2.0/k : 2.0*q/m
            SQ = 0.0
        else
            # Exact Analytical Arrays
            V_over_r = q * (-expm1(-r^2)) / r^2
            ω = m * V_over_r + k * E
            
            S = r^2 / (m^2 + k^2 * r^2)
            S_over_r = r / (m^2 + k^2 * r^2)
            S_over_r2 = 1.0 / (m^2 + k^2 * r^2)
            
            SQ_over_r = 2 * E * (m * q - k * r^2) / (m^2 + k^2 * r^2)
            SQ = SQ_over_r * r
        end
        
        γ = sigma + ω
        γ_reg = γ - 1im * ε * exp(-abs2(γ) / (2 * ε^2))
        
        # 1. Exact Pressure (Pi)
        Pi_mode[i] = γ * (S * R_prime[i] + S_over_r * R_mode[i]) - SQ * R_mode[i]

        # 2. Regularized Axial Velocity (Z)
        Z_mode[i] = (1im * dW / γ_reg) * R_mode[i] - (k / γ_reg) * Pi_mode[i]
        
        # 3. Exact Azimuthal Velocity (Theta)
        if m == 0
            # For m=0, the Pi term is multiplied by m, so it vanishes safely.
            Theta_mode[i] = (1im * V_sum / γ_reg) * R_mode[i]
        else
            # For m >= 1, use the distributed Pi/r analytical formula
            Pi_over_r = γ * (S_over_r * R_prime[i] + S_over_r2 * R_mode[i]) - SQ_over_r * R_mode[i]
            Theta_mode[i] = (1im * V_sum / γ_reg) * R_mode[i] - (m / γ_reg) * Pi_over_r
        end
    end
    
    return Theta_mode, Z_mode, Pi_mode
end

# --- RUN AND PLOT ---
# Parameters
k_test = 0.1
q_test = 0.1
m_test = 1 # Try m=0, m=1, m=2!

r_comp, sigma, R_mode, all_sigmas, L_op, D_r = solve_q_vortex(k_test, m_test, q_test, N=800, L=15.0, ε=1e-9)
r_grid = real.(r_comp)

println("Most Unstable Eigenvalue (σ) = ", sigma)

# Call the function after you get R_mode and sigma
Theta_mode, Z_mode, Pi_mode = extract_full_mode(r_grid, R_mode, sigma, q_test, m_test, k_test, D_r; ε=1e-1)

# Plotting
p1 = plot(r_grid, [abs.(R_mode) abs.(Theta_mode) abs.(Z_mode)], 
              labels=["|u_r|" "|u_θ|" "|u_z|"], 
              lw=2, 
              title="Full 3D Eigenfunction (m=$m_test)",
              xlabel="Radius (r)", 
              xlims=(0, 5))

p2 = scatter(real.(all_sigmas), imag.(all_sigmas), 
             title="Eigenvalue Spectrum", 
             xlabel="Re(σ)", ylabel="Im(σ)", 
             marker=:circle, label=false, markersize=3)

plot(p1, p2, layout=(2,1), size=(800, 800))