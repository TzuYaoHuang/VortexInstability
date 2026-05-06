using LinearAlgebra

function cheb_diff(N)
    x = cos.(pi * (0:N) / N)
    c = [2; ones(N-1); 2] .* (-1).^(0:N)
    X = repeat(x, 1, N+1)
    dX = X - X'
    D = (c * (1 ./ c)') ./ (dX + I) 
    D -= diagm(sum(D, dims=2)[:])
    return x, D
end

function get_khorrami_arrays(r_vec, q, n, α)
    N = length(r_vec)
    W = zeros(N); dW = zeros(N)
    V = zeros(N);
    V_over_r = zeros(N); dV_plus_V_over_r = zeros(N)
    r⁻¹ = zeros(N); r⁻² = zeros(N)
    
    for i in 1:N
        r = r_vec[i]
        E = exp(-r^2)
        
        W[i] = E
        dW[i] = -2 * r * E
        
        if abs(r) < 1e-8
            V[i] = q*r
            V_over_r[i] = q
            dV_plus_V_over_r[i] = 2.0 * q
            r⁻¹[i] = 0.0  
            r⁻²[i] = 0.0 
        else
            V[i] = q * (1 - E) / r
            V_over_r[i] = q * (-expm1(-r^2)) / r^2
            dV_plus_V_over_r[i] = 2.0 * q * E
            r⁻¹[i] = 1.0 / r
            r⁻²[i] = 1.0 / (r^2)
        end
    end
    ωₑ = α .* W .+ n .* V_over_r
    return W, dW, V, V_over_r, dV_plus_V_over_r, r⁻¹, r⁻², ωₑ
end

function solve_khorrami_qvortex(α, n, q; N=100, L=1000, halfgridL=3, Re=Inf)
    ξ, D_ξ = cheb_diff(N)

    # set up of r ξ transformation
    a = halfgridL # Half of the grid is within r<a
    b = 1+2a/L

    r = @.  a * (1 +ξ)/(b -ξ)
    drdξ = @. a*(b+1)/(b-ξ)^2
    D_r = D_ξ ./ drdξ
    
    W, dW, V, V_over_r, dV_plus_V_over_r, r⁻¹, r⁻², ωₑ = get_khorrami_arrays(r, q, n, α)
    imωₑ = im*ωₑ
    
    I_mat = I(N+1)
    imI = im*I_mat
    Z = zeros(N+1, N+1)
    
    D2_r = D_r^2
    L_n = D2_r + diagm(r⁻¹) * D_r - diagm(n^2 .* r⁻²) - (α^2) * I_mat
    neginvRe = -inv(Re)
    
    # --- RIGOROUSLY AUDITED 4x4 BLOCKS ---
    # (uᵣ,uₜ, uz, p) = (R(r), Θ(r), Z(r), P(r)) exp(i(αz+nθ-σt))

    # 1. r-momentum (Eq 12 in typical texts)
    A_FF = diagm(imωₑ) .+ neginvRe * (L_n .- diagm(r⁻²))
    A_FG = diagm(-2V_over_r) .+ neginvRe*diagm(-2im*n*r⁻²)
    A_FH = Z
    A_FP = D_r
    
    # 2. θ-momentum (Eq 13 in typical texts)
    A_GF = diagm(dV_plus_V_over_r) .+ neginvRe*diagm(2im*n*r⁻²)
    A_GG = diagm(imωₑ) .+ neginvRe * (L_n .- diagm(r⁻²))
    A_GH = Z
    A_GP = diagm(im * n .* r⁻¹)
    
    # 3. z-momentum (Eq 14 in typical texts)
    A_HF = diagm(dW)
    A_HG = Z
    A_HH = diagm(imωₑ) .+ neginvRe * L_n
    A_HP = α * imI 
    
    # 4. Continuity (Eq 15 in typical texts)
    A_CF = D_r + diagm(r⁻¹)
    A_CG = diagm(1im * n .* r⁻¹)
    A_CH = α * imI
    A_CP = Z

    A = [A_FF A_FG A_FH A_FP;
         A_GF A_GG A_GH A_GP;
         A_HF A_HG A_HH A_HP;
         A_CF A_CG A_CH A_CP]
         
    B = [imI Z   Z   Z;
         Z   imI Z   Z;
         Z   Z   imI Z;
         Z   Z   Z   Z]
         
    # --- BOUNDARY CONDITIONS ---
    idx_F = 1:N+1;           idx_G = (N+2):(2N+2)
    idx_H = (2N+3):(3N+3);   idx_P = (3N+4):(4N+4)
    
    far_F = 1;      core_F = N+1
    far_G = N+2;    core_G = 2N+2
    far_H = 2N+3;   core_H = 3N+3
    far_P = 3N+4;   core_P = 4N+4

    # 1. Far-field (r=L)
    for row in [far_F, far_G, far_H, far_P]
        A[row, :] .= 0; A[row, row] = 1.0; B[row, :] .= 0
    end

    # 2. Centerline (r=0) compatibility relations
    for row in [core_F, core_G, core_H, core_P]
        A[row, :] .= 0; B[row, :] .= 0
    end

    if n == 0
        A[core_F, core_F] = 1.0  
        A[core_G, core_G] = 1.0  
        A[core_H, idx_H] = D_r[N+1, :] 
        A[core_P, idx_P] = D_r[N+1, :] 
    elseif abs(n) == 1
        A[core_F, core_F] = 1.0; A[core_F, core_G] = 1im * n 
        A[core_H, core_H] = 1.0  
        A[core_P, core_P] = 1.0  
        
        A[core_G, idx_F] = 2.0 .* D_r[N+1, :]
        A[core_G, idx_G] = (1im * n) .* D_r[N+1, :]
    else 
        A[core_F, core_F] = 1.0  
        A[core_G, core_G] = 1.0  
        A[core_H, core_H] = 1.0  
        A[core_P, core_P] = 1.0  
    end

    vals, vecs = eigen(A, B)
    
    valid_idx = isfinite.(vals) .& (abs.(vals) .< 50.0)
    vals = vals[valid_idx]
    vecs = vecs[:, valid_idx]
    
    sort_idx = sortperm(imag.(vals),rev=true) 
    
    best_val = vals[sort_idx[1]]
    best_vec = vecs[:, sort_idx[1]]
    scalefac = maximum(abs,best_vec[1:3N+3])
    
    F_mode = best_vec[idx_F]./scalefac
    G_mode = best_vec[idx_G]./scalefac
    H_mode = best_vec[idx_H]./scalefac
    P_mode = best_vec[idx_P]./scalefac

    enconFreq = @. (n*V_over_r + α*W - best_val)
    
    return r, best_val, F_mode, G_mode, H_mode, P_mode, vals, vecs, sort_idx, enconFreq
end

function build_inviscid_domain_blocks(r, D_r, q, α, n)
    W, dW, V, V_over_r, dV_plus_V_over_r, r⁻¹, r⁻², ωₑ = get_khorrami_arrays(r, q, n, α)

    N_pts = length(r)
    I_mat = I(N_pts)
    imωₑ = im*ωₑ
    imI = im*I_mat
    Z = zeros(N_pts,N_pts)

    # --- LHS Matrix (A) ---
    # 1. r-momentum (Eq 12 in typical texts)
    A_FF = diagm(imωₑ)
    A_FG = diagm(-2V_over_r)
    A_FH = Z
    A_FP = D_r
    
    # 2. θ-momentum (Eq 13 in typical texts)
    A_GF = diagm(dV_plus_V_over_r)
    A_GG = diagm(imωₑ)
    A_GH = Z
    A_GP = diagm(im * n .* r⁻¹)
    
    # 3. z-momentum (Eq 14 in typical texts)
    A_HF = diagm(dW)
    A_HG = Z
    A_HH = diagm(imωₑ)
    A_HP = α * imI 
    
    # 4. Continuity (Eq 15 in typical texts)
    A_CF = D_r + diagm(r⁻¹)
    A_CG = diagm(1im * n .* r⁻¹)
    A_CH = α * imI
    A_CP = Z
        
    A = [A_FF A_FG A_FH A_FP;
         A_GF A_GG A_GH A_GP;
         A_HF A_HG A_HH A_HP;
         A_CF A_CG A_CH A_CP]
    B = [imI Z   Z   Z;
         Z   imI Z   Z;
         Z   Z   imI Z;
         Z   Z   Z   Z]

    return A, B, V, W, ωₑ
end

# 3. Multiphase Solver with Domain Stitching
function solve_inviscid_multiphase_qvortex(α, n, q, rᵥ, rho1, rho2; N1=50, N2=50, L=1000.0, halfgridL=3)
    # --- Domain 1 Mapping [0, rᵥ] ---
    ξ₁, D_ξ₁ = cheb_diff(N1)
    r₁ = (rᵥ / 2) .* (1 .- ξ₁)
    D_r₁ = D_ξ₁ .* (-2.0 / rᵥ)
    # --- Domain 2 Mapping [rᵥ, L] ---
    ξ₂, D_ξ₂ = cheb_diff(N2)
    # set up of r ξ transformation
    a = halfgridL # Half of the grid is within r<a
    b = 1+2a/(L-rᵥ)
    r₂ = @. rᵥ + a * (1 - ξ₂) ./(b + ξ₂)
    dr₂dξ₂ = @. -a*(b+1)/(b+ξ₂)^2
    D_r₂ = D_ξ₂ ./ dr₂dξ₂
    
    # Build interior physics
    A1, B1, V1_arr, W1_arr, ωₑ1_arr = build_inviscid_domain_blocks(r₁, D_r₁, q, α, n)
    A2, B2, V2_arr, W2_arr, ωₑ2_arr = build_inviscid_domain_blocks(r₂, D_r₂, q, α, n)
    
    S1 = N1 + 1
    S2 = N2 + 1
    
    # Allocate Global Matrices
    A = zeros(ComplexF64, 4*S1 + 4*S2, 4*S1 + 4*S2)
    B = zeros(ComplexF64, 4*S1 + 4*S2, 4*S1 + 4*S2)
    
    # Embed Domain 1
    A[1:4*S1, 1:4*S1] .= A1
    B[1:4*S1, 1:4*S1] .= B1
    
    # Embed Domain 2
    offset = 4*S1
    A[offset+1:end, offset+1:end] .= A2
    B[offset+1:end, offset+1:end] .= B2
    
    # --- Mapping Key Row Indices ---
    idx_F1 = 1:S1;             idx_G1 = S1+1:2S1
    idx_H1 = 2S1+1:3S1;        idx_P1 = 3S1+1:4S1
    
    idx_F2 = offset+1:offset+S2;         idx_G2 = offset+S2+1:offset+2S2
    idx_H2 = offset+2S2+1:offset+3S2;    idx_P2 = offset+3S2+1:offset+4S2
    
    core_F1 = 1;               core_G1 = S1+1
    core_H1 = 2S1+1;           core_P1 = 3S1+1
    
    intf_F1 = S1;              intf_P1 = 4S1
    intf_F2 = offset + 1;      intf_P2 = offset + 3S2 + 1
    far_F2 = offset + S2
    
    # --- APPLY BOUNDARY CONDITIONS ---
    
    # 1. Centerline (r=0) Compatibility limits
    for row in [core_F1, core_G1, core_H1, core_P1]
        A[row, :] .= 0; B[row, :] .= 0
    end
    
    if n == 0
        A[core_F1, core_F1] = 1.0  
        A[core_G1, core_G1] = 1.0  
        A[core_H1, idx_H1] .= D_r₁[1, :] 
        A[core_P1, idx_P1] .= D_r₁[1, :] 
    elseif abs(n) == 1
        A[core_F1, core_F1] = 1.0; A[core_F1, core_G1] = 1im * n 
        A[core_H1, core_H1] = 1.0  
        A[core_P1, core_P1] = 1.0  
        A[core_G1, idx_F1] .= 2.0 .* D_r₁[1, :]
        A[core_G1, idx_G1] .= (1im * n) .* D_r₁[1, :]
    else 
        A[core_F1, core_F1] = 1.0  
        A[core_G1, core_G1] = 1.0  
        A[core_H1, core_H1] = 1.0  
        A[core_P1, core_P1] = 1.0  
    end
    
    # 2. Far-field (r=L) Kinematic Condition (Inviscid)
    A[far_F2, :] .= 0; A[far_F2, far_F2] = 1.0; B[far_F2, :] .= 0
    
    # 3. INTERFACE STITCHING (r = rᵥ) FOR DISCONTINUOUS BASE FLOW
    A[intf_F1, :] .= 0; B[intf_F1, :] .= 0
    A[intf_F2, :] .= 0; B[intf_F2, :] .= 0
    
    # Get discontinuous base flow values directly from the arrays
    V1_intf = V1_arr[end]
    W1_intf = W1_arr[end] # Make sure to return W from your array builder
    
    V2_intf = V2_arr[1]
    W2_intf = W2_arr[1]
    
    omega_e1 = n * (V1_intf / rᵥ) + α * W1_intf
    omega_e2 = n * (V2_intf / rᵥ) + α * W2_intf
    
    # --- Kinematic Match: ω_e2 * R1 - ω_e1 * R2 = σ * (R1 - R2) ---
    A[intf_F1, intf_F1] = omega_e2
    A[intf_F1, intf_F2] = -omega_e1
    
    B[intf_F1, intf_F1] = 1.0
    B[intf_F1, intf_F2] = -1.0
    
    # --- Dynamic Pressure Match (Eliminating ζ_0 via R1) ---
    # i*ω_e1*(ρ1*Π1 - ρ2*Π2) + R1*(ρ1*V1^2 - ρ2*V2^2)/rᵥ = i*σ*(ρ1*Π1 - ρ2*Π2)
    
    A[intf_F2, intf_P1] = 1im * omega_e1 * rho1
    A[intf_F2, intf_P2] = -1im * omega_e1 * rho2
    A[intf_F2, intf_F1] = (rho1 * V1_intf^2 - rho2 * V2_intf^2) / rᵥ
    
    B[intf_F2, intf_P1] = 1im * rho1
    B[intf_F2, intf_P2] = -1im * rho2
    
    # --- EIGEN SOLVE ---
    vals, vecs = eigen(A, B)
    
    # Filter physical modes
    valid_idx = isfinite.(vals) .& (abs.(vals) .< 50.0)
    vals = vals[valid_idx]
    vecs = vecs[:, valid_idx]
    
    sort_idx = sortperm(imag.(vals), rev=true) 
    
    best_val = vals[sort_idx[1]]
    best_vec = vecs[:, sort_idx[1]]
    
    # Stitch the full physical domain together (Duplicate point at rᵥ is retained)
    r_global = vcat(r₁, r₂)
    F_mode = vcat(best_vec[idx_F1], best_vec[idx_F2])
    G_mode = vcat(best_vec[idx_G1], best_vec[idx_G2])
    H_mode = vcat(best_vec[idx_H1], best_vec[idx_H2])
    P_mode = vcat(best_vec[idx_P1], best_vec[idx_P2])

    scalefac = max(maximum(abs, F_mode), maximum(abs, G_mode), maximum(abs, H_mode))
    
    F_mode ./= scalefac
    G_mode ./= scalefac
    H_mode ./= scalefac
    P_mode ./= scalefac
    
    ζ₀F1 = F_mode[N1+1]/(im*(ωₑ1_arr[end]-best_val))
    ζ₀F2 = F_mode[N1+2]/(im*(ωₑ2_arr[1]-best_val))
    ζ₀ = (ζ₀F1+ζ₀F2)/2

    return r_global, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, vals, vecs, sort_idx
end

function getMultiphaseMode(best_vec, N1, N2)
    S1 = N1 + 1
    S2 = N2 + 1
    offset = 4*S1
    
    # --- Mapping Key Row Indices ---
    idx_F1 = 1:S1;             idx_G1 = S1+1:2S1
    idx_H1 = 2S1+1:3S1;        idx_P1 = 3S1+1:4S1
    
    idx_F2 = offset+1:offset+S2;         idx_G2 = offset+S2+1:offset+2S2
    idx_H2 = offset+2S2+1:offset+3S2;    idx_P2 = offset+3S2+1:offset+4S2

    F_mode = vcat(best_vec[idx_F1], best_vec[idx_F2])
    G_mode = vcat(best_vec[idx_G1], best_vec[idx_G2])
    H_mode = vcat(best_vec[idx_H1], best_vec[idx_H2])
    P_mode = vcat(best_vec[idx_P1], best_vec[idx_P2])

    scalefac = max(maximum(abs, F_mode), maximum(abs, G_mode), maximum(abs, H_mode))
    
    F_mode ./= scalefac
    G_mode ./= scalefac
    H_mode ./= scalefac
    P_mode ./= scalefac

    return F_mode, G_mode, H_mode, P_mode
end