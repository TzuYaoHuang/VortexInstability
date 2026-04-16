using LinearAlgebra, Plots

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
    V_over_r = zeros(N); dV_plus_V_over_r = zeros(N)
    inv_r = zeros(N); inv_r2 = zeros(N)
    
    for i in 1:N
        r = r_vec[i]
        E = exp(-r^2)
        
        W[i] = E
        dW[i] = -2 * r * E
        
        if abs(r) < 1e-8
            V_over_r[i] = q
            dV_plus_V_over_r[i] = 2.0 * q
            inv_r[i] = 0.0  
            inv_r2[i] = 0.0 
        else
            V_over_r[i] = q * (-expm1(-r^2)) / r^2
            dV_plus_V_over_r[i] = 2.0 * q * E
            inv_r[i] = 1.0 / r
            inv_r2[i] = 1.0 / (r^2)
        end
    end
    
    Ω_b = α .* W .+ n .* V_over_r
    return W, dW, V_over_r, dV_plus_V_over_r, inv_r, inv_r2, Ω_b
end

function solve_khorrami_qvortex_inviscid(α, n, q; N=100, L=20.0, Re=10000.0)
    ξ, D_ξ = cheb_diff(N)

    # set up of r ξ transformation
    a = 10
    b = 1+2a/L

    r = @.  a * (1 +ξ) ./(b -ξ)
    drdξ = @. a*(b+1)/(b-ξ)^2
    D_r = D_ξ ./ drdξ
    
    W, dW, V_over_r, dV_plus_V_over_r, inv_r, inv_r2, Ω_b = get_khorrami_arrays(r, q, n, α)
    imΩ_b = im*Ω_b
    
    I_mat = I(N+1)
    imI = im*I_mat
    Z = zeros(N+1, N+1)
    
    D2_r = D_r^2
    L_n = D2_r + diagm(inv_r) * D_r - diagm(n^2 .* inv_r2) - (α^2) * I_mat
    
    # --- RIGOROUSLY AUDITED 4x4 BLOCKS ---
    
    # 1. r-momentum (Eq 12 in typical texts)
    A_FF = diagm(imΩ_b)
    A_FG = diagm(-2V_over_r)
    A_FH = Z
    A_FP = D_r
    
    # 2. θ-momentum (Eq 13 in typical texts)
    A_GF = diagm(dV_plus_V_over_r)
    A_GG = diagm(imΩ_b)
    A_GH = Z
    A_GP = diagm(im * n .* inv_r)
    
    # 3. z-momentum (Eq 14 in typical texts)
    A_HF = diagm(dW)
    A_HG = Z
    A_HH = diagm(imΩ_b)
    A_HP = α * I_mat 
    
    # 4. Continuity (Eq 15 in typical texts)
    A_CF = D_r + diagm(inv_r)
    A_CG = diagm(1im * n .* inv_r)
    A_CH = 1im * α * I_mat
    A_CP = Z

    A = [A_FF A_FG A_FH A_FP;
         A_GF A_GG A_GH A_GP;
         A_HF A_HG A_HH A_HP;
         A_CF A_CG A_CH A_CP]
         
    B = [-imI Z Z Z;
         Z -imI Z Z;
         Z Z -imI Z;
         Z Z Z Z]
         
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
        A[core_F, core_F] = 1.0; A[core_F, core_G] = +1im * n 
        A[core_H, core_H] = 1.0  
        A[core_P, core_P] = 1.0  
        
        A[core_G, idx_F] = 2.0 .* D_r[N+1, :]
        A[core_G, idx_G] = -(1im * n) .* D_r[N+1, :]
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
    
    sort_idx = sortperm(imag.(vals)) 
    
    best_val = vals[sort_idx[1]]
    best_vec = vecs[:, sort_idx[1]]
    
    F_mode = best_vec[idx_F]
    G_mode = best_vec[idx_G]
    H_mode = best_vec[idx_H]
    P_mode = best_vec[idx_P]
    
    return r, best_val, F_mode, G_mode, H_mode, P_mode, vals, vecs, sort_idx
end

# --- RUN AND PLOT ---
# Parameters
α_test = 0.5
q_test = 0.8
n_test = -1 # Try m=0, m=1, m=2!

r_grid, best_val, F_mode, G_mode, H_mode, P_mode, all_sigmas, all_vecs, sort_idx = solve_khorrami_qvortex_inviscid(α_test, n_test, q_test, N=101, L=200, Re=100)
# r_grid = real.(r_comp)

println("Most Unstable Eigenvalue (σ) = ", best_val)

# Plotting
p1 = plot(r_grid, [abs.(F_mode) abs.(G_mode) abs.(H_mode)]./maximum(abs.(F_mode)), 
              labels=["|u_r|" "|u_θ|" "|u_z|"], 
              lw=2, 
              title="Full 3D Eigenfunction (m=$n_test)",
              xlabel="Radius (r)", 
              xlims=(0, 5))

p2 = scatter(real.(all_sigmas), imag.(all_sigmas), 
             title="Eigenvalue Spectrum", 
             xlabel="Re(σ)", ylabel="Im(σ)", 
             marker=:circle, label=false, markersize=3)

plot(p1, p2, layout=(2,1), size=(800, 800))