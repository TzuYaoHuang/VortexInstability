using GLMakie
using OrdinaryDiffEq
using LinearAlgebra

include("util.jl")

# --- RUN AND PLOT ---
# Flow parameters
q_test = 0.1
rv_test = 1.12
λρ = 1

# Perturbation parameters
α_test = 0.35
n_test = -3

# gridding
N1g = 30
N2g = 30


r_grid, best_val, F_mode, G_mode, H_mode, P_mode, ζ₀, all_sigmas, all_vecs, sort_idx = solve_inviscid_multiphase_qvortex(α_test, n_test, q_test, rv_test, 1, λρ, N1=N1g, N2=N2g)

println("Most Unstable Eigenvalue (σ) = ", best_val)
println("ζ₀ = ", ζ₀)

# Global activation for Screen Space Ambient Occlusion (SSAO)
# Set to true to enable contact shadows between overlapping helical structures
GLMakie.activate!(ssao=true)

# --- Parameters ---
const q = q_test       # Swirl strength
const W = 1.0          # Axial velocity scale
const α = α_test       # Axial wavenumber
const n = n_test       # Azimuthal wavenumber
const σ = real(best_val)+0.3im*imag(best_val)     # Complex frequency (σ_r + i*σ_i)
const ε = 0.4          # Initial perturbation amplitude
const R0 = rv_test     # Base interface radius
const time_span = 50.0 # Total animation time

# --- Physics Functions ---
function base_flow(u, p, t)
    x, y, z = u
    r_sq = x^2 + y^2
    r = sqrt(r_sq) + 1e-6
    
    uz = W * exp(-r_sq)
    uθ = (q / r) * (1 - exp(-r_sq))
    
    vx = -uθ * (y / r)
    vy =  uθ * (x / r)
    vz =  uz
    return [vx, vy, vz]
end

# --- Visualization Setup ---
fig = Figure(size = (600, 1000), backgroundcolor = :white)

# Initialize Axis3 without the invalid 'scenekw' argument
ax = Axis3(fig[1, 1], 
           title = "Batchelor Vortex: Growing Helical Instability (n=2 mode)",
           titlecolor = :white,
           aspect = :data, 
           azimuth = 1.15π, elevation = 0.2π,
           perspectiveness = 0.5, 
           protrusions = 100)

# Correct Method: Direct assignment to the internal scene lights vector
# 1. API Fix: Lighting is set on the scene, not the plot object
set_ambient_light!(ax.scene, RGBf(0.3, 0.3, 0.3))
set_lights!(ax.scene, [DirectionalLight(RGBf(1, 1, 1), Vec3f(1, 1, -1))])

# Style axes for dark background
# ax.xgridcolor = :gray30
# ax.ygridcolor = :gray30
# ax.zgridcolor = :gray30
# ax.xticklabelcolor = :white
# ax.yticklabelcolor = :white
# ax.zticklabelcolor = :white
# ax.xlabelcolor = :white
# ax.ylabelcolor = :white
# ax.zlabelcolor = :white
hidedecorations!(ax)
hidespines!(ax)

# Time as an observable for animation
t_obs = Observable(0.0)

# Grid for the interface
θ_grid = range(0, 2π, length=80)
z_grid = range(0, 2π/α, length=80)

# Compute coordinates and the perturbation magnitude simultaneously
surface_data = lift(t_obs) do t
    X = zeros(length(θ_grid), length(z_grid))
    Y = zeros(length(θ_grid), length(z_grid))
    Z = zeros(length(θ_grid), length(z_grid))
    Perturbation_Color = zeros(length(θ_grid), length(z_grid))
    
    growth = exp(imag(σ)*t)
    freq_shift = real(σ)*t
    
    for (i, θ) in enumerate(θ_grid)
        for (j, z) in enumerate(z_grid)
            argument = α*z + n*θ - freq_shift
            p_mag = real(exp(im * argument))
            
            Perturbation_Color[i,j] = p_mag 
            R = R0 + ε * growth * p_mag
            
            X[i,j] = R * cos(θ)
            Y[i,j] = R * sin(θ)
            Z[i,j] = z
        end
    end
    return X, Y, Z, Perturbation_Color
end

xs = @lift($surface_data[1])
ys = @lift($surface_data[2])
zs = @lift($surface_data[3])
cols = @lift($surface_data[4])

cmap = :PRGn

# Colorbar(fig[1, 2], label="Perturbation Peak/Trough", labelcolor=:white, 
#          ticklabelcolor=:white, colormap=cmap, colorrange=(-1, 1))

surface!(ax, xs, ys, zs, 
         color = cols, 
         colorrange = (-1, 1), 
         colormap = cmap,
         alpha = 0.9, 
         transparency = true, 
         
         # Use FastShading or MultiLightShading for modern Makie
         shading = true, 
         
         # Material properties (valid for surface!)
         diffuse = Vec3f(1.0, 1.0, 1.0), 
         specular = Vec3f(1.5, 1.5, 1.5), 
         shininess = 64.0f0)

# --- Particle Tracing (Base Flow) ---
n_particles = 8
seeds_inner = [ [0.5R0*cos(θ), 0.5R0*sin(θ), 0.0] for θ in range(0, 2π, length=n_particles) ]
seeds_outer = [ [1.5R0*cos(θ), 1.5R0*sin(θ), 0.0] for θ in range(0, 2π, length=n_particles) ]
seeds = vcat(seeds_inner, seeds_outer)
tspan_trace = (0.0, 30*2π/α)

for s in seeds
    prob = ODEProblem(base_flow, s, tspan_trace)
    sol = solve(prob, Tsit5(), reltol=1e-7) 
    ptrace = stack(sol.u)'
    lines!(ax, ptrace[:, 1], ptrace[:, 2], ptrace[:, 3], color = :darkorange2, linewidth = 1.75, alpha=1)
end

xlims!(ax, -2,2); ylims!(ax, -2,2); zlims!(ax, 0, 2π/α)

time_text = lift(t_obs) do t
    "Time: $(round(t, digits=1))"
end
text!(ax, 1.5, 1.5, 6π/α + 0.2; text=time_text, color=:white, align=(:center, :bottom), fontsize=24)

# --- Animation and Saving Loop ---
framerate = 30
timestamps = range(0, time_span, step=0.1)

println("Rendering and saving video...")

# This block generates and saves the MP4 file
record(fig, "vortex_instability.mp4", timestamps; framerate = framerate) do t
    t_obs[] = t
end

println("Video saved successfully as 'vortex_instability.mp4'.")

display(fig)