using LinearAlgebra
using Plots

const D = 2

VectorF = Vector{Float64}
PosArray = Array{Float64, 2}
GridArr = Array{Float64, D+1}

mutable struct Grid
    mass::Array{Float64,D}
    momentum::Array{Float64, D+1}
    position::Array{Float64, D+1}
    velocity::Array{Float64, D+1}
    h::Float64
end

mutable struct Particles
    mass::Vector{Float64}
    volume::Array{Float64}
    position::Array{Float64, 2}
    velocity::Array{Float64, 2}
    F::Array{Float64, 3} #deformation gradient
    # weird types below because their sizes vary
    w::Vector{Vector{Float64}} # weights (N_i)
    w_grad::Vector{Matrix{Float64}} #gradient of w
    grid_inds::Vector{Vector{CartesianIndex}} # grid nodes that are neighbors
end

function generate_grid(start, stop, len::Int)
    # Kind of complicated way to generate multidimensional points
    lenD = ((len for i=1:D)...,)
    lenDp1 = (D,lenD...)
    ranges = (LinRange(start, stop, l) for l in lenD)
    arr = [Iterators.flatten([[x...] for x in Iterators.product(ranges...)])...]
    points = reshape(arr, lenDp1)
    g = Grid(
        zeros(Float64, lenD),
        zeros(Float64, lenDp1),
        points,
        zeros(Float64, lenDp1),
        (stop - start) / (len - 1),
    )
end

function get_box_particles(
    particle_mass::Float64,
    n::Int64,
    center::Vector{Float64} = zeros(D),
    extents::Vector{Float64} = ones(D),
)::Particles
    positions = add_box(n, center, extents)
    w_init = [zeros(Float64, 0) for i = 1:n]
    w_grad_init = [zeros(Float64, D,0) for i = 1:n]
    grid_inds_init = [CartesianIndex[] for i=1:n]
    Particles(
        ones(n) .* particle_mass,
        zeros(n),
        positions,
        zeros(D, n),
        zeros(D, D, n) .+ Matrix(I,D,D),
        w_init,
        w_grad_init,
        grid_inds_init)
end


function add_box(
    n::Int64,
    center::Vector{Float64} = zeros(D),
    extents::Vector{Float64} = ones(D),
)::Array{Float64, 2}
    #(rand(Float64, (D, n)) .- 0.5) .* extents .+ center
    n_ex = Int(floor(n.^(1. / D)))
    min_ext = center - extents ./ 2
    max_ext = center + extents ./ 2
    ranges = (LinRange(min_ext[i], max_ext[i], n_ex) for i=1:D)
    arr = [Iterators.flatten([[x...] for x in Iterators.product(ranges...)])...]
    final_len = Int(floor(size(arr,1)/D))
    points = reshape(arr, (D, final_len))
end

function kernel(x)
    xabs = abs(x)
    if (xabs < 1)
        o = 0.5 * xabs^3.0 - xabs^2 + 2.0 / 3.0
    else
        o = max(((2 - xabs)^3) / 6.0, 0.0)
    end
    o
end

function dkernel(x)
    xabs = abs(x)
    o = 0
    if(xabs < 1)
        o = 0.5 * x * (xabs * 3. - 4.)
    elseif(xabs < 2)
        o = -0.5 *(2-xabs)^2.0
        o = sign(x) * o
    end
    o
end


function kernel_indices(inds, max_inds)::Vector{CartesianIndex}
    cardinals = Iterators.product(-2:2, -2:2)
    new_inds = [CartesianIndex((inds.+x)...) for x=cardinals if all(inds.+x .>= 1) && all(inds.+x .<= max_inds)]
end

# Get the indices that match
function particle_grid_indices(pos::Array{Float64, 1}, grid::Grid)::Vector{CartesianIndex}
    inds = CartesianIndex{D}()
    h = grid.h
    indices = floor.(Int, (pos .- grid.position[:,inds]) ./ h) .+ 1
    kinds = kernel_indices(indices, size(grid.position)[2:end])
end

function grid_subset(grid::Grid, grid_inds::Vector{CartesianIndex})::Array{Float64, 2}
    @inbounds grid.position[:,grid_inds]
end

function grid_subset_vel(grid::Grid, grid_inds::Vector{CartesianIndex})::Array{Float64, 2}
    @inbounds grid.velocity[:,grid_inds]
end

function particle_weights_unreduced(pos::VectorF, grid_poses::PosArray, h::Float64, func)
    h = grid.h
    diffs = pos .- grid_poses
    diffs = diffs ./ h
    kernel_values = func.(diffs)
end

function particle_weights(pos::VectorF, grid_poses::PosArray, h::Float64)::Vector{Float64}
    kernel_values = particle_weights_unreduced(pos, grid_poses, h, kernel)
    out = prod(kernel_values, dims=1)
    vec(out)
end

function particle_grad(pos::VectorF, grid_poses::PosArray, h::Float64)::Matrix{Float64}
    weights = particle_weights_unreduced(pos, grid_poses, h, kernel)
    dweights = particle_weights_unreduced(pos, grid_poses, h, dkernel)
    out = zeros(D, size(grid_poses, 2))
    for i=1:D
        temp_w = weights
        temp_w[i,:] = dweights[i,:]
        out[i,:] = prod(temp_w,dims=1) * 1. / h
    end
    out
end

function particle_weights_from_grid_inds(pos::VectorF, grid_inds::Vector{CartesianIndex}, grid::Grid)::Vector{Float64}
    grid_poses = grid_subset(grid, grid_inds)
    particle_weights(pos, grid_poses, grid.h)
end

function particle_grad_weights_from_grid_inds(pos::VectorF, grid_inds::Vector{CartesianIndex}, grid::Grid)
    grid_poses = grid_subset(grid, grid_inds)
    out = particle_grad(pos, grid_poses, grid.h)
end

function generate_weights2!(particles::Particles, grid::Grid)
    plen = size(particles.mass,1)
    h = grid.h
    for i = 1:plen
        pos = particles.position[:,i]
        grid_inds = particle_grid_indices(pos, grid)
        particles.grid_inds[i] = grid_inds
        N_i = particle_weights_from_grid_inds(pos, grid_inds, grid)
        particles.w[i] = N_i
        p_w = particle_grad_weights_from_grid_inds(pos, grid_inds, grid)
        particles.w_grad[i] = p_w
    end
end

function generate_weights!(particles::Particles, grid::Grid)
    plen = size(particles.mass,1)
    particles.grid_inds = [particle_grid_indices(particles.position[:,i], grid) for i=1:plen]
    particles.w = [particle_weights_from_grid_inds(particles.position[:,i], particles.grid_inds[i], grid) for i=1:plen]
    particles.w_grad = [particle_grad_weights_from_grid_inds(particles.position[:,i], particles.grid_inds[i], grid) for i=1:plen]
end

function p2g!(particles::Particles, grid::Grid)
    plen = size(particles.mass,1)
    h = grid.h
    grid.mass .= 0
    grid.momentum .= 0
    grid.velocity .= 0
    generate_weights!(particles, grid)
    for i = 1:plen
        grid_inds = particles.grid_inds[i]
        N_i = particles.w[i]
        part_mass = particles.mass[i]
        part_vel = particles.velocity[:,i]
        for j=1:size(N_i,1)
            N_ij = N_i[j]
            @inbounds grid.mass[grid_inds[j]] += part_mass*N_ij
            grid.momentum[:,grid_inds[j]] += part_mass.*part_vel.*N_ij
        end
    end
    for i in CartesianIndices(grid.mass)
        if (grid.mass[i] > 0)
            grid.velocity[:, i] = grid.momentum[:, i] ./ grid.mass[i]
        end
    end
end

function neohookean(F::AbstractMatrix{Float64}, E::Float64, nu::Float64)
    mu = E / (2 * (1 + nu))
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    FF = F' * F
    F_t = inv(F')
    J = det(F)
    logJ = log(J)
    den_energy = mu*(tr(FF) - D) - mu * J + lambda / 2 * logJ ^ 2
    # P = mu .* (F - F_t) + lambda * logJ * F_t
    P = mu .* F + (lambda * logJ - mu) .* F_t
    cauchy = 1. / J .* P * F'
    cauchy
end

function apply_boundary!(grid::Grid)
    # set boundaries to zero for now

    grid.momentum[1,[1, end],:] = -grid.momentum[1,[1, end],:]
    grid.momentum[2,:,[1, end]] = -grid.momentum[2,:,[1, end]]
end

function mass_bool_to_vec_bool(mass_mask::BitArray{D})::BitArray{D+1}
    out = BitArray(undef, (D, size(mass_mask)...))
    colons = (Colon() for i=1:D)
    for i = 1:D
        out[i,colons...] = mass_mask
    end
    return out
end

function apply_grid_forces!(grid, grid_forces::Array{Float64, D+1}, dt::Float64)
    # println("max before grid_forces ",maximum(abs.(grid.velocity)))
    for ind in CartesianIndices(grid.mass)
        if(grid.mass[ind] == 0.)
            continue
        end
        m = grid.mass[ind]
        f = grid_forces[:,ind]
        dv_dt = f ./ m .* dt
        dv_dt[end] += -dt
        grid.velocity[:,ind] += dv_dt
    end
    apply_boundary!(grid)
    # println("max after grid_forces ",maximum(abs.(grid.velocity)), " " , findmax(abs.(grid.velocity)))
end

function update_deformation_gradient(F::Matrix{Float64}, w_grad::Matrix{Float64}, grid_inds::Vector{CartesianIndex}, grid::Grid, dt::Float64)
    vw_sum = zeros(D,D)
    for i=1:size(grid_inds,1)
        v = grid.velocity[:,grid_inds[i]]
        w_grad_i = w_grad[:,i]
        vw_sum += reshape(v,(D,1)) * w_grad_i'
    end
    vw_sum = vw_sum .* dt
    inner = Matrix(I, 2,2) + vw_sum
    F_new = inner * F
end

function update_particle_vel(vel::VectorF, w::VectorF,  grid_inds::Vector{CartesianIndex}, grid::Grid, alpha::Float64)
    g_vels = grid_subset_vel(grid, grid_inds)
    v_p = sum(g_vels * w,dims=2)
end

# Returns D x 2 matrix where columns are min/max corners
function grid_extents(grid::Grid)::Matrix{Float64}
    hcat(grid.position[:,1,1], grid.position[:,end,end])
end

function update_particle_pos(pos::VectorF, vel::VectorF, dt::Float64, grid::Grid)
    pos += vel .* dt
    # bound position by extents of grid
    ext = grid_extents(grid)
    new_pos = max.(min.(pos, ext[:,end]), ext[:, 1])
end

function g2p!(grid::Grid, particles::Particles, dt::Float64)
    plen = size(particles.mass,1)
    for i=1:plen
        w = particles.w[i]
        w_grad = particles.w_grad[i]
        grid_inds = particles.grid_inds[i]
        F = particles.F[:,:,i]
        pos = particles.position[:,i]
        vel = particles.velocity[:,i]
        particles.velocity[:,i] = update_particle_vel(vel, w, grid_inds, grid, .95)
        particles.F[:,:,i] = update_deformation_gradient(F, w_grad, grid_inds, grid, dt)
        particles.position[:,i] = update_particle_pos(pos, particles.velocity[:,i], dt, grid)
    end
end

function calculate_forces(particles, grid, dt::Float64)::Array{Float64, D+1}
    grid_forces = zeros(size(grid.momentum))
    grid_forces[end,:,:] = grid.mass .* -100.
    # Get the stress based force
    # C is {{2, 2}, N} array here
    c = neohookean.(eachslice(particles.F, dims=3), 1e4, 0.2)
    # Length N
    J = det.(eachslice(particles.F, dims=3))
    c = c .* particles.volume .* J
    for i=1:size(particles.mass,1)
        subC = c[i]
        w_grad = particles.w_grad[i]
        subC_w_grad = subC * w_grad
        grid_inds = particles.grid_inds[i]
        for j=1:size(grid_inds,1)
            grid_forces[:,grid_inds[j]] += subC_w_grad[:,j]
        end
    end
    grid_forces
end

function pvol!(particles::Particles, grid::Grid)
    # map densities back to particles
    plen = size(particles.mass,1)
    for i=1:plen
        grid_inds = particles.grid_inds[i]
        @inbounds masses = grid.mass[grid_inds]
        w = particles.w[i]
        rho_p = sum(w.*masses)
        particles.volume[i] = particles.mass[i] ./ rho_p .* grid.h.^D
    end
end

function timestep(particles, grid, dt::Float64)
    p2g!(particles, grid)
    grid_forces = calculate_forces(particles, grid, dt)
    apply_grid_forces!(grid, grid_forces, dt)
    g2p!(grid, particles, dt)
end

function plot_sim(particles::Particles, grid::Grid)
    min_ex = grid.position[:,1,1]
    max_ex = grid.position[:,end,end]
    x = particles.position[1,:]
    y = particles.position[2,:]
    scatter(x, y)
    xlims!((min_ex[1],max_ex[1]))
    ylims!((min_ex[2],max_ex[2]))
end

N = 30
particles = get_box_particles(1e-2, N^D, [1., 0.6])
particles.velocity[1,(N-1)*N+1:N*N] .+= 10.
particles.velocity[1,1:N] .+= -10.
grid = generate_grid(0.0, 2.0, 41)
generate_weights!(particles, grid)
p2g!(particles, grid)
pvol!(particles, grid)
anim = @gif for i=1:100
# for i = 1:100
    plot_sim(particles,grid)
    if(i%5 == 0)
        println(i)
    end
    @time timestep(particles, grid, 0.0005)
end every 10
