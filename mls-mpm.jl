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
    B::Array{Float64, 3} # APIC Matrix
    F::Array{Float64, 3} #deformation gradient
    # weird types below because their sizes vary
    w::Vector{Vector{Float64}} # weights (N_i)
    w_grad::Vector{Matrix{Float64}} #gradient of w
    grid_inds::Vector{Vector{CartesianIndex}} # grid nodes that are neighbors
end


function combine_particles(a::Particles, b::Particles)
    Particles(
        lcat(a.mass, b.mass),
        lcat(a.volume, b.volume),
        lcat(a.position, b.position),
        lcat(a.velocity, b.velocity),
        lcat(a.B, b.B),
        lcat(a.F, b.F),
        lcat(a.w, b.w),
        lcat(a.w_grad, b.w_grad),
        lcat(a.grid_inds, b.grid_inds)
    )
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
        zeros(D, D, n),
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
        o = x * (xabs * 1.5 - 2.)
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
    #particles.w_grad = [particle_grad_weights_from_grid_inds(particles.position[:,i], particles.grid_inds[i], grid) for i=1:plen]
end

function p2g!(particles::Particles, grid::Grid, dt::Float64)
    plen = size(particles.mass,1)
    h = grid.h
    grid.mass .= 0
    grid.momentum .= 0
    grid.velocity .= 0
    generate_weights!(particles, grid)
    D_pinvdt = 3.0 / grid.h.^2 .* dt
    for i = 1:plen
        grid_inds = particles.grid_inds[i]
        grid_pos = grid_subset(grid,grid_inds)
        N_i = particles.w[i]
        part_mass = particles.mass[i]
        part_vel = particles.velocity[:,i]
        dx = grid_pos .- particles.position[:,i]
        B_p = particles.B[:,:,i]
        F = particles.F[:,:,i]
        J = det(F)
        # eq from S6 of MLS-MPM paper replaces 173 from course
        # Do I need det(F) here?
        C = -neohookean(particles.F[:,:,i], 1e5, 0.3) .* D_pinvdt .* particles.volume[i]
        C += B_p .* part_mass
        # end S6
        for j=1:size(N_i,1)
            N_ij = N_i[j]
            m = part_mass*N_ij
            @inbounds grid.mass[grid_inds[j]] += m
            inner = (part_mass .* part_vel + C * dx[:,j])
            grid.momentum[:,grid_inds[j]] += N_ij * inner
        end

    end
    for i in CartesianIndices(grid.mass)
        if (grid.mass[i] > 0)
            grid.velocity[:, i] = grid.momentum[:, i] ./ grid.mass[i]
        end
    end
    if (any(isnan.(grid.velocity)))
        println("nans")
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
    cauchy =  P * F' ./ J
    cauchy
end

function apply_boundary!(grid::Grid)
    # set boundaries to zero for now

    grid.momentum[1,[1, end],:] = -grid.momentum[1,[1, end],:]
    grid.momentum[2,:,[1, end]] = -grid.momentum[2,:,[1, end]]
    grid.velocity[1,[1, end],:] = -grid.velocity[1,[1, end],:]
    grid.velocity[2,:,[1, end]] = -grid.velocity[2,:,[1, end]]
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
    for ind in CartesianIndices(grid.mass)
        if(grid.mass[ind] == 0.)
            continue
        end
        m = grid.mass[ind]
        f = grid_forces[:,ind]
        dv_dt = f ./ m .* dt
        # dv_dt[end] += -dt # gravity
        grid.velocity[:,ind] += dv_dt
    end
    apply_boundary!(grid)
end

function update_deformation_gradient!(grid::Grid, particles::Particles)
    plen = size(particles.mass,1)
    for i=1:plen
        inner = Matrix(I, D, D) .+ particles.B[:,:,i]
        particles.F[:,:,i] = inner * particles.F[:,:,i]
    end
end

function update_particle_vel!(grid::Grid, particles::Particles)
    plen = size(particles.mass, 1)
    for i = 1:plen
        grid_inds = particles.grid_inds[i]
        w = particles.w[i]
        g_vels = grid_subset_vel(grid, grid_inds)
        v_p = sum(g_vels * w,dims=2)
        particles.velocity[:,i] = v_p
        if (any(isnan.(v_p)))
            println(grid_inds)
            println(g_vels)
            error("nan")
        end
    end
end

# Returns D x 2 matrix where columns are min/max corners
function grid_extents(grid::Grid)::Matrix{Float64}
    hcat(grid.position[:,1,1], grid.position[:,end,end])
end

function update_particle_pos!(grid, particles, dt)
    particles.position += particles.velocity .* dt;
    # bound position by extents of grid
    ext = grid_extents(grid)
    plen = size(particles.mass, 1)
    for i=1:plen
        pos = particles.position[:,i]
        particles.position[:,i] = max.(min.(pos, ext[:,end]), ext[:, 1])
    end
end

function update_particle_B!(grid::Grid, particles::Particles)
    plen = size(particles.mass, 1)
    D_pinv = 3.0 / grid.h.^2
    for i=1:plen
        pos = particles.position[:,i]
        grid_inds = particles.grid_inds[i]
        w = particles.w[i]
        g_pos = grid_subset(grid, grid_inds)
        g_vels = grid_subset_vel(grid, grid_inds)
        dx = g_pos .- pos
        for i=1:D
            dx[i,:] = w .* dx[i,:]
        end
        dx = dx .* D_pinv
        B_p = g_vels * dx'
    end
end

function g2p!(grid::Grid, particles::Particles, dt::Float64)
    plen = size(particles.mass,1)
    D_pinv = ones(D, D) .* 3.0 / grid.h.^2
    update_particle_B!(grid, particles)
    update_deformation_gradient!(grid, particles)
    update_particle_vel!(grid, particles)
    update_particle_pos!(grid, particles, dt)
end

function calculate_forces(particles, grid, dt::Float64)::Array{Float64, D+1}
    grid_forces = zeros(size(grid.momentum))
    #grid_forces[end,:,:] = grid.mass .* -1.
    # Get the stress based force
    # C is {{2, 2}, N} array here
    c = neohookean.(eachslice(particles.F, dims=3), 1e2, 0.5)
    # Length N
    J = det.(eachslice(particles.F, dims=3))
    c = c .* particles.volume .* J .* dt
    B_m = particles.B .* paricles.mass
    for i=1:size(particles.mass,1)
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
    p2g!(particles, grid, dt)
    #grid_forces = calculate_forces(particles, grid, dt)
    #apply_grid_forces!(grid, grid_forces, dt)
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

function lcat(a, b)
    cat(a, b, dims=length(size(a)))
end


N = 20
dt = 0.1
box_mass = 100.0
particles = get_box_particles(box_mass / N^D, N^D, [0.5, 0.1], [0.2, 0.2])
p1 = get_box_particles(box_mass / N^D, N^D, [0.35, 0.39], [0.2, 0.2])
p1.velocity[2,:] .+= -0.01
particles = combine_particles(particles, p1)
#particles.velocity[2,:] .+= -2.
grid = generate_grid(0.0, 1.0, 41)
p2g!(particles, grid, dt)
pvol!(particles, grid)
anim = @gif for i=1:1000
# for i = 1:100
    plot_sim(particles,grid)
    if(i%5 == 0)
        println(i)
    end
    @time timestep(particles, grid, dt)
end every 5
