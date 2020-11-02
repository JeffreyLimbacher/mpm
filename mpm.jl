using LinearAlgebra

D = 2
N = 81

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
    position::Array{Float64, 2}
    velocity::Array{Float64, 2}
    F::Array{Float64, 3} #deformation gradient
    # weird types below because their sizes vary
    w::Vector{Vector{Float64}} # weights (N_i)
    w_grad::Vector{Matrix{Float64}} #gradient of w
    grid_inds::Vector{Matrix{Int}} # grid nodes that are neighbors
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
    grid_inds_init = [zeros(Int,0,0) for i=1:n]
    Particles(
        ones(n) .* particle_mass,
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
    (rand(Float64, (D, n)) .- 0.5) .* extents .+ center
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
    if(xabs < 1)
        o = 3.0/2.0 * xabs^2.0
    else
        o = -0.5 *(2-xabs)^2.0
    end
    o = sign(x) * o
end


function kernel_indices(inds, max_inds)::Matrix{Int}
    cardinals = Iterators.product(-2:2, -2:2)
    new_inds = [inds.+x for x=cardinals if all(inds.+x .>= 1) && all(inds.+x .<= max_inds)]
    hcat(new_inds...)
end

# Get the indices that match
function particle_grid_indices(pos::Array{Float64, 1}, grid::Grid)::Matrix{Int}
    inds = (1 for i=1:D)
    h = grid.h
    indices = floor.(Int, (pos .- grid.position[:,inds...]) ./ h) .+ 1
    kinds = kernel_indices(indices, size(grid.position)[2:end])
end

function grid_subset(grid::Grid, grid_inds::Matrix{Int})::Array{Float64, 2}
    grid_poses = [grid.position[:,grid_inds[:,i]...] for i=1:size(grid_inds,2)]
    grid_poses = hcat(grid_poses...)
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

function particle_weights_from_grid_inds(pos::VectorF, grid_inds::Matrix{Int}, grid::Grid)::Vector{Float64}
    grid_poses = grid_subset(grid, grid_inds)
    particle_weights(pos, grid_poses, grid.h)
end

function particle_grad_weights_from_grid_inds(pos::VectorF, grid_inds::Matrix{Int}, grid::Grid)
    grid_poses = grid_subset(grid, grid_inds)
    out = particle_grad(pos, grid_poses, grid.h)
end

function p2g!(particles::Particles, grid::Grid)
    plen = size(particles.mass,1)
    h = grid.h
    grid.mass = zeros(size(grid.mass))
    grid.momentum = zeros(size(grid.momentum))
    grid.velocity = zeros(size(grid.velocity))
    for i = 1:plen
        pos = particles.position[:,i]
        grid_inds = particle_grid_indices(pos, grid)
        particles.grid_inds[i] = grid_inds
        N_i = particle_weights_from_grid_inds(pos, grid_inds, grid)
        particles.w[i] = N_i
        p_w = particle_grad_weights_from_grid_inds(pos, grid_inds, grid)
        particles.w_grad[i] = particle_grad_weights_from_grid_inds(pos, grid_inds, grid)
        part_mass = particles.mass[i]
        part_vel = particles.velocity[:,i]
        for j=1:size(N_i,1)
            N_ij = N_i[j]
            grid.mass[grid_inds[:,j]...] += part_mass*N_ij
            grid.momentum[:,grid_inds[:,j]...] += part_mass.*part_vel.*N_ij
        end
    end
    colons = (Colon() for i=1:D)
    for i = 1:D
        grid.velocity[i, colons...] =  grid.momentum[i, colons...] ./ grid.mass
    end
    # Hack
    nan_inds = isnan.(grid.velocity)
    grid.velocity[nan_inds] .= 0.0
    return
end

particles = get_box_particles(1e-5, 500, [5.0, 5.0])
grid = generate_grid(0.0, 10.0, 101)
p2g!(particles, grid)
