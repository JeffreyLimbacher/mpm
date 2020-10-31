

D = 2
N = 81

VectorF = Vector{Float64}

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

function add_box(
    n::Int64,
    center::Vector{Float64} = zeros(D),
    extents::Vector{Float64} = ones(D),
)::Array{Float64, 2}
    (rand(Float64, (D, n)) .- 0.5) .* extents .+ center
end

function get_box_particles(
    particle_mass::Float64,
    n::Int64,
    center::Vector{Float64} = zeros(D),
    extents::Vector{Float64} = ones(D),
)::Particles
    positions = add_box(n, center, extents)
    Particles(ones(n) .* particle_mass, positions, zeros(D, n))
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

function kernel_indices(inds, max_inds)::Array{Array{Int, 1}, 1}
    cardinals = Iterators.product(-1:1, -1:1)
    new_inds = [inds.+x for x=cardinals if all(inds.+x .>= 1) && all(inds.+x .<= max_inds)]
end

function particle_grid_indices(pos::Array{Float64, 1}, grid::Grid)::Array{Array{Int, 1}, 1}
    inds = (1 for i=1:D)
    h = grid.h
    indices = floor.(Int, (pos .- grid.position[:,inds...]) ./ h)
    kinds = kernel_indices(indices, size(grid.position)[2:end])
end

function particle_weights(part_pos::Vector{Float64}, grid::Grid)
    h = grid.h
    # get indices of grid, grid[1,1,:] is the coordinates of bottom indices
    grid_inds = particle_grid_indices(part_pos, grid)
    grid_poses = [grid.position[:,g...] for g in grid_inds]
    grid_poses = hcat(grid_poses...)
    diffs = part_pos .- grid_poses
    diffs = diffs ./ h
    # adjust for one based indexing
    diffs = diffs .- 1
    kernel_values = kernel.(diffs)
end

function p2g!(particles::Particles, grid::Grid)
    plen = size(particles.mass,1)
    h = grid.h
    for i = 1:plen
        pos = particles.position[i]
        N_i = particle_weights(pos, grid)
    end
end

particles = get_box_particles(1e-5, 500, [5.0, 5.0])
grid = generate_grid(0.0, 10.0, 101)
particle_weights(particles.position[:,1], grid)
