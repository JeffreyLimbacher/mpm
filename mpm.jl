

D = 2
N = 81

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
        0.5 * xabs^3.0 - xabs^2 + 2.0 / 3.0
    else
        max(((2 - xabs)^3) / 6.0)
    end
end

function p2g!(particles::Particles, grid::Grid)
    plen = size(particles.mass,1)
    for i = 1:plen
        pos = particles.position[i]
        diffs = grid.position .- [pos]
        diffs = diffs ./ grid.h
    end
end

particles = get_box_particles(1e-5, 500, [5.0, 5.0])
grid = generate_grid(0.0, 10.0, 101)
p2g!(particles, grid)
