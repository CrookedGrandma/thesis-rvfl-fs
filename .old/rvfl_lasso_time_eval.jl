using BenchmarkTools
using CSV
using DataFrames
using Lasso
using LinearAlgebra
using Statistics

include("./RVFL.jl")
include("./read_data.jl")

# Data reading
dat = read("card")
x = dat[:x][1]
x_list = dat[:x_list][1]
y = dat[:y][1]
y_vars = dat[:y_vars][1]

# Model creation
model = RVFL(51, 2, 2, radbas)

# Model training
get_output_lassopaths(x::Matrix{Float64}, y::Vector{Vector{Float64}}, λs::Vector{Float64}) =
    (v -> fit(LassoPath, x, v, λ = λs, standardize = false, intercept = false)).(y)

function benchmark_rvfl_lasso(ms::Vector{Int}, trials::Integer, x::Matrix{Float64}, y::Vector{Float64}, λs::Vector{Float64})::DataFrame
    mat::Matrix{Float64} = zeros(trials, length(ms))
    inputs = size(x, 2)
    x_list = [x[i,:] for i in 1:size(x,1)]
    mcount = 1
    for m in ms
        for t in 1:trials
            println("Benchmarking: variables = $(m + inputs), trial $t")
            model = RVFL(inputs, m, 1)
            enhancedX = enhanced(model, x_list)
            bench = @benchmark get_output_lassopaths($enhancedX, [$y], $λs) samples=100 evals=1 seconds=Inf
            mat[t, mcount] = median(bench.times)/1000 # in microseconds
        end
        mcount += 1
    end
    return DataFrame(mat, string.(ms .+ inputs))
end

ms = [0; collect(9:10:449)]
# ms = [0,1,2,3,4]
λs = .5 .^ collect(-5:14)

bench = benchmark_rvfl_lasso(ms, 100, x, y_vars[1], λs)
CSV.write("output_data/benchmark.csv", bench)