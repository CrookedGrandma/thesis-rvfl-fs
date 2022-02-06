using CSV
using DataFrames
using Lasso

include("./RVFL.jl")
include("./read_data.jl")

get_weights(x::Matrix{Float64}, y::Vector{Vector{Float64}}, λ::Float64)::Matrix{Float64} =
    Matrix(reduce(vcat, (lp::LassoPath -> transpose(lp.coefs)).(
        (v -> fit(LassoPath, x, v, λ = [λ], standardize = false, intercept = false, cd_maxiter = typemax(Int))).(y)
    )))

same_class(ŷ::Vector{Float64}, y::Vector{Float64})::Bool = argmax(ŷ) == argmax(y)

accuracy(model::RVFL, W::Matrix{Float64}, x_list::Vector{Vector{Float64}}, y_list::Vector{Vector{Float64}})::Float64 =
    sum(same_class.((x::Vector{Float64} -> model(x, W)).(x_list), y_list)) / length(x_list)

function benchmark_rvfl_lasso_perf(ms::Vector{Int}, trials::Int, λs::Vector{Float64},
        x_train::Matrix{Float64}, y_train_vars::Vector{Vector{Float64}},
        x_test_list::Vector{Vector{Float64}}, y_test_list::Vector{Vector{Float64}})::DataFrame
    msize::Int = length(ms)
    λsize::Int = length(λs)
    in::Int = size(x_train, 2)
    out::Int = length(y_train_vars)
    mat::Matrix{Float64} = zeros(msize * trials * λsize, 3)
    errcount::Int = 0
    mcount::Int = 1
    for m::Int in ms
        for t::Int in 1:trials
            λcount::Int = 1
            for λ::Float64 in λs
                println("Testing: variables = $(m + in), trial $t, lambda = $λ")
                acc::Float64 = 0
                try
                    model::RVFL = RVFL(in, m, out)
                    enhancedX = enhanced(model, x_train)
                    W::Matrix{Float64} = get_weights(enhancedX, y_train_vars, λ)
                    acc = accuracy(model, W, x_test_list, y_test_list)
                catch e
                    println("Error encountered: $(sprint(showerror, e))")
                    acc = -1
                    errcount += 1
                end
                mat[(mcount - 1) * trials * λsize + (t - 1) * λsize + λcount, :] = [in + m, λ, acc]
                λcount += 1
            end
        end
        mcount += 1
    end
    if errcount > 0
        println("$errcount errors occurred while testing this file.")
    end
    return DataFrame(mat, [:vars, :lambda, :accuracy])
end

benchmark_rvfl_lasso_perf(ms::Vector{Int}, trials::Int, λs::Vector{Float64}, dat::Dict{Symbol, AbstractArray})::DataFrame =
    benchmark_rvfl_lasso_perf(ms, trials, λs, dat[:x][1], dat[:y_vars][1], dat[:x_list][3], dat[:y_list][3])

λs = .5 .^ collect(-5:14)

datasets = ["cancer", "card", "diabetes", "gene", "glass", "heart", "horse", "mushroom", "soybean", "thyroid"]
for dataset in datasets
    println("Benchmarking performance of $dataset dataset...")
    dat = readd(dataset)
    to_ten = 10 - dat[:info][1] % 10
    ms = [0; round.(Int32, range(to_ten, 500 - dat[:info][1], length=23))]
    bench = benchmark_rvfl_lasso_perf(ms, 50, λs, dat)
    CSV.write("output_data/benchmark_perf_$dataset.csv", bench)
    sleep(5)
end
