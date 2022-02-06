using BSON: @save
using CSV
using DataFrames
using Flux, Flux.Optimise, Flux.Losses
using Flux: @epochs

include("./read_data.jl")

same_class(ŷ::Vector{Float64}, y::Vector{Float64})::Bool = argmax(ŷ) == argmax(y)

accuracy(model, x_list::Vector{Vector{Float64}}, y_list::Vector{Vector{Float64}})::Float64 =
    sum(same_class.(model.(x_list), y_list)) / length(x_list)

ind(i::Int, f::Int)::Int = convert(Int32, i / f) + 1

function insert_mat(mat::Matrix{Float64}, i::Int, log_freq::Int, acc::Float64)
    mat[i == 1 ? 1 : ind(i, log_freq), :] = [i, acc]
end

function train_and_log(model, traindat::Iterators.Zip, x_test_list::Vector{Vector{Float64}},
        y_test_list::Vector{Vector{Float64}}, i::Int, log_freq::Int, mat::Matrix{Float64}, name::String="")
    loss(x, y) = mse(model(x), y)
    ps = Flux.params(model)
    train!(loss, ps, traindat, ADAM())
    if i % log_freq == 0 || i == 1
        acc::Float64 = accuracy(model, x_test_list, y_test_list)
        if i == 1 || acc > maximum(mat[:, 2])
            println("Accuracy improvement - saving model.")
            @save "models/$name-$(round(acc, digits=3)).bson" model
        end
        insert_mat(mat, i, log_freq, acc)
    end
end

function benchmark_fnn_perf(traindat::Iterators.Zip, x_test_list::Vector{Vector{Float64}},
        y_test_list::Vector{Vector{Float64}}, epochs::Int, log_freq::Int, name::String="")::DataFrame
    mat::Matrix{Float64} = zeros(round(Int, epochs / log_freq + 1, RoundDown), 2)
    in::Int = length(traindat.is[1][1])
    out::Int = length(traindat.is[2][1])
    hidden::Int = in * 2
    slfn = Chain(
        Dense(in, hidden, σ),
        Dense(hidden, out),
        softmax
    )
    i::Int = 0
    @epochs epochs (i += 1; train_and_log(slfn, traindat, x_test_list, y_test_list, i, log_freq, mat, name))
    if i % log_freq != 0
        acc::Float64 = accuracy(slfn, x_test_list, y_test_list)
        if i == 1 || acc > maximum(mat[:, 2])
            println("Accuracy improvement - saving model.")
            @save "models/$name-$(round(acc, digits=3)).bson" model
        end
        insert_mat(mat, i, log_freq, acc)
    end
    return DataFrame(mat, [:epoch, :accuracy])
end

benchmark_fnn_perf(dat::Dict{Symbol, AbstractArray}, epochs::Int, log_freq::Int)::DataFrame =
    benchmark_fnn_perf(dat[:zip][1], dat[:x_list][3], dat[:y_list][3], epochs, log_freq, dat[:info][6])

datasets = ["cancer", "card", "diabetes", "gene", "glass", "heart", "horse", "mushroom", "soybean", "thyroid"]
# datasets = ["cancer", "diabetes", "gene", "glass", "heart", "horse", "mushroom", "soybean", "thyroid"]
for dataset in datasets
    println("Benchmarking performance of $dataset dataset...")
    dat = readd(dataset)
    bench = benchmark_fnn_perf(dat, 1000, 5)
    CSV.write("output_data/benchmark_fnn_perf_$dataset.csv", bench)
    sleep(5)
end
