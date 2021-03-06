module DataMod

using CSV
using DataFrames
using MLJ: macro_f1score
using Random
using Statistics
using StatsBase
using StatsModels
using ..Helpers

export Data, Reader
export read_proben, read_uci, read_toy, preader, ureader, readers_full, readers_preliminary
export f1

struct Data
    df::DataFrame
    xdf::DataFrame
    xmat::Matrix{Float64}
    ydf::DataFrame
    ymat::VecOrMat{N} where N <: Number
    y1d::Vector{Int}
    xs::Vector{Symbol}
    ys::Vector{Symbol}
    nclass::Int
    function Data(df::DataFrame, xs::Vector{Symbol}, ys::Vector{Symbol})
        xdf = select(df, xs)
        ydf = select(df, ys)
        ymat = Matrix(ydf)
        new(df, xdf, Matrix(xdf), ydf, ymat, ys1d(ymat), xs, ys, length(ys))
    end
    function Data(data::Data, selxs::Vector{Symbol})
        selxdf = select(data.df, selxs)
        new(select(data.df, [selxs; data.ys]), selxdf, Matrix(selxdf), data.ydf, data.ymat, data.y1d, selxs, data.ys, length(data.ys))
    end
    function Data(data::Data, enhanced_df::DataFrame, enhanced_xs::Vector{Symbol})
        xdf = select(enhanced_df, enhanced_xs)
        new(enhanced_df, xdf, Matrix(xdf), data.ydf, data.ymat, data.y1d, enhanced_xs, data.ys, data.nclass)
    end
end
Base.show(io::IO, data::Data) = Base.show(io, data.df)
Base.iterate(d::Data) = (d, nothing)
Base.iterate(::Data, ::Any) = nothing
Base.length(d::Data) = 1

struct Reader
    name::String
    read::Function
    uci::Bool
    cv::Bool
    Reader(dataset::String, read::Function, uci::Bool) = new(dataset, read, uci, uci ? !isfile("uci/$dataset" * "_fix_test.csv") : false)
end
Base.show(io::IO, r::Reader) = begin
    cv = r.cv ? " - using " * string(getfield(r.read, 3)) * "-fold CV with RNG " * string(getfield(r.read, 2)) : ""
    repo = r.uci ? "UCI" * cv : "Proben1"
    print(io, "Reader of '" * r.name * "' dataset of " * repo)
end

function f1(models::Vector{<:RegressionModel}, data::Data)
    predmat = hcat([predict(m, data.xmat) for m in models]...)
    predy1d = ys1d(predmat)
    return macro_f1score(predy1d, data.y1d)
end

function kfold_cv_part(dat::DataFrame, rng::AbstractRNG, k::Int)::Vector{Tuple{DataFrame, DataFrame}}
    n = nrow(dat)
    idx = shuffle(rng, 1:n)
    rawsize = n / k
    nceil = k * (rawsize % 1)
    size = floor(Int, rawsize)
    sizes = [[size for _ in 1:(k - nceil)]; [size + 1 for _ in 1:nceil]]
    sum = 0
    idxs = []
    for s in sizes
        push!(idxs, idx[(sum + 1):(sum + s)])
        sum += s
    end
    return [(dat[setdiff(idx, i), :], dat[i, :]) for i in idxs]
end

function xnames(n::Integer; y::Bool = false)::Vector{Symbol}
    n == 1 && return y ? [:y] : [:x]
    return Symbol.(y ? 'y' : 'x', 1:n)
end
ynames(n::Integer)::Vector{Symbol} = xnames(n, y=true)

function read_proben(dataset::String, permutation::Int = 1, collapse_binary::Bool = true)::Tuple{Data, Data}#, Data}
    path = "proben1/$dataset/$dataset$permutation.dt"
    # 1 = bool_in; 2 = real_in; 3 = bool_out; 4 = bool_in; 5 = training_examples; 6 = validation_examples; 7 = test_examples
    heads = parse.(Int32, [x[2] for x in split.(Iterators.take(eachline(path), 7), '=')])
    nin = max(heads[1], heads[2])
    nout = max(heads[3], heads[4])
    ntrain = heads[5]
    nvalid = heads[6]
    ntest = heads[7]
    xs = xnames(nin)
    ys = ynames(nout)
    dat = CSV.read(path, DataFrame, header=[xs; ys], skipto=8, delim=' ', types=Float64, ignorerepeated=true)
    if collapse_binary && length(ys) == 2
        select!(dat, Not(ys[1]))
        deleteat!(ys, 1)
    end
    one_value = filter(x -> length(unique(dat[!, x])) == 1, xs)
    no_class = filter(y -> mean(dat[!, y]) == 0, ys)
    select!(dat, Not([one_value; no_class]))
    setdiff!(xs, one_value)
    setdiff!(ys, no_class)
    d = @inbounds dat[1:ntrain, :]
    # d_valid = @inbounds dat[ntrain+1:ntrain+nvalid, :]
    d_test = @inbounds dat[ntrain+nvalid+1:ntrain+nvalid+ntest, :]
    return (
        Data(d, xs, ys),
        # Data(d_valid, xs, ys),
        Data(d_test, xs, ys)
    )
end

function read_uci(dataset::String, rng::AbstractRNG = MersenneTwister(), k::Int = 4)::Union{Tuple{Data, Data}, Vector{Tuple{Data, Data}}}
    path = "uci/$dataset" * "_fix.csv"
    testpath = "uci/$dataset" * "_fix_test.csv"
    heads = parse.(Int32, [x[2] for x in split.(Iterators.take(eachline(path), 2), '=')])
    nin = heads[1]
    nout = heads[2]
    xs = xnames(nin)
    ys = ynames(nout)
    dat = CSV.read(path, DataFrame, header=[xs; ys], skipto=3, delim=',', types=Float64, ignorerepeated=true)
    if isfile(testpath)
        testdat = CSV.read(testpath, DataFrame, header=[xs; ys], skipto=3, delim=',', types=Float64, ignorerepeated=true)
        return (Data(dat, xs, ys), Data(testdat, xs, ys))
    else
        cv_dat = kfold_cv_part(dat, rng, k)
        return [(Data(dat, xs, ys), Data(testdat, xs, ys)) for (dat, testdat) in cv_dat]
    end
end

read_toy()::Data = read_proben("_toy")[1]

preader(dataset::String)::Reader = Reader(dataset, () -> read_proben(dataset), false)
ureader(dataset::String, rng::AbstractRNG = MersenneTwister(), k::Int = 4)::Reader = Reader(dataset, () -> read_uci(dataset, rng, k), true)

function readers_full(start::Int = 1)::Vector{Reader}
    dp = ["cancer", "card", "gene", "glass", "heart", "horse", "soybean", "thyroid"]
    du = ["abalone", "adult", "covtype", "madelon", "optdigits", "page-blocks", "pendigits", 
          "poker", "satimage", "segmentation", "shuttle", "spect", "vehicle", "waveform"]
    return [[preader(d) for d in dp]; [ureader(du[d], MersenneTwister(123d)) for d in eachindex(du)]][start:end]
end
function readers_preliminary(start::Int = 1)::Vector{Reader}
    dp = ["cancer", "card", "gene", "glass", "heart", "horse", "mushroom", "soybean", "thyroid"][start:end]
    return [preader(d) for d in dp]
end

end

using .DataMod
