using CSV
using DataFrames
using LsqFit
using Plots; pyplot()
using Statistics
using StatsPlots

include("./df-gather.jl")
include("./plot-twiny.jl")

# Time benchmark

bench = CSV.read("output_data/benchmark.csv", DataFrame, header=false, transpose=true)
rename!(bench, :Column1 => :vars)
bench = gather(bench, :vars, :times)
bench[!, :times] /= 1e3

@df bench boxplot(:vars, :times, label=false, bar_width=8, markersize=2, title="LASSO training on card dataset",
    xlabel="Input features", ylabel="Time (ms)", size=(800, 600))

means = combine(groupby(bench, :vars), :times => mean => :mean)
frange = range(50, 500, length = 100)
@. fitmodel(x, p) = p[1] * x/100 + p[2] * (x/100)^2 + p[3] * (x/100)^3
fit = curve_fit(fitmodel, means[!, :vars], means[!, :mean], [.5, .5, .5])
plot!(frange, fitmodel(frange, coef(fit)), linecolor=:red, linewidth=2, label="3rd degree polynomial fit", legend=:bottomright)
png("plots/time_card")

# Performance benchmark

datasets = ["cancer", "card", "diabetes", "gene", "glass", "heart", "horse", "mushroom", "soybean", "thyroid"]
function plot_perf(dataset::String)
    λs = .5 .^ collect(-5:14)
    bench = CSV.read("output_data/benchmark_perf_$dataset.csv", DataFrame)
    bench = combine(groupby(bench, [:vars, :lambda]), :accuracy => mean => :mean_acc)
    bench[!, :vars] = convert.(Int, bench[!, :vars])
    plt = @df bench plot(:lambda, :mean_acc, group=:vars, xaxis=(λs, :log2, :flip),
        linewidth=2, linealpha=0.8, linestyle=:auto, title="RVFL: $dataset dataset",
        legend=:topleft, legendtitle="#inputs", legendtitlefontsize=9,
        xlabel="λ", ylabel="Mean accuracy", ylims=(0, 1), widen=true, size=(800, 600))
    png("plots/rvfl_$dataset")
    return plt
end
function plot_perf_fnn(dataset::String)
    fnn = CSV.read("output_data/benchmark_fnn_perf_$dataset.csv", DataFrame)
    plt = @df fnn plot(:epoch, :accuracy, label=false, xlabel="Epochs", ylabel="Accuracy",
        legend=:bottomright, title="FNN: $dataset dataset", ylims=(0, 1), widen=true, size=(800, 600))
    png("plots/fnn_$dataset")
    return plt
end
function plotnext(;fnn::Bool = false)
    if (!(@isdefined i) || i >= length(datasets))
        global i = 0
    end
    global i += 1
    if !fnn
        plot_perf(datasets[i])
    else
        plot_perf_fnn(datasets[i])
    end
end

for i in 1:10; plotnext(); end
for i in 1:10; plotnext(fnn = true); end
