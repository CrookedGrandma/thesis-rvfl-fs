using DataFrames

function gather(df::DataFrame, col::Symbol, new_colname::Symbol)::DataFrame
    ids::typeof(df[!, col]) = df[!, col]
    cols::DataFrame = df[!, setdiff(names(df), [string(col)])]
    total::Vector{Matrix{Float64}} = Matrix{Float64}[]
    idcount::Int32 = 1
    for id in ids
        newrows = [[id, c] for c in cols[idcount, :]]
        push!(total, reduce(vcat, transpose.(newrows)))
        idcount += 1
    end
    alltotal::Matrix{Float64} = reduce(vcat, total)
    return DataFrame(alltotal, [col, new_colname])
end
