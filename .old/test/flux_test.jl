# Code to activate project:
using Pkg; Pkg.activate(".")
# Code to activate project & install necessary packages
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Flux, Flux.Optimise, Flux.Losses
using Flux: @epochs
using DelimitedFiles
using Statistics

# Custom Dense layer

struct Affine
    W
    b
end

Affine(in::Integer, out::Integer) =
    Affine(randn(out, in), randn(out))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

Flux.@functor Affine # Needed to enable a lot of functions for the custom layer
#Flux.trainable(a::Affine) = (a.W,) # Define only the weights to be learnable

# Testing simple MLP: data read

datamat = readdlm("proben1/card/card1.dt", ' ', Float64, '\n', skipstart=7)
x = datamat[1:345, 1:51]
x = [x[i,:] for i in 1:size(x,1)]
y = datamat[1:345, 52:53]
y = [y[i,:] for i in 1:size(y,1)]
data = zip(x, y)

test_x = datamat[346:519, 1:51]
test_x = [test_x[i,:] for i in 1:size(test_x,1)]
test_y = datamat[346:519, 52:53]
test_y = [test_y[i,:] for i in 1:size(test_y,1)]

# Testing simple MLP: model creation and training

model = Chain(
    Dense(51, 128, σ),
    Dense(128, 2),
    softmax
)
ps = params(model)

loss(x, y) = mse(model(x), y)
eval() = mean([loss(x[i], y[i]) for i in 1:size(x,1)])
# eval() = mean([loss(test_x[i], test_y[i]) for i in 1:size(test_x,1)])
train_and_eval = function()
    train!(loss, ps, data, ADAM())
    @show(eval())
end
@show(eval())
@epochs 200 train_and_eval()