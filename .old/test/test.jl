# LASSO test
using DataFrames
using Lasso

data = DataFrame(X=[1,2,3], Y=[2,4,7])
X = data[!, setdiff(names(data), ["Y"])]
Y = data[!, "Y"]
lmodel = fit(LassoPath, Matrix(X), Y, Normal())

# MXNet test
using MXNet

batch_size = 100
filenames = mx.get_mnist_ubyte()
train_provider = mx.MNISTProvider(image=filenames[:train_data],
    label=filenames[:train_label],
    data_name=:data, label_name=:softmax_label,
    batch_size=batch_size, shuffle=true, flat=true, silent=true)
eval_provider = mx.MNISTProvider(image=filenames[:test_data],
    label=filenames[:test_label],
    data_name=:data, label_name=:softmax_label,
    batch_size=batch_size, shuffle=false, flat=true, silent=true)

println("Test print")

mlp = @mx.chain mx.Variable(:data) =>
    mx.FullyConnected(name=:fc1, num_hidden=128) =>
    mx.Activation(name=:relu1, act_type=:relu) =>
    mx.FullyConnected(name=:fc2, num_hidden=64) =>
    mx.Activation(name=:relu2, act_type=:relu) =>
    mx.FullyConnected(name=:fc3, num_hidden=10) =>
    mx.SoftmaxOutput(name=:softmax)

model = mx.FeedForward(mlp, context=mx.cpu())
optimizer = mx.SGD(η=0.1, μ=0.9, λ=0.00001)
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)

probs = mx.predict(model, eval_provider)