using MLDatasets
using Flux
using Random
using OneHotArrays
using StatsBase

trainset = MNIST.traindata()

x_train, y_train = trainset[:]
len_train = length(y_train)
x_train_ = reshape(x_train, 28, 28, 1, :)
y_train_ = convert(Array{Float32}, onehotbatch(y_train, 0:9))
# @info "Train data" typeof(trainset[1]) size(trainset[1]) typeof(trainset[2]) size(trainset[2])

model = Chain(
    Conv((3, 3), 1 => 4, relu),
    Conv((3, 3), 4 => 8, relu),
    x -> reshape(x, 24 * 24 * 8, :),
    # Dropout(0.5),
    Dense(24 * 24 * 8, 64, elu),
    # Dropout(0.5),
    Dense(64, 10, elu),
    softmax
)

lossF = (x, y) -> begin
    y_ = model(x)
    return mean(-sum(y .* log.(y_), dims=1))
end

accuracy = (x_, y_) -> round(sum(argmax(model(x_), dims=1) .== argmax(y_, dims=1)) / size(y_, 2), digits=3)

@info "Before training" loss

# opt = ADAM(0.01)
opt = AdamW(0.01, (0.9, 0.999), 0.001)
params = Flux.params(model)

BATCH_SIZE = 100
for epoch in 1:10
    # shuffle training data
    # s = shuffle(1:len_train)
    s = 1:len_train
    x_train_s = x_train_[:, :, :, s]
    y_train_s = y_train_[:, s]
    # train batch
    for i in 1:BATCH_SIZE:len_train
        x = x_train_s[:, :, :, i:i+BATCH_SIZE-1]
        y = y_train_s[:, i:i+BATCH_SIZE-1]
        grads = gradient(() -> lossF(x, y), params)
        Flux.update!(opt, params, grads)
    end
    @info "Train epoch" epoch lossF(x_train_, y_train_) accuracy(x_train_, y_train_)
end

# test
testset = MNIST.testdata()

x_test, y_test = trainset[:]
x_test_ = reshape(x_test, 28, 28, 1, :)
y_test_ = convert(Array{Float32}, onehotbatch(y_test, 0:9))
# @info "Test data" typeof(trainset[1]) size(trainset[1]) typeof(trainset[2]) size(trainset[2])

testmode!(model)
@info "Test result" lossF(x_test_, y_test_) accuracy(x_test_, y_test_)
