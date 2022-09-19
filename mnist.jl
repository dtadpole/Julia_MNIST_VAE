using ArgParse
using MLDatasets
using Flux
using CUDA
using Random
using OneHotArrays
using StatsBase

##################################################
# Parse command line arguments
function parse_commandline()

    s = ArgParseSettings()
    @add_arg_table s begin

        "--model_cuda"
        help = "model cuda number"
        arg_type = Int
        default = -1
    end
    return parse_args(s)
end

args = parse_commandline()

if args["model_cuda"] >= 0
    CUDA.allowscalar(false)
    CUDA.device!(args["model_cuda"])
end

##################################################
# Training dataset
trainset = MNIST.traindata()

x_train, y_train = trainset[:]
const len_train = length(y_train)
x_train_ = reshape(x_train, 28, 28, 1, :)
y_train_ = convert(Array{Float32}, onehotbatch(y_train, 0:9))
# @info "Train data" typeof(trainset[1]) size(trainset[1]) typeof(trainset[2]) size(trainset[2])

##################################################
# Test dataset
testset = MNIST.testdata()

x_test, y_test = trainset[:]
x_test_ = reshape(x_test, 28, 28, 1, :)
y_test_ = convert(Array{Float32}, onehotbatch(y_test, 0:9))
# @info "Test data" typeof(trainset[1]) size(trainset[1]) typeof(trainset[2]) size(trainset[2])


##################################################
# training
function train()

    global len_train
    global x_train_
    global y_train_
    global x_test_
    global y_test_

    model = Chain(
        Conv((3, 3), 1 => 4, relu, pad=(1, 1), stride=(1, 1)),
        Conv((3, 3), 4 => 8, relu, pad=(1, 1), stride=(1, 1)),
        x -> reshape(x, (28 * 28 * 8, :)),
        # Dropout(0.5),
        Dense(28 * 28 * 8 => 64, elu),
        # Dropout(0.5),
        Dense(64 => 10, elu),
        softmax
    )
    if args["model_cuda"] >= 0
        model = model |> gpu
    end

    lossF = (x, y) -> begin
        y_ = model(x)
        return mean(-sum(y .* log.(y_), dims=1))
    end

    accuracy = (x_, y_) -> round(mean(argmax(model(x_), dims=1) .== argmax(y_, dims=1)), digits=3)

    @info "Before training" accuracy(x_train_ |> cpu, y_train_ |> cpu) accuracy(x_test_ |> cpu, y_test_ |> cpu)

    # opt = ADAM(0.01)
    opt = AdamW(0.01, (0.9, 0.999), 0.00001)
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end

    params = Flux.params(model)

    BATCH_SIZE = 100
    for epoch in 1:10
        # shuffle training data
        s = shuffle(1:len_train) # s = 1:len_train
        x_train_s = x_train_[:, :, :, s]
        y_train_s = y_train_[:, s]
        # train batch
        for i in 1:BATCH_SIZE:len_train
            x = x_train_s[:, :, :, i:i+BATCH_SIZE-1]
            y = y_train_s[:, i:i+BATCH_SIZE-1]
            if args["model_cuda"] >= 0
                x = x |> gpu
                y = y |> gpu
            end
            grads = gradient(() -> lossF(x, y), params)
            Flux.update!(opt, params, grads)
        end
        @info "Train epoch" epoch accuracy(x_train_ |> cpu, y_train_ |> cpu) accuracy(x_test_ |> cpu, y_test_ |> cpu)
    end
end

##################################################
# test
function test()
    global x_test_
    global y_test_
    # run test
    testmode!(model)
    @info "Test result" accuracy(x_test_ |> cpu, y_test_ |> cpu)
end

train()

# test()
