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
LEN_TRAIN = length(y_train)
x_train_cpu = convert(Array{Float32}, reshape(x_train, 28, 28, 1, :))
y_train_cpu = convert(Array{Float32}, onehotbatch(y_train, 0:9))
if args["model_cuda"] >= 0
    x_train_ = x_train_cpu |> gpu
    y_train_ = y_train_cpu |> gpu
else
    x_train_ = x_train_cpu
    y_train_ = y_train_cpu
end
@info "Train data" typeof(x_train_) size(x_train_) typeof(y_train_) size(y_train_)


##################################################
# Test dataset
testset = MNIST.testdata()

x_test, y_test = testset[:]
x_test_cpu = convert(Array{Float32}, reshape(x_test, 28, 28, 1, :))
y_test_cpu = convert(Array{Float32}, onehotbatch(y_test, 0:9))
if args["model_cuda"] >= 0
    x_test_ = x_test_cpu |> gpu
    y_test_ = y_test_cpu |> gpu
else
    x_test_ = x_test_cpu
    y_test_ = y_test_cpu
end
@info "Test data" typeof(x_test_) size(x_test_) typeof(y_test_) size(y_test_)


##################################################
# Model
model = Chain(
    Conv((3, 3), 1 => 8, relu, pad=(1, 1)),
    Conv((3, 3), 8 => 16, relu, pad=(1, 1)),
    Conv((3, 3), 16 => 16, relu, pad=(1, 1)),
    x -> reshape(x, (28 * 28 * 16, :)),
    Dropout(0.4),
    Dense(28 * 28 * 16 => 128, elu),
    Dropout(0.4),
    Dense(128 => 10, elu)
)
if args["model_cuda"] >= 0
    model = model |> gpu
end

lossF = (x, y) -> begin
    y_ = softmax(model(x), dims=1)
    return mean(-sum(y .* log.(y_), dims=1))
end

accuracy = (x_, y_; test_mode=false) -> begin
    # accuracy on cpu
    model_cpu = model |> cpu
    if test_mode
        testmode!(model_cpu)
    end
    acc = mean(argmax(model_cpu(x_ |> cpu), dims=1) .== argmax(y_ |> cpu, dims=1))
    return round(acc, digits=3)
end


##################################################
# training
function train(model)

    # opt = ADAM(0.01)
    opt = AdamW(0.001, (0.9, 0.999), 0.0001)
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end

    params = Flux.params(model)

    @info "Before training" accuracy(x_train_cpu, y_train_cpu) accuracy(x_test_cpu, y_test_cpu)

    BATCH_SIZE = 100
    for epoch in 1:10
        # shuffle training data
        s = shuffle(1:LEN_TRAIN) # s = 1:len_train
        x_train_s = x_train_[:, :, :, s]
        y_train_s = y_train_[:, s]
        # train batch
        for i in 1:BATCH_SIZE:LEN_TRAIN
            x = x_train_s[:, :, :, i:i+BATCH_SIZE-1]
            y = y_train_s[:, i:i+BATCH_SIZE-1]
            grads = gradient(() -> lossF(x, y), params)
            Flux.update!(opt, params, grads)
        end
        @info "Train epoch" epoch accuracy(x_train_cpu, y_train_cpu) accuracy(x_test_cpu, y_test_cpu)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end
    end
end


##################################################
# Test
function test(model)

    # run test
    @info "Test result" accuracy(x_test_cpu, y_test_cpu, test_mode=true)
end


##################################################
# Main
if abspath(PROGRAM_FILE) == @__FILE__
    train(model)
    if args["model_cuda"] >= 0
        CUDA.reclaim()
    end
    test(model)
end
