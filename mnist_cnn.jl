include("./args.jl")

using MLDatasets
using Flux
using CUDA
using Random
using OneHotArrays
using StatsBase


##################################################
# Training dataset
trainset = MNIST.traindata()

x_train_, y_train_ = trainset[:]
x_train_ = convert(Array{Float32}, reshape(x_train_, 28, 28, 1, :))
y_train_ = convert(Array{Float32}, onehotbatch(y_train_, 0:9))
@info "Train data" typeof(x_train_) size(x_train_) typeof(y_train_) size(y_train_)

TRAIN_LENGTH = size(y_train_, 2)

##################################################
# Test dataset
testset = MNIST.testdata()

x_test_, y_test_ = testset[:]
x_test_ = convert(Array{Float32}, reshape(x_test_, 28, 28, 1, :))
y_test_ = convert(Array{Float32}, onehotbatch(y_test_, 0:9))
@info "Test data" typeof(x_test_) size(x_test_) typeof(y_test_) size(y_test_)


##################################################
# returns a function that returns the model
modelF = (() -> begin
    model = Chain(
        Conv((3, 3), 1 => 32, relu, pad=(1, 1)),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 32, relu, pad=(1, 1)),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 32, relu, pad=(1, 1)),
        x -> reshape(x, (7 * 7 * 32, :)),
        Dropout(0.5),
        Dense(7 * 7 * 32 => 128, elu),
        Dropout(0.5),
        Dense(128 => 10, elu),
        softmax
    )
    # model to GPU if available
    if args["model_cuda"] >= 0
        model = model |> gpu
    end
    # @info "Model" model
    # return a function that returns the model
    return () -> model
end)()

# return a function that returns loss function
lossF = (model_, x_, y_) -> begin
    y_pred = model_(x_)
    mean(-sum(y_ .* log.(y_pred), dims=1))
end

# accuracy function
accuracy = (model_, x_, y_; test_mode=false, size_=nothing) -> begin
    # accuracy on cpu
    # if args["model_cuda"] >= 0
    #     model_ = model_ |> cpu
    #     x_ = x_ |> cpu
    #     y_ = y_ |> cpu
    # end
    # sample size if specified
    if size_ !== nothing
        s = sample(1:size(y_, 2), size_, replace=false)
        x_ = x_[:, :, :, s]
        y_ = y_[:, s]
        if args["model_cuda"] >= 0
            x_ = x_ |> gpu
            y_ = y_ |> gpu
        end
    end
    # disable dropout
    if test_mode
        testmode!(model_)
    end
    # accuracy
    acc = mean(argmax(model_(x_), dims=1) .== argmax(y_, dims=1))
    return round(acc, digits=3)
end


##################################################
# training
function train()

    # get model
    model_ = modelF()
    @info "Model" model_

    # opt = ADAM(0.01)
    opt = AdamW(0.001, (0.9, 0.999), 0.0001)
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end

    params = Flux.params(model_)

    BATCH_SIZE = 100

    function train_epoch()
        # shuffle training data
        s = shuffle(1:TRAIN_LENGTH) # s = 1:len_train
        x_train_s = x_train_[:, :, :, s]
        y_train_s = y_train_[:, s]
        # train batch
        for i in 1:BATCH_SIZE:TRAIN_LENGTH
            x_ = x_train_s[:, :, :, i:i+BATCH_SIZE-1]
            y_ = y_train_s[:, i:i+BATCH_SIZE-1]
            # @info "sizes" size(x_) size(y_)
            if args["model_cuda"] >= 0
                # x = Array{Float32,4}(undef, size(x_))
                # x[:, :, :, 1:BATCH_SIZE] = x_
                # x_ = x |> gpu
                x_ = x_ |> gpu
                # y = Array{Float32,2}(undef, size(y_))
                # y[:, 1:BATCH_SIZE] = y_
                # y_ = y |> gpu
                y_ = y_ |> gpu
            end
            grads = gradient(() -> lossF(model_, x_, y_), params)
            Flux.update!(opt, params, grads)
            # reclaim GPU memory
            if mod(i, 20) == 0
                GC.gc(true)
                if args["model_cuda"] >= 0
                    CUDA.reclaim()
                end
            end
        end
    end

    start_time = time()
    accuracy_train = accuracy(model_, x_train_, y_train_, size_=5_000)
    accuracy_test = accuracy(model_, x_test_, y_test_, size_=5_000)
    accuracy_time = round(time() - start_time, digits=1)
    @info "Before training" accuracy_time accuracy_train accuracy_test

    for epoch in 1:10
        # start time
        start_time = time()
        # train epoch
        train_epoch()
        train_time = round(time() - start_time, digits=1)
        # GC and reclaim GPU memory
        GC.gc(true)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end
        # calculate accuracy
        start_time = time()
        accuracy_train = accuracy(model_, x_train_, y_train_, size_=5_000)
        accuracy_test = accuracy(model_, x_test_, y_test_, size_=5_000)
        accuracy_time = round(time() - start_time, digits=1)
        @info "[$(train_time)s] Train epoch [$(epoch)]" accuracy_time accuracy_train accuracy_test
    end
end


##################################################
# Test
function test()
    model_ = modelF()
    start_time = time()
    accuracy_test = accuracy(model_, x_test_, y_test_, test_mode=true, size_=5_000)
    accuracy_time = round(time() - start_time, digits=1)
    @info "Test result" accuracy_time accuracy_test
end


##################################################
# Main
if abspath(PROGRAM_FILE) == @__FILE__
    train()
    if args["model_cuda"] >= 0
        CUDA.reclaim()
    end
    test()
end
