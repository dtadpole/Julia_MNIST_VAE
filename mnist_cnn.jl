include("./data.jl")

using Flux
using CUDA
using Random
using StatsBase


##################################################
# returns a constructed model
modelF = (dim_1::Int, dim_2::Int, channel_n::Int) -> begin
    model = Chain(
        Conv((3, 3), 1 => channel_n, relu, pad=(1, 1)),
        MaxPool((2, 2)),
        Conv((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        MaxPool((2, 2)),
        Conv((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        Flux.flatten,
        Dropout(0.5),
        Dense(div(dim_1, 4) * div(dim_2, 4) * channel_n => channel_n * 4, elu),
        Dropout(0.5),
        Dense(channel_n * 4 => 10, elu),
        softmax
    )
    # model to GPU if available
    if args["model_cuda"] >= 0
        model = model |> gpu
    end
    # @info "Model" model
    # return a function that returns the model
    return model
end

# loss function
lossF = (model_, x_, y_) -> begin
    y_pred = model_(x_)
    mean(-sum(y_ .* log.(y_pred), dims=1))
end

# accuracy function
accuracyF = (model_, x_, y_; test_mode=false, size_=nothing) -> begin
    # if sample size if specified
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

# get model
model_ = modelF(28, 28, args["model_channel_n"])
@info "Model" model_

##################################################
# training
function train()

    global model_

    # opt = ADAM(0.01)
    opt = AdamW(args["train_lr"], (0.9, 0.999), args["train_weight_decay"])
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end

    params = Flux.params(model_)

    BATCH_SIZE = args["train_batch_size"]

    function train_epoch()
        # shuffle training data
        s = shuffle(1:TRAIN_LENGTH) # s = 1:len_train
        x_train_s = x_train_[:, :, :, s]
        y_train_s = y_train_[:, s]
        # train batch
        for i in 1:BATCH_SIZE:TRAIN_LENGTH
            (() -> begin
                x_ = x_train_s[:, :, :, i:i+BATCH_SIZE-1]
                y_ = y_train_s[:, i:i+BATCH_SIZE-1]
                # @info "sizes" size(x_) size(y_)
                if args["model_cuda"] >= 0
                    x = Array{Float32,4}(undef, size(x_))
                    x[:, :, :, 1:BATCH_SIZE] = x_
                    x_ = x |> gpu
                    # x_ = x_ |> gpu
                    y = Array{Float32,2}(undef, size(y_))
                    y[:, 1:BATCH_SIZE] = y_
                    y_ = y |> gpu
                    # y_ = y_ |> gpu
                end
                grads = gradient(() -> lossF(model_, x_, y_), params)
                Flux.update!(opt, params, grads)
            end)()
            # reclaim GPU memory
            if mod(i, BATCH_SIZE * 100) == 1
                GC.gc(true)
                if args["model_cuda"] >= 0
                    CUDA.reclaim()
                end
            end
        end
    end

    start_time = time()
    accuracy_train = accuracyF(model_, x_train_, y_train_, size_=5_000)
    accuracy_test = accuracyF(model_, x_test_, y_test_, size_=5_000)
    accuracy_time = round(time() - start_time, digits=1)
    @info "Before training" accuracy_time accuracy_train accuracy_test
    # GC and reclaim GPU memory
    GC.gc(true)
    if args["model_cuda"] >= 0
        CUDA.reclaim()
    end

    for epoch in 1:args["train_epochs"]
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
        accuracy_train = accuracyF(model_, x_train_, y_train_, size_=5_000)
        accuracy_test = accuracyF(model_, x_test_, y_test_, size_=5_000)
        accuracy_time = round(time() - start_time, digits=1)
        @info "[$(train_time)s] Train epoch [$(epoch)]" accuracy_time accuracy_train accuracy_test
        # GC and reclaim GPU memory
        GC.gc(true)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end
    end
end


##################################################
# Test
function test()
    global model_
    start_time = time()
    accuracy_test = accuracyF(model_, x_test_, y_test_, test_mode=true, size_=5_000)
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
