include("./data.jl")

using Flux
using Zygote
using CUDA
using Random
using StatsBase
using ProgressMeter: Progress, next!

BATCH_SIZE = args["train_batch_size"]

##################################################
# returns a constructed model
modelF = (dim_1::Int, dim_2::Int, channel_n::Int) -> begin
    model = Chain(
        Conv((4, 4), 1 => channel_n, relu, pad=(1, 1), stride=(2, 2)),
        Conv((4, 4), channel_n => channel_n * 2, relu, pad=(1, 1), stride=(2, 2)),
        Conv((4, 4), channel_n * 2 => channel_n * 4, relu, pad=(1, 1), stride=(2, 2)),
        Flux.flatten,
        Dropout(0.5),
        Dense(div(dim_1, 8) * div(dim_2, 8) * channel_n * 4 => channel_n * 4, relu),
        Dropout(0.5),
        Dense(channel_n * 4 => 10),
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
model_ = modelF(size(x_train_)[1], size(x_train_)[2], args["model_channel_n"])
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

    data_loader = Flux.Data.DataLoader((x_train_, y_train_), batchsize=BATCH_SIZE, shuffle=true)

    function train_epoch(epoch)
        # total loss
        loss_total = 0.0f0
        # progress tracker
        progress_tracker = Progress(length(data_loader), 1, "Training epoch $(epoch): ")
        for (x_, y_) in data_loader
            (() -> begin
                if args["model_cuda"] >= 0
                    x_ = x_ |> gpu
                    y_ = y_ |> gpu
                end
                #(loss_curr, loss_recon, loss_kl, mu, log_var)
                loss_curr, back = pullback(params) do
                    lossF(model_, x_, y_)
                end
                gradients = back(1.0f0)
                Flux.update!(opt, params, gradients)
                # update progress tracker
                loss_total += loss_curr
                loss_avg = loss_total / (progress_tracker.counter + 1)
                next!(progress_tracker; showvalues=[
                    (:loss, round(loss_avg, digits=2)),
                    (:curr, round(loss_curr, digits=2)),
                    # (:recon, round(loss_recon, digits=2)),
                    # (:kl, round(loss_kl, digits=2)),
                    # (:mu, round.(mu, digits=2)),
                    # (:log_var, round.(log_var, digits=2)),
                ])
            end)()
            # reclaim GPU memory
            if mod(progress_tracker.counter, 100) == 0
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
        train_epoch(epoch)
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
