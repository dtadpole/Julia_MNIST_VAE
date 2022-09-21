include("./data.jl")

using Flux
using Zygote
using CUDA
using Random
using StatsBase
using Distributions
using ProgressMeter: Progress, next!

# multivariate normal distribution
multivariate_normal = Distributions.MvNormal(zeros(Float32, args["latent_n"]), ones(Float32, args["latent_n"]))
normal = Distributions.Normal(0.0f0, 1.0f0)


##################################################
# returns a function that returns the model
create_vae = (dim_1::Int, dim_2::Int, channel_n::Int, latent_n::Int) -> begin

    # returns a function that returns the encoder
    encoder = Chain(
        Conv((4, 4), 1 => channel_n, relu, pad=(1, 1), stride=(2, 2)),
        # MaxPool((2, 2)),
        Conv((4, 4), channel_n => channel_n, relu, pad=(1, 1), stride=(2, 2)),
        # MaxPool((2, 2)),
        Conv((4, 4), channel_n => channel_n, relu, pad=(1, 1), stride=(2, 2)),
        Split(
            Chain(
                # Conv((1, 1), channel_n => div(channel_n, 4), relu),
                Flux.flatten,
                # Dropout(0.5),
                Dense(div(dim_1, 8) * div(dim_2, 8) * channel_n => channel_n * 8, relu),
                # Dropout(0.5),
                Dense(channel_n * 8 => channel_n * 8, relu),
                # Dropout(0.5),
                Dense(channel_n * 8 => latent_n), # mu : mean -- IMPORTANT : no activation function !!!
            ),
            Chain(
                # Conv((1, 1), channel_n => div(channel_n, 4), relu),
                Flux.flatten,
                # Dropout(0.5),
                Dense(div(dim_1, 8) * div(dim_2, 8) * channel_n => channel_n * 8, relu),
                # Dropout(0.5),
                Dense(channel_n * 8 => channel_n * 8, relu),
                # Dropout(0.5),
                Dense(channel_n * 8 => latent_n), # sigma: log_var -- IMPORTANT : no activation function !!!
            )
        ),
    )

    encoder = Chain(
        Flux.flatten,
        Dropout(0.5),
        Dense(dim_1 * dim_2 => channel_n * 8, relu),
        Dropout(0.5),
        Dense(channel_n * 8 => channel_n * 8, relu),
        Split(
            Chain(
                # Dropout(0.5),
                # Dense(channel_n * 8 => channel_n * 8, relu),
                Dropout(0.5),
                Dense(channel_n * 8 => latent_n),  # mu : mean -- IMPORTANT : no activation function !!!
            ),
            Chain(
                # Dropout(0.5),
                # Dense(channel_n * 8 => channel_n * 8, relu),
                Dropout(0.5),
                Dense(channel_n * 8 => latent_n),  # sigma : log_var -- IMPORTANT : no activation function !!!
            ),
        )
    )

    # encoder to GPU if available
    if args["model_cuda"] >= 0
        encoder = encoder |> gpu
    end

    @info "Encoder" encoder

    # returns a function that returns the decoder
    decoder = Chain(
        Dense(latent_n => channel_n * 8, elu),
        # Dropout(0.5),
        Dense(channel_n * 8 => channel_n * 8, elu),
        # Dropout(0.5),
        Dense(channel_n * 8 => div(dim_1, 8) * div(dim_2, 8) * channel_n, elu),
        # Dropout(0.5),
        # x -> reshape(x, (div(dim_1, 8), div(dim_2, 8), channel_n, :)),
        Reshape(div(dim_1, 8), div(dim_2, 8), channel_n, :),
        # ConvTranspose((1, 1), div(channel_n, 4) => channel_n, relu),
        ConvTranspose((4, 4), channel_n => channel_n, relu, pad=(1, 1), stride=(2, 2)),
        # Upsample((2, 2)),
        ConvTranspose((4, 4), channel_n => channel_n, relu, pad=(1, 1), stride=(2, 2)),
        # Upsample((2, 2)),
        ConvTranspose((4, 4), channel_n => 1, pad=(1, 1), sigmoid, stride=(2, 2)),
        # x -> reshape(x, (dim_1, dim_2, 1, :)),
        # sigmoid
    )

    decoder = Chain(
        Dense(latent_n => channel_n * 8, relu),
        Dropout(0.5),
        Dense(channel_n * 8 => channel_n * 8, relu),
        # Dense(channel_n * 8 => channel_n * 8, relu),
        Dropout(0.5),
        Dense(channel_n * 8 => dim_1 * dim_2, sigmoid),
        Reshape(dim_1, dim_2, 1, :),
        # sigmoid
    )

    # decoder to GPU if available
    if args["model_cuda"] >= 0
        decoder = decoder |> gpu
    end

    @info "Decoder" decoder

    # return a function that returns the model
    return encoder, decoder

end

encoder_, decoder_ = create_vae(32, 32, args["model_channel_n"], args["latent_n"])

# return a function that returns loss function
lossF = (encoder, decoder, x_) -> begin

    mu, log_var = encoder(x_)
    eps = randn(Float32, size(log_var))
    if args["model_cuda"] >= 0
        eps = eps |> gpu
    end
    z_ = mu .+ exp.(log_var .* 0.5f0) .* eps
    x_pred = decoder(z_)

    # x_pred, mu, log_var = model(x_)
    # x_softmax = softmax(x_, dims=1:2)
    loss_reconstruction = mean(sum((x_ - x_pred) .^ 2, dims=1:2)) # / (size(x_, 1) * size(x_, 2))
    # loss_reconstruction = mean(-sum(x_ .* log.(x_pred) .+ (1 .- x_) .* log.(1 .- x_pred), dims=1:2))
    loss_kl = mean(-0.5f0 * sum(1.0f0 .+ log_var .- mu .^ 2 .- exp.(log_var), dims=1))
    # loss_kl = 0
    loss = loss_reconstruction + loss_kl
    return loss # , loss_reconstruction, loss_kl
end

lossF_sample = (encoder, decoder, x_, size_::Int=2_000) -> begin
    len = size(x_)[end]
    if size_ > 0 && size_ <= len
        s = sample(1:len, size_, replace=false)
        x_ = x_[:, :, :, s]
    end
    if args["model_cuda"] >= 0
        x = Array{Float32,4}(undef, size(x_))
        x[:, :, :, 1:size_] = x_
        x_ = x |> gpu
        # x_ = x_ |> gpu
    end
    lossF(encoder, decoder, x_)
end

##################################################
# training
function train()

    # opt = ADAM(args["train_lr"])
    opt = AdamW(args["train_lr"], (0.9, 0.999), args["train_weight_decay"])
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end
    @info "Optimizer" opt

    params = Flux.params(encoder_, decoder_)
    @info "Params" sum(x -> length(x), params)

    BATCH_SIZE = args["train_batch_size"]

    data_loader = Flux.Data.DataLoader((x_train_, y_train_), batchsize=BATCH_SIZE, shuffle=true)

    function train_epoch(epoch::Int)
        # total loss
        total_loss = 0.0f0
        # progress tracker
        progress_tracker = Progress(length(data_loader), 1, "Training epoch $(epoch): ")
        for (x_, y_) in data_loader
            (() -> begin
                if args["model_cuda"] >= 0
                    x_ = x_ |> gpu
                end
                loss, back = pullback(params) do
                    lossF(encoder_, decoder_, x_)
                end
                gradients = back(1.0f0)
                Flux.update!(opt, params, gradients)
                next!(progress_tracker; showvalues=[(:loss, loss)])
                total_loss += loss
            end)()
            # reclaim GPU memory
            # if mod(i, BATCH_SIZE * 50) == 1
            #     # @show mean(lossVector) # size(lossVector)
            #     # lossVector = Vector{Float32}()
            #     GC.gc(true)
            #     if args["model_cuda"] >= 0
            #         CUDA.reclaim()
            #     end
            # end
        end
        return total_loss / length(data_loader)
    end

    start_time = time()
    loss_train = lossF_sample(encoder_, decoder_, x_train_)
    loss_test = lossF_sample(encoder_, decoder_, x_test_)
    loss_time = round(time() - start_time, digits=1)
    @info "Before training" loss_time loss_train loss_test
    # GC and reclaim GPU memory
    GC.gc(true)
    if args["model_cuda"] >= 0
        CUDA.reclaim()
    end

    for epoch in 1:args["train_epochs"]
        # start time
        start_time = time()
        # train epoch
        loss_epoch = train_epoch(epoch)
        train_time = round(time() - start_time, digits=1)
        # GC and reclaim GPU memory
        GC.gc(true)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end
        # calculate accuracy
        start_time = time()
        loss_test = lossF_sample(encoder_, decoder_, x_test_)
        loss_time = round(time() - start_time, digits=1)
        @info "[$(train_time)s] Train epoch [$(epoch)]" loss_time loss_epoch loss_test
        # GC and reclaim GPU memory
        GC.gc(true)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end
    end
end


##################################################
# Main
if abspath(PROGRAM_FILE) == @__FILE__
    train()
    # if args["model_cuda"] >= 0
    #     CUDA.reclaim()
    # end
end
