include("./data.jl")

using Flux
using CUDA
using Random
using StatsBase
using Distributions

##################################################
# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)


# multivariate normal distribution
multivariate_normal = Distributions.MvNormal(zeros(Float32, args["latent_n"]), ones(Float32, args["latent_n"]))
normal = Distributions.Normal(0.0f0, 1.0f0)
# multivariate_normal to GPU if available
# if args["model_cuda"] >= 0
#     multivariate_normal = multivariate_normal |> gpu
# end



##################################################
# returns a function that returns the model
modelF = (dim_1::Int, dim_2::Int, channel_n::Int, latent_n::Int) -> begin

    # returns a function that returns the encoder
    encoder = Chain(
        Conv((3, 3), 1 => channel_n, relu, pad=(1, 1)),
        # MaxPool((2, 2)),
        Conv((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        # MaxPool((2, 2)),
        Conv((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        Split(
            Chain(
                Conv((1, 1), channel_n => div(channel_n, 4), relu),
                Flux.flatten,
                # Dropout(0.5),
                Dense(div(dim_1, 1) * div(dim_2, 1) * div(channel_n, 4) => channel_n * 2, relu),
                # Dropout(0.5),
                # Dense(div(dim_1, 2) * div(dim_2, 2) * channel_n => channel_n * 2, elu),
                # Dropout(0.5),
                Dense(channel_n * 2 => latent_n, relu), # mu : mean
            ),
            Chain(
                Conv((1, 1), channel_n => div(channel_n, 4), relu),
                Flux.flatten,
                # Dropout(0.5),
                Dense(div(dim_1, 1) * div(dim_2, 1) * div(channel_n, 4) => channel_n * 2, relu),
                # Dropout(0.5),
                # Dense(div(dim_1, 2) * div(dim_2, 2) * channel_n => channel_n * 2, elu),
                # Dropout(0.5),
                Dense(channel_n * 2 => latent_n, relu), # sigma: log_var
            )
        ),
    )

    encoder = Chain(
        Flux.flatten,
        Dense(dim_1 * dim_2 => channel_n * 16, relu),
        Split(
            Chain(
                Dense(channel_n * 16 => latent_n, relu),  # mu
            ),
            Chain(
                Dense(channel_n * 16 => latent_n, relu),  # sigma
            ),
        )
    )

    # encoder to GPU if available
    if args["model_cuda"] >= 0
        encoder = encoder |> gpu
    end

    @info "Encoder" encoder

    # returns a function that returns the sampling function
    sampling = (mu, log_var) -> begin
        # eps = rand(multivariate_normal, size(mu)[end])
        eps = randn(Float32, size(mu))
        # eps = rand(normal, size(mu))
        # @show size(eps) size(mu) size(log_var)
        if args["model_cuda"] >= 0
            eps = eps |> gpu
        end
        return mu .+ exp.(log_var .* 0.5f0) .* eps
    end

    # returns a function that returns the decoder
    decoder = Chain(
        Dense(latent_n => channel_n * 2, relu),
        # Dropout(0.5),
        Dense(channel_n * 2 => div(dim_1, 1) * div(dim_2, 1) * div(channel_n, 4), relu),
        # Dropout(0.5),
        x -> reshape(x, (div(dim_1, 1), div(dim_2, 1), div(channel_n, 4), :)),
        ConvTranspose((1, 1), div(channel_n, 4) => channel_n, relu),
        ConvTranspose((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        # Upsample((2, 2)),
        ConvTranspose((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        # Upsample((2, 2)),
        ConvTranspose((3, 3), channel_n => 1, relu, pad=(1, 1)),
        x -> reshape(x, (dim_1, dim_2, 1, :)),
        sigmoid
    )

    decoder = Chain(
        Dense(latent_n => channel_n * 16, relu),
        Dense(channel_n * 16 => dim_1 * dim_2, relu),
        x -> reshape(x, (dim_1, dim_2, 1, :)),
        sigmoid
    )

    # decoder to GPU if available
    if args["model_cuda"] >= 0
        decoder = decoder |> gpu
    end

    @info "Decoder" decoder

    # return a function that returns the model
    return encoder, decoder

end

encoder_, decoder_ = modelF(28, 28, args["model_channel_n"], args["latent_n"])

# model_ = (x) -> begin
#     mu, log_var = encoder_(x)
#     z = sampling_(mu, log_var)
#     y_ = decoder_(z)
#     return y_, mu, log_var
# end
# @info "Model" model_

# return a function that returns loss function
lossF = (encoder, decoder, x_) -> begin

    mu, log_var = encoder(x_)
    eps = randn(Float32, size(mu)) |> gpu
    # if args["model_cuda"] >= 0
    #     eps = eps |> gpu
    # end
    z_ = mu .+ exp.(log_var .* 0.5f0) .* eps
    x_pred = decoder(z_)

    # x_pred, mu, log_var = model(x_)
    # x_softmax = softmax(x_, dims=1:2)
    # loss_reconstruction = mean(sum((x_ - x_pred) .^ 2, dims=1:2)) # / (size(x_, 1) * size(x_, 2))
    loss_reconstruction = mean(-sum(x_ .* log.(x_pred) .+ (1 .- x_) .* log.(1 .- x_pred), dims=1:2))
    # loss_reconstruction = mean(-sum(x_softmax .* log.(x_softmax) - x_softmax .* log.(x_pred), dims=1:2))
    # loss_kl = sum(log.(log_var / 1.0) + (1.0^2 + (mu - 0.0)^2) / (2 * log_var^2) - 0.5, dims=1)
    loss_kl = mean(-0.5f0 * sum(1.0f0 .+ log_var .- mu .^ 2 .- exp.(log_var), dims=1))
    # loss_kl = 0
    loss = loss_reconstruction + loss_kl
    return loss, loss_reconstruction, loss_kl
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

    opt = ADAM(args["train_lr"])
    # opt = AdamW(args["train_lr"], (0.9, 0.999), args["train_weight_decay"])
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end
    @info "Optimizer" opt

    params = Flux.params(encoder_, decoder_)
    # @show params

    BATCH_SIZE = args["train_batch_size"]

    function train_epoch()
        lossVector = Vector{Float32}()
        # shuffle training data
        s = shuffle(1:TRAIN_LENGTH) # s = 1:len_train
        x_train_s = x_train_[:, :, :, s]
        # train batch
        for i in 1:BATCH_SIZE:TRAIN_LENGTH
            (() -> begin
                x_ = x_train_s[:, :, :, i:i+BATCH_SIZE-1]
                if args["model_cuda"] >= 0
                    # x = Array{Float32,4}(undef, size(x_))
                    # x[:, :, :, 1:BATCH_SIZE] = x_
                    # x_ = x |> gpu
                    x_ = x_ |> gpu
                end
                grads = gradient(() -> lossF(encoder_, decoder_, x_)[1], params)
                Flux.update!(opt, params, grads)
                push!(lossVector, lossF(encoder_, decoder_, x_)[1])
                # Flux.train!(lossF, params, x_, opt, cb=Flux.throttle(() -> @show lossF(model_, x_, x_), 10))
            end)()
            # reclaim GPU memory
            if mod(i, BATCH_SIZE * 50) == 1
                @show mean(lossVector)
                lossVector = Vector{Float32}()
                GC.gc(true)
                if args["model_cuda"] >= 0
                    CUDA.reclaim()
                end
            end
        end
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
        train_epoch()
        train_time = round(time() - start_time, digits=1)
        # GC and reclaim GPU memory
        GC.gc(true)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end
        # calculate accuracy
        start_time = time()
        loss_train = lossF_sample(encoder_, decoder_, x_train_)
        loss_test = lossF_sample(encoder_, decoder_, x_test_)
        loss_time = round(time() - start_time, digits=1)
        @info "[$(train_time)s] Train epoch [$(epoch)]" loss_time loss_train loss_test
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
