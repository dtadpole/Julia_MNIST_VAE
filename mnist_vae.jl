include("./data.jl")

using Flux
using CUDA
using Random
using StatsBase
using Distributions

LATENT_N = 8

##################################################
# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)


##################################################
# returns a function that returns the model
modelF = (dim_1::Int, dim_2::Int, channel_n::Int, latent_n::Int) -> begin

    # returns a function that returns the encoder
    encoder = Chain(
        Conv((3, 3), 1 => channel_n, relu, pad=(1, 1)),
        MaxPool((2, 2)),
        Conv((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        MaxPool((2, 2)),
        Conv((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        Flux.flatten,
        Split(
            Chain(
                # Dropout(0.5),
                Dense(div(dim_1, 4) * div(dim_2, 4) * channel_n => channel_n * 4, elu),
                # Dropout(0.5),
                Dense(channel_n * 4 => latent_n, tanh), # mu : mean
            ),
            Chain(
                # Dropout(0.5),
                Dense(div(dim_1, 4) * div(dim_2, 4) * channel_n => channel_n * 4, elu),
                # Dropout(0.5),
                Dense(channel_n * 4 => latent_n, elu), # log_var
                # softmax
            )
        ),
    )

    # encoder to GPU if available
    if args["model_cuda"] >= 0
        encoder = encoder |> gpu
    end

    @info "Encoder" encoder

    # multivariate normal distribution
    multivariate_normal = Distributions.MvNormal(zeros(Float32, latent_n), ones(Float32, latent_n))

    # multivariate_normal to GPU if available
    if args["model_cuda"] >= 0
        multivariate_normal = multivariate_normal |> gpu
    end

    # returns a function that returns the sampling function
    sampler = (mean, log_var) -> begin
        eps = rand(multivariate_normal)
        eps .* exp.(log_var ./ 2) .+ mean
    end

    # returns a function that returns the decoder
    decoder = Chain(
        Dense(latent_n => channel_n * 4, elu),
        # Dropout(0.5),
        Dense(channel_n * 4 => div(dim_1, 4) * div(dim_2, 4) * channel_n, elu),
        x -> reshape(x, (div(dim_1, 4), div(dim_2, 4), channel_n, :)),
        # Dropout(0.5),
        ConvTranspose((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        Upsample((2, 2)),
        ConvTranspose((3, 3), channel_n => channel_n, relu, pad=(1, 1)),
        Upsample((2, 2)),
        ConvTranspose((3, 3), channel_n => 1, relu, pad=(1, 1)),
        x -> reshape(x, (dim_1, dim_2, 1, :)),
        sigmoid
    )

    # decoder to GPU if available
    if args["model_cuda"] >= 0
        decoder = decoder |> gpu
    end

    @info "Decoder" decoder

    model = (x) -> begin
        mu, log_var = encoder(x)
        z = sampler(mu, log_var)
        y_ = decoder(z)
        return y_, mu, log_var
    end

    # return a function that returns the model
    return model

end

# return a function that returns loss function
lossF = (model_, x_) -> begin
    x_pred, mu, log_var = model_(x_)
    loss_reconstruction = sum((x_ - x_pred) .^ 2, dims=1:2) / (size(x_, 1) * size(x_, 2))
    # loss_kl = sum(log.(log_var / 1.0) + (1.0^2 + (mu - 0.0)^2) / (2 * log_var^2) - 0.5, dims=1)
    loss_kl = -0.5f0 * sum(1.0f0 .+ log_var .- mu .^ 2 .- exp.(log_var), dims=1)
    loss = mean(loss_reconstruction .+ loss_kl), mean(loss_reconstruction), mean(loss_kl)
end

lossF_sample = (model_, x_, size_::Int=10_000) -> begin
    len = size(x_)[end]
    if size_ > 0 && size_ <= len
        s = sample(1:len, size_, replace=false)
        x_ = x_[:, :, :, s]
    end
    lossF(model_, x_)
end

model_ = modelF(28, 28, args["model_channel_n"], LATENT_N)
@info "Model" model_

##################################################
# training
function train()

    # opt = ADAM(0.01)
    opt = AdamW(0.001, (0.9, 0.999), 0.0001)
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end

    params = Flux.params(model_)

    BATCH_SIZE = args["batch_size"]

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
                    x_ = x_ |> gpu
                end
                grads = gradient(() -> lossF(model_, x_)[1], params)
                Flux.update!(opt, params, grads)
                push!(lossVector, lossF(model_, x_)[1])
                # Flux.train!(lossF, params, x_, opt, cb=Flux.throttle(() -> @show lossF(model_, x_, x_), 10))
            end)()
            # reclaim GPU memory
            if mod(i, 2_000) == 1
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
    loss_train = lossF_sample(model_, x_train_)
    loss_test = lossF_sample(model_, x_test_)
    loss_time = round(time() - start_time, digits=1)
    @info "Before training" loss_time loss_train loss_test
    # GC and reclaim GPU memory
    GC.gc(true)
    if args["model_cuda"] >= 0
        CUDA.reclaim()
    end

    for epoch in 1:args["epochs"]
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
        loss_train = lossF_sample(model_, x_train_)
        loss_test = lossF_sample(model_, x_test_)
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
