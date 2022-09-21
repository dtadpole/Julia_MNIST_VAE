include("./args.jl")

using MLDatasets
using OneHotArrays
using ImageFiltering
using Flux

##################################################
# Training dataset
trainset = MNIST.traindata()

x_train_, y_train_ = trainset[:]
x_train_ = convert(Array{Float32}, reshape(x_train_, 28, 28, 1, :)) # / 255.0f0 - data already normalized to Float
x_train_ = parent(padarray(x_train_, Fill(0.0f0, (2, 2, 0, 0))))
y_train_ = convert(Array{Float32}, onehotbatch(y_train_, 0:9))
@info "Train data" typeof(x_train_) size(x_train_) typeof(y_train_) size(y_train_)

TRAIN_LENGTH = size(y_train_, 2)

##################################################
# Test dataset
testset = MNIST.testdata()

x_test_, y_test_ = testset[:]
x_test_ = convert(Array{Float32}, reshape(x_test_, 28, 28, 1, :)) # / 255.0f0 - data already normalized to Float
x_test_ = parent(padarray(x_test_, Fill(0.0f0, (2, 2, 0, 0))))
y_test_ = convert(Array{Float32}, onehotbatch(y_test_, 0:9))
@info "Test data" typeof(x_test_) size(x_test_) typeof(y_test_) size(y_test_)

TEST_LENGTH = size(y_test_, 2)

##################################################
# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

##################################################
# We define a reshape layer to use in our decoder
struct Reshape
    shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()
