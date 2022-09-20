include("./args.jl")

using MLDatasets
using OneHotArrays

##################################################
# Training dataset
trainset = MNIST.traindata()

x_train_, y_train_ = trainset[:]
x_train_ = convert(Array{Float32}, reshape(x_train_, 28, 28, 1, :)) # / 255.0f0 - data already normalized to Float
y_train_ = convert(Array{Float32}, onehotbatch(y_train_, 0:9))
@info "Train data" typeof(x_train_) size(x_train_) typeof(y_train_) size(y_train_)

TRAIN_LENGTH = size(y_train_, 2)

##################################################
# Test dataset
testset = MNIST.testdata()

x_test_, y_test_ = testset[:]
x_test_ = convert(Array{Float32}, reshape(x_test_, 28, 28, 1, :)) # / 255.0f0 - data already normalized to Float
y_test_ = convert(Array{Float32}, onehotbatch(y_test_, 0:9))
@info "Test data" typeof(x_test_) size(x_test_) typeof(y_test_) size(y_test_)

TEST_LENGTH = size(y_test_, 2)

##################################################
