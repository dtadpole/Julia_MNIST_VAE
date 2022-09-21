using ArgParse
using CUDA

##################################################
# Parse command line arguments
function parse_commandline()

    s = ArgParseSettings()
    @add_arg_table s begin

        "--model_cuda"
        help = "model cuda number"
        arg_type = Int
        default = -1

        "--model_type"
        help = "model type : conv or dense"
        arg_type = String
        range_tester = x -> x in ["conv", "dense"]
        default = "dense"

        "--model_channel_n"
        help = "model channel number"
        arg_type = Int
        default = 32

        "--model_latent_n"
        help = "latent variable number"
        arg_type = Int
        default = 3

        "--train_batch_size"
        help = "batch size"
        arg_type = Int
        default = 100

        "--train_epochs"
        help = "train epochs"
        arg_type = Int
        default = 20

        "--train_lr"
        help = "learning rate"
        arg_type = Float32
        default = 0.0001f0

        "--train_weight_decay"
        help = "weight decay"
        arg_type = Float32
        default = 0.00001f0

    end

    return parse_args(s)
end

args = parse_commandline()

if args["model_cuda"] >= 0
    CUDA.allowscalar(false)
    CUDA.device!(args["model_cuda"])
end
