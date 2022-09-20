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

        "--model_channel_n"
        help = "model channel number"
        arg_type = Int
        default = 32

        "--latent_n"
        help = "latent variable number"
        arg_type = Int
        default = 2

        "--train_batch_size"
        help = "batch size"
        arg_type = Int
        default = 50

        "--train_epochs"
        help = "train epochs"
        arg_type = Int
        default = 50

        "--train_lr"
        help = "learning rate"
        arg_type = Float32
        default = 0.0005f0

        "--train_weight_decay"
        help = "weight decay"
        arg_type = Float32
        default = 0.00005f0

    end

    return parse_args(s)
end

args = parse_commandline()

if args["model_cuda"] >= 0
    CUDA.allowscalar(false)
    CUDA.device!(args["model_cuda"])
end
