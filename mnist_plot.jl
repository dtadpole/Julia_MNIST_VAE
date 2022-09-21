include("./mnist_vae.jl")

using ImageView
using Images

function plot_latent_images(decoder, n, size_=32)
    # display a n*n 2D manifold of digits

    model_latent_n = args["model_latent_n"]

    normal = Normal()
    grid_x = quantile(normal, range(0.05f0, stop=0.95f0, length=n))
    grid_y = quantile(normal, range(0.05f0, stop=0.95f0, length=n))
    image_width = size_ * n
    image_height = size_ * n
    canvas = zeros(Float32, image_width, image_height)

    s = shuffle(1:model_latent_n)

    for i in 1:n, j in 1:n
        # z_sample = [grid_x[i], grid_y[j]]
        z_sample = zeros(Float32, model_latent_n)
        for k in 1:model_latent_n
            if s[k] == 1
                z_sample[k] = grid_x[i]
            elseif s[k] == 2
                z_sample[k] = grid_y[j]
            else
                z_sample[k] = randn()
            end
        end
        x_decoded = decoder(z_sample)
        x_decoded = reshape(x_decoded, size_, size_)
        canvas[(i-1)*size_+1:i*size_, (j-1)*size_+1:j*size_] = x_decoded
    end

    result = imshow(transpose(canvas))
    @info "imshow" result

end

##################################################
# Main
if abspath(PROGRAM_FILE) == @__FILE__
    encoder, decoder = load_model()
    for i in 1:args["plot_image_n"]
        plot_latent_images(decoder, 15)
    end
    readline()
end
