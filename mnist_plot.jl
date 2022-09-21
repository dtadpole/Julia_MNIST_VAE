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
                z_sample[k] = randn() * 0.5f0
            end
        end
        x_decoded = decoder(z_sample)
        x_decoded = reshape(x_decoded, size_, size_)
        canvas[(i-1)*size_+1:i*size_, (j-1)*size_+1:j*size_] = x_decoded
    end

    canvas = transpose(canvas)
    result = imshow(canvas)
    @info "imshow" result

    return canvas
end

##################################################
# Main
if abspath(PROGRAM_FILE) == @__FILE__

    model_type = args["model_type"]
    model_latent_n = args["model_latent_n"]
    model_channel_n = model_type == "dense" ? args["model_channel_n"] * 8 : args["model_channel_n"]

    encoder, decoder = load_model()
    for i in 1:args["plot_image_n"]
        img = plot_latent_images(decoder, 15)
        img_filename = "images/plot_$(model_latent_n)_$(model_type)_$(model_channel_n)__$(i).png"
        save(img_filename, colorview(Gray, img))
    end
    readline()
end
