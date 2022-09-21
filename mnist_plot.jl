include("./mnist_vae.jl")

using ImageView
using Images

function plot_latent_images(decoder, n, size_=32)
    # display a n*n 2D manifold of digits

    normal = Normal()
    grid_x = quantile(normal, range(0.05f0, stop=0.95f0, length=n))
    grid_y = quantile(normal, range(0.05f0, stop=0.95f0, length=n))
    image_width = size_ * n
    image_height = size_ * n
    canvas = zeros(Float32, image_width, image_height)

    for i in 1:n, j in 1:n
        z_sample = [grid_x[i], grid_y[j]]
        x_decoded = decoder(z_sample)
        x_decoded = reshape(x_decoded, size_, size_)
        canvas[(i-1)*size_+1:i*size_, (j-1)*size_+1:j*size_] = x_decoded
    end

    result = imshow(canvas)
    @info "imshow" result
    readline()

end

##################################################
# Main
if abspath(PROGRAM_FILE) == @__FILE__
    encoder, decoder = load_model()
    plot_latent_images(decoder, 15)
end
