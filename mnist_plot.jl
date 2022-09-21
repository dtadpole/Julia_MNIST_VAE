include("./mnist_vae.jl")

function plot_latent_images(encoder, decoder, n, digit_size=32)
    # display a n*n 2D manifold of digits
    figure(figsize=(10, 10))
    grid("off")
    subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    linspace_x = range(-4, 4, length=n)
    linspace_y = range(-4, 4, length=n)

    # use the model to generate the digits
    for i in 1:n, j in 1:n
        z_sample = [linspace_x[i], linspace_y[j]]
        x_decoded = decoder(z_sample)
        digit = x_decoded[:, :, 1]
        imshow(digit, cmap="gray")
        axis("off")
    end
end

##################################################
# Main
if abspath(PROGRAM_FILE) == @__FILE__
end
