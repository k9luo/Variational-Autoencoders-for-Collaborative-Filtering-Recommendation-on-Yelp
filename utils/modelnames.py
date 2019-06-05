from models.vae import vae_cf, VAE
from models.nearest_neighbor import nearest_neighbor

models = {
    "VAE-CF": vae_cf,
    "NN": nearest_neighbor
}
