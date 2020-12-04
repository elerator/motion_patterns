from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_vae_loss(inputs, outputs, z_mean, z_log_var, reconstruction_loss = "mse", original_dim = 130, impact_reconstruction_loss = 1):
    # VAE loss = mse_loss or binary_crossentropy + kl_loss
    if reconstruction_loss == "binary_crossentropy":
        reconstruction_loss = binary_crossentropy(inputs,outputs)
    elif reconstruction_loss == "mse":
        reconstruction_loss = mse(inputs, outputs)

    reconstruction_loss *= original_dim
    
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(impact_reconstruction_loss*reconstruction_loss + kl_loss)
    return vae_loss


def mlp_encoder(input_shape, latent_dim = 2, intermediate_dim = 512, layers = 3):
    # Encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for l in range(layers):
        x = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    return inputs, [z_mean, z_log_var, z]

def mlp_decoder(original_dim, latent_dim = 2, intermediate_dim = 512, layers = 3):
    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs
    for l in range(layers):
        x = Dense(intermediate_dim, activation='relu')(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    return latent_inputs, outputs
