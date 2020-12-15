from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Lambda, Input, Dense
#from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow as tf

from keras.models import Sequential
from keras.layers import *
from keras import layers, models

def to_functional_model(seqmodel):
    input_layer = Input(batch_shape=seqmodel.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seqmodel.layers:
        prev_layer = layer(prev_layer)
    output_layer = prev_layer
    funcmodel = models.Model([input_layer], [prev_layer])
    return input_layer, output_layer, funcmodel

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

def get_vae_loss(image_input, values_input, image_decoded, values_decoded, z_mean, z_log_var, reconstruction_loss = "mse", impact_reconstruction_loss = 1):
    # VAE loss = mse_loss or binary_crossentropy + kl_loss
    if reconstruction_loss == "binary_crossentropy":
        img_reconstruction_loss = binary_crossentropy(image_input, image_decoded)
        values_reconstruction_loss = binary_crossentropy(value_input, values_decoded)

    elif reconstruction_loss == "mse":
        img_reconstruction_loss = mse(image_input, image_decoded)
        values_reconstruction_loss = mse(values_input, values_decoded)

    reconstruction_loss = K.mean(img_reconstruction_loss) + K.mean(values_reconstruction_loss) * 16#64

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(impact_reconstruction_loss*reconstruction_loss + kl_loss)
    return vae_loss

def get_value_encoder_model():
    value_model = Sequential()
    value_model.add(Flatten(input_shape=[130, 1, 1]))
    value_model.add(Dense(16))
    value_model.add(Activation('relu'))
    value_model.add(Dense(16))
    value_model.add(Activation('relu'))
    value_model.add(Dense(16))
    value_model.add(Activation('relu'))
    return value_model

def value_encoder():
    input = Input(shape=(130,), name='value_inputs')
    x = Dense(16, activation='relu')(input)
    x = Dense(16, activation='relu')(x)
    output = Dense(16, activation='relu')(x)
    return input, output

def image_encoder():
    image_model = Sequential()
    image_model.add(Conv2D(32, (8, 8), subsample=(4, 4), input_shape=[64, 128,1], activation = "relu"))
    image_model.add(Conv2D(64, (4, 4), subsample=(2, 2), activation = "relu"))
    image_model.add(Conv2D(64, (3, 3), subsample=(1, 1), activation = "relu"))
    image_model.add(Flatten())
    image_model.add(Dense(512, activation = "relu"))
    image_model.add(Dense(512, activation = "relu"))
    image_input, image_output, image_model = to_functional_model(image_model)
    return image_input, image_output

def encoder():
    image_input, image_output = image_encoder()
    value_input, value_output, = value_encoder()
    x = concatenate([image_output, value_output])
    z_mean = Dense(2, name='z_mean')(x)
    z_log_var = Dense(2, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(2,), name='z')([z_mean, z_log_var])
    return [image_input, value_input], [z_mean, z_log_var, z]

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

def image_decoder(encoded):
    shape = [-1,16,32, 1]
    x = Dense(512, activation = "relu")(encoded)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(1, (1, 1), activation='relu')(x)
    #x = layers.UpSampling2D((2, 2))(x)
    return x

def value_decoder(encoded):
    x = Dense(16, activation='relu')(encoded)
    x = Dense(16, activation='relu')(x)
    x = Dense(130, activation='relu')(x)
    return x

def decoder():
    latent_inputs = Input(shape=(2,), name='z_sampling')
    x = Dense(512+16, activation='relu')(latent_inputs)
    branch_values = Lambda( lambda x: tf.slice(x, (0, 0), (-1, 16)))(x)
    branch_image = Lambda( lambda x: tf.slice(x, (0, 16), (-1, 512)))(x)
    image_decoded = image_decoder(branch_image)
    values_decoded = value_decoder(branch_values)
    return latent_inputs, [image_decoded, values_decoded]
    #x = latent_inputs
    #for l in range(layers):
    #    x = Dense(intermediate_dim, activation='relu')(x)
    #outputs = Dense(original_dim, activation='sigmoid')(x)
    #return latent_inputs, outputs
