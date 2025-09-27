"""
Modelo VAE Unificado del Ecosistema de Redes Neuronales
======================================================

Implementación unificada de Variational Autoencoders.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class VAEUnified:
    """Clase unificada para modelos VAE"""
    
    def __init__(self, input_shape, latent_dim=64):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        self.encoder = None
        self.decoder = None
        self.vae = None
        
        self._build_models()
    
    def _build_models(self):
        """Construye encoder, decoder y VAE"""
        # Encoder
        encoder_inputs = keras.Input(shape=self.input_shape)
        x = layers.Flatten()(encoder_inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling
        z = layers.Lambda(self._sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(latent_inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(np.prod(self.input_shape), activation='sigmoid')(x)
        decoder_outputs = layers.Reshape(self.input_shape)(x)
        
        self.decoder = keras.Model(latent_inputs, decoder_outputs)
        
        # VAE completo
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = keras.Model(encoder_inputs, outputs)
        
        # Compilar con pérdida personalizada
        self.vae.add_loss(self._vae_loss(encoder_inputs, outputs, z_mean, z_log_var))
        self.vae.compile(optimizer='adam')
    
    def _sampling(self, args):
        """Función de muestreo para el espacio latente"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _vae_loss(self, inputs, outputs, z_mean, z_log_var):
        """Función de pérdida VAE (reconstrucción + KL divergence)"""
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(inputs, outputs), axis=(1, 2)
            )
        )
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        kl_loss *= -0.5
        return reconstruction_loss + kl_loss
    
    def train(self, x_train, epochs=50, batch_size=32):
        """Entrena el VAE"""
        return self.vae.fit(x_train, epochs=epochs, batch_size=batch_size)
    
    def generate(self, num_samples=1):
        """Genera nuevas muestras"""
        z_sample = np.random.normal(size=(num_samples, self.latent_dim))
        return self.decoder.predict(z_sample)