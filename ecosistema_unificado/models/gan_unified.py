"""
Modelo GAN Unificado del Ecosistema de Redes Neuronales
======================================================

Implementación unificada de Generative Adversarial Networks que combina
las mejores características de ambos proyectos con funcionalidades avanzadas.

Características:
- GAN básica y avanzada (DCGAN, WGAN, etc.)
- Múltiples arquitecturas de generador y discriminador
- Técnicas de estabilización de entrenamiento
- Métricas avanzadas de evaluación
- Soporte para diferentes tipos de datos

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List
import os

class GANUnified:
    """
    Clase unificada para modelos GAN con funcionalidades avanzadas
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 latent_dim: int = 100,
                 gan_type: str = 'basic',
                 learning_rate: float = 0.0002,
                 beta_1: float = 0.5):
        """
        Inicializa el modelo GAN unificado
        
        Args:
            input_shape: Forma de entrada de los datos reales
            latent_dim: Dimensión del espacio latente
            gan_type: Tipo de GAN ('basic', 'dcgan', 'wgan', 'conditional')
            learning_rate: Tasa de aprendizaje
            beta_1: Parámetro beta_1 para Adam optimizer
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.gan_type = gan_type
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        
        # Modelos
        self.generator = None
        self.discriminator = None
        self.combined = None
        
        # Historial de entrenamiento
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'd_accuracy': []
        }
        
        self._build_models()
    
    def _build_models(self):
        """Construye los modelos según el tipo de GAN especificado"""
        if self.gan_type == 'basic':
            self.generator = self._build_basic_generator()
            self.discriminator = self._build_basic_discriminator()
        elif self.gan_type == 'dcgan':
            self.generator = self._build_dcgan_generator()
            self.discriminator = self._build_dcgan_discriminator()
        elif self.gan_type == 'wgan':
            self.generator = self._build_dcgan_generator()
            self.discriminator = self._build_wgan_discriminator()
        elif self.gan_type == 'conditional':
            self.generator = self._build_conditional_generator()
            self.discriminator = self._build_conditional_discriminator()
        else:
            raise ValueError(f"Tipo de GAN no soportado: {self.gan_type}")
        
        self._compile_models()
    
    def _build_basic_generator(self):
        """Construye un generador básico (fully connected)"""
        model = models.Sequential([
            layers.Dense(256, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(np.prod(self.input_shape), activation='tanh'),
            layers.Reshape(self.input_shape)
        ])
        
        noise = layers.Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return models.Model(noise, img, name='Generator')
    
    def _build_basic_discriminator(self):
        """Construye un discriminador básico (fully connected)"""
        model = models.Sequential([
            layers.Flatten(input_shape=self.input_shape),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        img = layers.Input(shape=self.input_shape)
        validity = model(img)
        
        return models.Model(img, validity, name='Discriminator')
    
    def _build_dcgan_generator(self):
        """Construye un generador tipo DCGAN (convolucional)"""
        # Calcular dimensiones iniciales
        init_size = self.input_shape[0] // 4
        
        model = models.Sequential([
            layers.Dense(128 * init_size * init_size, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((init_size, init_size, 128)),
            
            layers.UpSampling2D(),
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            
            layers.UpSampling2D(),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2D(self.input_shape[-1], kernel_size=3, padding='same'),
            layers.Activation('tanh')
        ])
        
        noise = layers.Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return models.Model(noise, img, name='DCGAN_Generator')
    
    def _build_dcgan_discriminator(self):
        """Construye un discriminador tipo DCGAN (convolucional)"""
        model = models.Sequential([
            layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.input_shape, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            layers.ZeroPadding2D(padding=((0,1),(0,1))),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        
        img = layers.Input(shape=self.input_shape)
        validity = model(img)
        
        return models.Model(img, validity, name='DCGAN_Discriminator')
    
    def _build_wgan_discriminator(self):
        """Construye un discriminador para WGAN (sin sigmoid final)"""
        model = models.Sequential([
            layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.input_shape, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Flatten(),
            layers.Dense(1)  # Sin activación para WGAN
        ])
        
        img = layers.Input(shape=self.input_shape)
        validity = model(img)
        
        return models.Model(img, validity, name='WGAN_Discriminator')
    
    def _build_conditional_generator(self, num_classes: int = 10):
        """Construye un generador condicional"""
        # Input de ruido
        noise = layers.Input(shape=(self.latent_dim,))
        
        # Input de etiqueta
        label = layers.Input(shape=(1,), dtype='int32')
        label_embedding = layers.Flatten()(layers.Embedding(num_classes, self.latent_dim)(label))
        
        # Combinar ruido y etiqueta
        model_input = layers.Multiply()([noise, label_embedding])
        
        # Generador
        x = layers.Dense(256)(model_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        x = layers.Dense(np.prod(self.input_shape), activation='tanh')(x)
        img = layers.Reshape(self.input_shape)(x)
        
        return models.Model([noise, label], img, name='Conditional_Generator')
    
    def _build_conditional_discriminator(self, num_classes: int = 10):
        """Construye un discriminador condicional"""
        # Input de imagen
        img = layers.Input(shape=self.input_shape)
        
        # Input de etiqueta
        label = layers.Input(shape=(1,), dtype='int32')
        label_embedding = layers.Flatten()(layers.Embedding(num_classes, np.prod(self.input_shape))(label))
        label_embedding = layers.Reshape(self.input_shape)(label_embedding)
        
        # Combinar imagen y etiqueta
        concatenated = layers.Concatenate()([img, label_embedding])
        
        # Discriminador
        x = layers.Flatten()(concatenated)
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        validity = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model([img, label], validity, name='Conditional_Discriminator')
    
    def _compile_models(self):
        """Compila los modelos"""
        # Optimizadores
        optimizer_g = optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        optimizer_d = optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        
        # Compilar discriminador
        if self.gan_type == 'wgan':
            self.discriminator.compile(
                loss=self._wasserstein_loss,
                optimizer=optimizer_d,
                metrics=['accuracy']
            )
        else:
            self.discriminator.compile(
                loss='binary_crossentropy',
                optimizer=optimizer_d,
                metrics=['accuracy']
            )
        
        # Modelo combinado (generador + discriminador)
        if self.gan_type == 'conditional':
            z = layers.Input(shape=(self.latent_dim,))
            label = layers.Input(shape=(1,), dtype='int32')
            img = self.generator([z, label])
            
            self.discriminator.trainable = False
            validity = self.discriminator([img, label])
            
            self.combined = models.Model([z, label], validity, name='Combined_Model')
        else:
            z = layers.Input(shape=(self.latent_dim,))
            img = self.generator(z)
            
            self.discriminator.trainable = False
            validity = self.discriminator(img)
            
            self.combined = models.Model(z, validity, name='Combined_Model')
        
        # Compilar modelo combinado
        if self.gan_type == 'wgan':
            self.combined.compile(loss=self._wasserstein_loss, optimizer=optimizer_g)
        else:
            self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_g)
    
    def _wasserstein_loss(self, y_true, y_pred):
        """Función de pérdida de Wasserstein para WGAN"""
        return tf.reduce_mean(y_true * y_pred)
    
    def train(self, 
              x_train: np.ndarray,
              y_train: Optional[np.ndarray] = None,
              epochs: int = 10000,
              batch_size: int = 32,
              sample_interval: int = 1000,
              save_dir: str = 'gan_samples') -> Dict[str, List[float]]:
        """
        Entrena el modelo GAN
        
        Args:
            x_train: Datos de entrenamiento
            y_train: Etiquetas (para GAN condicional)
            epochs: Número de épocas
            batch_size: Tamaño del batch
            sample_interval: Intervalo para guardar muestras
            save_dir: Directorio para guardar muestras
            
        Returns:
            Historial de entrenamiento
        """
        # Crear directorio para muestras
        os.makedirs(save_dir, exist_ok=True)
        
        # Normalizar datos a [-1, 1]
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        
        # Etiquetas adversariales
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Entrenar Discriminador
            # ---------------------
            
            # Seleccionar batch aleatorio de imágenes reales
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            
            # Generar ruido y crear imágenes falsas
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            if self.gan_type == 'conditional' and y_train is not None:
                labels = y_train[idx]
                gen_imgs = self.generator.predict([noise, labels], verbose=0)
                
                # Entrenar discriminador
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            else:
                gen_imgs = self.generator.predict(noise, verbose=0)
                
                # Entrenar discriminador
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Entrenar Generador
            # ---------------------
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            if self.gan_type == 'conditional' and y_train is not None:
                sampled_labels = np.random.randint(0, 10, (batch_size, 1))  # Asumiendo 10 clases
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            else:
                g_loss = self.combined.train_on_batch(noise, valid)
            
            # Guardar progreso
            self.training_history['g_loss'].append(g_loss)
            self.training_history['d_loss'].append(d_loss[0])
            self.training_history['d_accuracy'].append(d_loss[1])
            
            # Imprimir progreso
            if epoch % sample_interval == 0:
                print(f"Época {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
                self.save_samples(epoch, save_dir)
        
        return self.training_history
    
    def save_samples(self, epoch: int, save_dir: str, samples: int = 25):
        """Guarda muestras generadas"""
        r, c = 5, 5
        noise = np.random.normal(0, 1, (samples, self.latent_dim))
        
        if self.gan_type == 'conditional':
            sampled_labels = np.arange(0, samples).reshape(-1, 1)
            gen_imgs = self.generator.predict([noise, sampled_labels], verbose=0)
        else:
            gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescalar a [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                if len(self.input_shape) == 3 and self.input_shape[-1] == 1:
                    axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                else:
                    axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        
        fig.savefig(f"{save_dir}/gan_epoch_{epoch}.png")
        plt.close()
    
    def generate_samples(self, num_samples: int = 1, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Genera muestras usando el generador entrenado
        
        Args:
            num_samples: Número de muestras a generar
            labels: Etiquetas para generación condicional
            
        Returns:
            Muestras generadas
        """
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        
        if self.gan_type == 'conditional':
            if labels is None:
                labels = np.random.randint(0, 10, (num_samples, 1))
            gen_imgs = self.generator.predict([noise, labels], verbose=0)
        else:
            gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescalar a [0, 1]
        return 0.5 * gen_imgs + 0.5
    
    def save_models(self, filepath_prefix: str):
        """Guarda los modelos"""
        self.generator.save(f"{filepath_prefix}_generator.h5")
        self.discriminator.save(f"{filepath_prefix}_discriminator.h5")
        if self.combined:
            self.combined.save(f"{filepath_prefix}_combined.h5")
    
    def load_models(self, filepath_prefix: str):
        """Carga modelos guardados"""
        self.generator = keras.models.load_model(f"{filepath_prefix}_generator.h5")
        self.discriminator = keras.models.load_model(f"{filepath_prefix}_discriminator.h5")
        
        # Recompilar el modelo combinado
        self._compile_models()
    
    def get_model_summary(self) -> str:
        """Retorna un resumen de los modelos"""
        summary = "=== GENERADOR ===\n"
        if self.generator:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self.generator.summary()
            sys.stdout = old_stdout
            summary += buffer.getvalue()
        
        summary += "\n=== DISCRIMINADOR ===\n"
        if self.discriminator:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self.discriminator.summary()
            sys.stdout = old_stdout
            summary += buffer.getvalue()
        
        return summary