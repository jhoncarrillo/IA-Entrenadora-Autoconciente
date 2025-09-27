"""
Variational Autoencoder (VAE) Avanzado para el Ecosistema Autónomo
Implementa VAE con capacidades de generación y representación latente
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import math

logger = logging.getLogger(__name__)

class VAEEncoder(nn.Module):
    """Encoder del VAE que mapea datos a distribución latente"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, 
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Construir capas del encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Capas para media y varianza
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        
        # Parámetros de la distribución latente
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    """Decoder del VAE que reconstruye datos desde el espacio latente"""
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.2, output_activation: str = 'sigmoid'):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.output_activation = output_activation
        
        # Construir capas del decoder (orden inverso al encoder)
        layers = []
        prev_dim = latent_dim
        
        # Invertir las dimensiones ocultas
        hidden_dims_reversed = list(reversed(hidden_dims))
        
        for hidden_dim in hidden_dims_reversed:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Activación de salida
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'relu':
            layers.append(nn.ReLU())
        # 'none' o cualquier otro valor no agrega activación
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)

class ConvVAEEncoder(nn.Module):
    """Encoder convolucional para datos de imagen"""
    
    def __init__(self, input_channels: int, input_size: int, latent_dim: int):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Segunda capa convolucional
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Tercera capa convolucional
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Cuarta capa convolucional
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calcular dimensión después de convoluciones
        self.conv_output_size = self._get_conv_output_size()
        
        # Capas fully connected
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
        
    def _get_conv_output_size(self):
        """Calcular el tamaño de salida después de las convoluciones"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def forward(self, x):
        # Aplicar convoluciones
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        # Parámetros de la distribución latente
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class ConvVAEDecoder(nn.Module):
    """Decoder convolucional para datos de imagen"""
    
    def __init__(self, latent_dim: int, output_channels: int, output_size: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Calcular dimensiones iniciales
        self.init_size = output_size // 16  # Después de 4 upsampling de factor 2
        self.init_channels = 256
        
        # Capa fully connected inicial
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_size * self.init_size)
        
        # Capas de deconvolución
        self.deconv_layers = nn.Sequential(
            # Primera deconvolución
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Segunda deconvolución
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Tercera deconvolución
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Capa de salida
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        # Expandir desde latente a feature maps
        h = self.fc(z)
        h = h.view(h.size(0), self.init_channels, self.init_size, self.init_size)
        
        # Aplicar deconvoluciones
        return self.deconv_layers(h)

class AdvancedVAE(nn.Module):
    """VAE Avanzado con múltiples configuraciones"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.data_type = config.get('data_type', 'tabular')  # 'tabular' o 'image'
        self.latent_dim = config.get('latent_dim', 64)
        self.beta = config.get('beta', 1.0)  # Para β-VAE
        
        if self.data_type == 'tabular':
            self.input_dim = config['input_dim']
            hidden_dims = config.get('hidden_dims', [512, 256, 128])
            dropout = config.get('dropout', 0.2)
            output_activation = config.get('output_activation', 'sigmoid')
            
            self.encoder = VAEEncoder(self.input_dim, hidden_dims, self.latent_dim, dropout)
            self.decoder = VAEDecoder(self.latent_dim, hidden_dims, self.input_dim, 
                                    dropout, output_activation)
            
        elif self.data_type == 'image':
            self.input_channels = config['input_channels']
            self.input_size = config['input_size']
            
            self.encoder = ConvVAEEncoder(self.input_channels, self.input_size, self.latent_dim)
            self.decoder = ConvVAEDecoder(self.latent_dim, self.input_channels, self.input_size)
        
        else:
            raise ValueError(f"Tipo de datos no soportado: {self.data_type}")
    
    def reparameterize(self, mu, logvar):
        """Truco de reparametrización"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encoding
        mu, logvar = self.encoder(x)
        
        # Reparametrización
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        x_recon = self.decoder(z)
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def generate(self, num_samples: int, device: torch.device = None):
        """Generar nuevas muestras"""
        if device is None:
            device = next(self.parameters()).device
        
        # Muestrear desde distribución normal estándar
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decodificar
        with torch.no_grad():
            generated = self.decoder(z)
        
        return generated
    
    def interpolate(self, x1, x2, num_steps: int = 10):
        """Interpolar entre dos puntos en el espacio latente"""
        with torch.no_grad():
            # Codificar ambos puntos
            mu1, _ = self.encoder(x1)
            mu2, _ = self.encoder(x2)
            
            # Crear interpolación lineal
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decoder(z_interp)
                interpolations.append(x_interp)
            
            return torch.stack(interpolations)

class BetaVAE(AdvancedVAE):
    """β-VAE para control de disentanglement"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.beta = config.get('beta', 4.0)  # β > 1 para mayor disentanglement

class ConditionalVAE(AdvancedVAE):
    """VAE Condicional que puede generar basado en etiquetas"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_classes = config['num_classes']
        self.condition_dim = config.get('condition_dim', self.num_classes)
        
        # Modificar encoder para incluir condición
        if self.data_type == 'tabular':
            # Agregar dimensión de condición al input
            original_input_dim = self.encoder.input_dim
            self.encoder.encoder[0] = nn.Linear(
                original_input_dim + self.condition_dim, 
                self.encoder.encoder[0].out_features
            )
        
        # Modificar decoder para incluir condición
        if self.data_type == 'tabular':
            original_latent_dim = self.decoder.latent_dim
            self.decoder.decoder[0] = nn.Linear(
                original_latent_dim + self.condition_dim,
                self.decoder.decoder[0].out_features
            )
    
    def forward(self, x, condition):
        # Concatenar condición con input
        if self.data_type == 'tabular':
            x_cond = torch.cat([x, condition], dim=1)
        else:
            # Para imágenes, se puede usar concatenación en el espacio latente
            x_cond = x
        
        # Encoding
        mu, logvar = self.encoder(x_cond)
        z = self.reparameterize(mu, logvar)
        
        # Concatenar condición con latente para decoding
        z_cond = torch.cat([z, condition], dim=1)
        
        # Decoding
        x_recon = self.decoder(z_cond)
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def generate_conditional(self, condition, num_samples: int = 1, device: torch.device = None):
        """Generar muestras condicionadas"""
        if device is None:
            device = next(self.parameters()).device
        
        # Expandir condición para múltiples muestras
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        condition = condition.repeat(num_samples, 1)
        
        # Muestrear latente
        z = torch.randn(num_samples, self.latent_dim, device=device)
        z_cond = torch.cat([z, condition], dim=1)
        
        # Generar
        with torch.no_grad():
            generated = self.decoder(z_cond)
        
        return generated

def vae_loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
    """Función de pérdida para VAE"""
    
    # Pérdida de reconstrucción
    if len(x.shape) > 2:  # Imágenes
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:  # Datos tabulares
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # Pérdida KL
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Pérdida total
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': recon_loss,
        'kl_loss': kl_loss
    }

def create_vae(data_type: str = 'tabular', **kwargs) -> AdvancedVAE:
    """Factory function para crear VAE"""
    
    if data_type == 'tabular':
        default_config = {
            'data_type': 'tabular',
            'input_dim': 784,  # Por defecto para MNIST
            'latent_dim': 64,
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'output_activation': 'sigmoid',
            'beta': 1.0
        }
    elif data_type == 'image':
        default_config = {
            'data_type': 'image',
            'input_channels': 1,
            'input_size': 28,
            'latent_dim': 64,
            'beta': 1.0
        }
    else:
        raise ValueError(f"Tipo de datos no soportado: {data_type}")
    
    # Actualizar configuración
    config = {**default_config, **kwargs}
    
    # Determinar tipo de VAE
    if 'num_classes' in config:
        return ConditionalVAE(config)
    elif config.get('beta', 1.0) != 1.0:
        return BetaVAE(config)
    else:
        return AdvancedVAE(config)

def create_adaptive_vae(data_characteristics: Dict[str, Any]) -> AdvancedVAE:
    """Crear VAE adaptado a las características de los datos"""
    
    data_type = data_characteristics.get('data_type', 'tabular')
    complexity = data_characteristics.get('complexity_score', 0.5)
    
    if data_type == 'tabular':
        input_dim = data_characteristics.get('feature_count', 784)
        
        # Configuración adaptativa basada en complejidad
        if complexity < 0.3:
            config = {
                'latent_dim': 32,
                'hidden_dims': [256, 128],
                'dropout': 0.1
            }
        elif complexity < 0.7:
            config = {
                'latent_dim': 64,
                'hidden_dims': [512, 256, 128],
                'dropout': 0.2
            }
        else:
            config = {
                'latent_dim': 128,
                'hidden_dims': [1024, 512, 256, 128],
                'dropout': 0.3
            }
        
        config.update({
            'data_type': 'tabular',
            'input_dim': input_dim
        })
        
    elif data_type == 'image':
        input_size = data_characteristics.get('image_size', 28)
        input_channels = data_characteristics.get('channels', 1)
        
        config = {
            'data_type': 'image',
            'input_channels': input_channels,
            'input_size': input_size,
            'latent_dim': 64 if complexity < 0.5 else 128
        }
    
    # Agregar información de clases si está disponible
    if 'num_classes' in data_characteristics:
        config['num_classes'] = data_characteristics['num_classes']
    
    return create_vae(**config)

if __name__ == "__main__":
    # Ejemplo de uso
    print("🧠 Creando VAE Avanzado...")
    
    # VAE para datos tabulares
    tabular_vae = create_vae('tabular', input_dim=100, latent_dim=32)
    print(f"VAE tabular creado: {tabular_vae}")
    
    # VAE para imágenes
    image_vae = create_vae('image', input_channels=3, input_size=64, latent_dim=128)
    print(f"VAE de imagen creado: {image_vae}")
    
    # VAE condicional
    conditional_vae = create_vae('tabular', input_dim=784, num_classes=10)
    print(f"VAE condicional creado: {conditional_vae}")
    
    # β-VAE
    beta_vae = create_vae('tabular', input_dim=784, beta=4.0)
    print(f"β-VAE creado: {beta_vae}")