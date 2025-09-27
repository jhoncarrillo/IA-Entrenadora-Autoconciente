"""
Modelos GAN Avanzados para el Ecosistema Autónomo
Implementa múltiples arquitecturas GAN con entrenamiento estable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import math

logger = logging.getLogger(__name__)

class SpectralNorm(nn.Module):
    """Normalización espectral para estabilizar entrenamiento GAN"""
    
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data))
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def spectral_norm(module, name='weight', power_iterations=1):
    """Aplicar normalización espectral a un módulo"""
    SpectralNorm.apply(module, name, power_iterations)
    return module

class SelfAttention(nn.Module):
    """Módulo de auto-atención para GANs"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calcular query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        
        # Atención
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Aplicar atención
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Conexión residual con parámetro aprendible
        out = self.gamma * out + x
        
        return out

class Generator(nn.Module):
    """Generador avanzado con auto-atención y normalización espectral"""
    
    def __init__(self, latent_dim: int = 128, output_channels: int = 3,
                 features: int = 64, image_size: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calcular número de upsampling layers necesarias
        self.num_layers = int(math.log2(image_size)) - 2  # -2 porque empezamos en 4x4
        
        # Capa inicial
        self.initial = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(latent_dim, features * 16, 4, 1, 0)),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True)
        )
        
        # Capas de upsampling
        self.layers = nn.ModuleList()
        in_features = features * 16
        
        for i in range(self.num_layers):
            out_features = in_features // 2
            
            layer = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(in_features, out_features, 4, 2, 1)),
                nn.BatchNorm2d(out_features),
                nn.ReLU(True)
            )
            self.layers.append(layer)
            
            # Agregar auto-atención en capas intermedias
            if i == self.num_layers // 2:
                self.layers.append(SelfAttention(out_features))
            
            in_features = out_features
        
        # Capa final
        self.final = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_features, output_channels, 4, 2, 1)),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.initial(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    """Discriminador avanzado con auto-atención y normalización espectral"""
    
    def __init__(self, input_channels: int = 3, features: int = 64,
                 image_size: int = 64):
        super().__init__()
        
        self.image_size = image_size
        
        # Calcular número de downsampling layers
        self.num_layers = int(math.log2(image_size)) - 2
        
        # Capa inicial
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Capas de downsampling
        self.layers = nn.ModuleList()
        in_features = features
        
        for i in range(self.num_layers):
            out_features = min(in_features * 2, 512)
            
            layer = nn.Sequential(
                spectral_norm(nn.Conv2d(in_features, out_features, 4, 2, 1)),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layers.append(layer)
            
            # Agregar auto-atención en capas intermedias
            if i == self.num_layers // 2:
                self.layers.append(SelfAttention(out_features))
            
            in_features = out_features
        
        # Capa final
        self.final = spectral_norm(nn.Conv2d(in_features, 1, 4, 1, 0))
        
    def forward(self, x):
        x = self.initial(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final(x)
        return x.view(x.size(0), -1)

class WGAN_GP(nn.Module):
    """Wasserstein GAN con Gradient Penalty"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.latent_dim = config.get('latent_dim', 128)
        self.image_size = config.get('image_size', 64)
        self.channels = config.get('channels', 3)
        self.lambda_gp = config.get('lambda_gp', 10.0)
        
        # Crear generador y discriminador
        self.generator = Generator(
            latent_dim=self.latent_dim,
            output_channels=self.channels,
            features=config.get('g_features', 64),
            image_size=self.image_size
        )
        
        self.discriminator = Discriminator(
            input_channels=self.channels,
            features=config.get('d_features', 64),
            image_size=self.image_size
        )
        
    def gradient_penalty(self, real_samples, fake_samples, device):
        """Calcular gradient penalty para WGAN-GP"""
        batch_size = real_samples.size(0)
        
        # Muestrear epsilon aleatorio
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
        epsilon = epsilon.expand_as(real_samples)
        
        # Interpolar entre muestras reales y falsas
        interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
        interpolated = interpolated.requires_grad_(True)
        
        # Calcular probabilidades para muestras interpoladas
        prob_interpolated = self.discriminator(interpolated)
        
        # Calcular gradientes
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Calcular penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def generator_loss(self, fake_samples):
        """Pérdida del generador"""
        fake_validity = self.discriminator(fake_samples)
        return -torch.mean(fake_validity)
    
    def discriminator_loss(self, real_samples, fake_samples):
        """Pérdida del discriminador con gradient penalty"""
        device = real_samples.device
        
        # Validez de muestras reales y falsas
        real_validity = self.discriminator(real_samples)
        fake_validity = self.discriminator(fake_samples.detach())
        
        # Gradient penalty
        gp = self.gradient_penalty(real_samples, fake_samples, device)
        
        # Pérdida total del discriminador
        d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + self.lambda_gp * gp
        
        return d_loss, gp

class StyleGAN2Generator(nn.Module):
    """Generador inspirado en StyleGAN2 (versión simplificada)"""
    
    def __init__(self, latent_dim: int = 512, style_dim: int = 512,
                 output_channels: int = 3, image_size: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.image_size = image_size
        
        # Mapping network
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim)
        )
        
        # Synthesis network (simplificado)
        self.num_layers = int(math.log2(image_size)) - 1
        
        # Constante inicial
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Capas de síntesis
        self.synthesis_layers = nn.ModuleList()
        in_channels = 512
        
        for i in range(self.num_layers):
            out_channels = min(512, 512 // (2 ** (i // 2)))
            
            # Capa de convolución modulada por estilo (simplificada)
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2)
            )
            self.synthesis_layers.append(layer)
            in_channels = out_channels
        
        # Capa de salida
        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, 1),
            nn.Tanh()
        )
        
    def forward(self, z, truncation_psi=1.0):
        # Mapping network
        w = self.mapping(z)
        
        # Truncation trick
        if truncation_psi < 1.0:
            w = truncation_psi * w
        
        # Synthesis
        x = self.const.repeat(z.size(0), 1, 1, 1)
        
        for layer in self.synthesis_layers:
            x = layer(x)
        
        x = self.to_rgb(x)
        return x

class ConditionalGAN(nn.Module):
    """GAN Condicional para generación controlada"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.latent_dim = config.get('latent_dim', 128)
        self.num_classes = config['num_classes']
        self.image_size = config.get('image_size', 64)
        self.channels = config.get('channels', 3)
        
        # Embedding para clases
        self.class_embedding = nn.Embedding(self.num_classes, self.latent_dim)
        
        # Generador condicional
        self.generator = Generator(
            latent_dim=self.latent_dim + self.latent_dim,  # z + class embedding
            output_channels=self.channels,
            image_size=self.image_size
        )
        
        # Discriminador condicional
        self.discriminator = Discriminator(
            input_channels=self.channels + self.num_classes,  # image + one-hot
            image_size=self.image_size
        )
        
    def forward(self, z, labels):
        # Embedding de clase
        class_emb = self.class_embedding(labels)
        
        # Concatenar ruido con embedding de clase
        z_cond = torch.cat([z, class_emb], dim=1)
        
        # Generar imagen
        fake_images = self.generator(z_cond)
        
        return fake_images
    
    def discriminator_forward(self, images, labels):
        # One-hot encoding de labels
        batch_size = images.size(0)
        one_hot = torch.zeros(batch_size, self.num_classes, self.image_size, self.image_size)
        one_hot = one_hot.to(images.device)
        
        for i in range(batch_size):
            one_hot[i, labels[i]] = 1
        
        # Concatenar imagen con one-hot
        x_cond = torch.cat([images, one_hot], dim=1)
        
        return self.discriminator(x_cond)

class CycleGAN(nn.Module):
    """CycleGAN para traducción imagen-a-imagen sin pares"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.channels = config.get('channels', 3)
        self.image_size = config.get('image_size', 64)
        self.lambda_cycle = config.get('lambda_cycle', 10.0)
        self.lambda_identity = config.get('lambda_identity', 0.5)
        
        # Generadores A->B y B->A
        self.gen_AB = Generator(
            latent_dim=self.channels,  # Input es imagen, no ruido
            output_channels=self.channels,
            image_size=self.image_size
        )
        
        self.gen_BA = Generator(
            latent_dim=self.channels,
            output_channels=self.channels,
            image_size=self.image_size
        )
        
        # Discriminadores para A y B
        self.disc_A = Discriminator(
            input_channels=self.channels,
            image_size=self.image_size
        )
        
        self.disc_B = Discriminator(
            input_channels=self.channels,
            image_size=self.image_size
        )
    
    def forward(self, real_A, real_B):
        # Generación
        fake_B = self.gen_AB(real_A)
        fake_A = self.gen_BA(real_B)
        
        # Cycle consistency
        cycle_A = self.gen_BA(fake_B)
        cycle_B = self.gen_AB(fake_A)
        
        # Identity mapping
        identity_A = self.gen_BA(real_A)
        identity_B = self.gen_AB(real_B)
        
        return {
            'fake_A': fake_A,
            'fake_B': fake_B,
            'cycle_A': cycle_A,
            'cycle_B': cycle_B,
            'identity_A': identity_A,
            'identity_B': identity_B
        }

def adversarial_loss(predictions, target_is_real):
    """Pérdida adversarial estándar"""
    if target_is_real:
        target = torch.ones_like(predictions)
    else:
        target = torch.zeros_like(predictions)
    
    return F.binary_cross_entropy_with_logits(predictions, target)

def cycle_consistency_loss(real_images, cycled_images):
    """Pérdida de consistencia cíclica"""
    return F.l1_loss(cycled_images, real_images)

def identity_loss(real_images, identity_images):
    """Pérdida de identidad"""
    return F.l1_loss(identity_images, real_images)

def create_gan(model_type: str = 'wgan_gp', **kwargs) -> nn.Module:
    """Factory function para crear modelos GAN"""
    
    default_config = {
        'latent_dim': 128,
        'image_size': 64,
        'channels': 3,
        'g_features': 64,
        'd_features': 64
    }
    
    # Actualizar configuración
    config = {**default_config, **kwargs}
    
    if model_type == 'wgan_gp':
        return WGAN_GP(config)
    elif model_type == 'conditional':
        if 'num_classes' not in config:
            raise ValueError("num_classes required for conditional GAN")
        return ConditionalGAN(config)
    elif model_type == 'cycle':
        return CycleGAN(config)
    elif model_type == 'stylegan2':
        return StyleGAN2Generator(
            latent_dim=config.get('latent_dim', 512),
            style_dim=config.get('style_dim', 512),
            output_channels=config['channels'],
            image_size=config['image_size']
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")

def create_adaptive_gan(data_characteristics: Dict[str, Any]) -> nn.Module:
    """Crear GAN adaptado a las características de los datos"""
    
    image_size = data_characteristics.get('image_size', 64)
    channels = data_characteristics.get('channels', 3)
    complexity = data_characteristics.get('complexity_score', 0.5)
    
    # Configuración adaptativa
    if complexity < 0.3:
        config = {
            'latent_dim': 64,
            'g_features': 32,
            'd_features': 32
        }
    elif complexity < 0.7:
        config = {
            'latent_dim': 128,
            'g_features': 64,
            'd_features': 64
        }
    else:
        config = {
            'latent_dim': 256,
            'g_features': 128,
            'd_features': 128
        }
    
    config.update({
        'image_size': image_size,
        'channels': channels
    })
    
    # Seleccionar tipo de GAN
    if 'num_classes' in data_characteristics:
        config['num_classes'] = data_characteristics['num_classes']
        return create_gan('conditional', **config)
    elif data_characteristics.get('has_paired_domains', False):
        return create_gan('cycle', **config)
    else:
        return create_gan('wgan_gp', **config)

if __name__ == "__main__":
    # Ejemplo de uso
    print("🎨 Creando Modelos GAN Avanzados...")
    
    # WGAN-GP
    wgan = create_gan('wgan_gp', image_size=64, channels=3)
    print(f"WGAN-GP creado: {wgan}")
    
    # GAN Condicional
    cgan = create_gan('conditional', num_classes=10, image_size=32)
    print(f"GAN Condicional creado: {cgan}")
    
    # StyleGAN2
    stylegan = create_gan('stylegan2', image_size=128, channels=3)
    print(f"StyleGAN2 creado: {stylegan}")
    
    # CycleGAN
    cyclegan = create_gan('cycle', image_size=64, channels=3)
    print(f"CycleGAN creado: {cyclegan}")
    
    # GAN adaptativo
    data_chars = {
        'image_size': 64,
        'channels': 3,
        'complexity_score': 0.8,
        'num_classes': 5
    }
    adaptive_gan = create_adaptive_gan(data_chars)
    print(f"GAN adaptativo creado: {adaptive_gan}")