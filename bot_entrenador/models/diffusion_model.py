"""
Modelo de Difusión Avanzado para el Ecosistema Autónomo
Implementa Denoising Diffusion Probabilistic Models (DDPM) para generación de datos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

class SinusoidalPositionEmbeddings(nn.Module):
    """Embeddings posicionales sinusoidales para el timestep"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Bloque residual con normalización temporal"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Agregar embedding temporal
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """Bloque de atención para el modelo de difusión"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalización
        x_norm = self.norm(x)
        
        # Calcular Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape para atención multi-cabeza
        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)
        k = k.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)
        
        # Atención
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        
        # Aplicar atención
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        
        # Proyección de salida
        out = self.proj_out(out)
        
        return x + out

class UNet(nn.Module):
    """Red U-Net para el modelo de difusión"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 features: List[int] = [64, 128, 256, 512],
                 time_emb_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        
        prev_channels = in_channels
        for feature in features:
            self.encoder.append(nn.ModuleList([
                ResidualBlock(prev_channels, feature, time_emb_dim, dropout),
                ResidualBlock(feature, feature, time_emb_dim, dropout),
                AttentionBlock(feature, num_heads) if feature >= 256 else nn.Identity()
            ]))
            self.pool.append(nn.Conv2d(feature, feature, kernel_size=4, stride=2, padding=1))
            prev_channels = feature
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(features[-1], features[-1] * 2, time_emb_dim, dropout),
            AttentionBlock(features[-1] * 2, num_heads),
            ResidualBlock(features[-1] * 2, features[-1], time_emb_dim, dropout)
        ])
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        reversed_features = list(reversed(features))
        for i, feature in enumerate(reversed_features):
            if i == 0:
                in_ch = feature * 2  # Skip connection from encoder
            else:
                in_ch = reversed_features[i-1] + feature  # Skip connection
            
            self.upconv.append(
                nn.ConvTranspose2d(reversed_features[i-1] if i > 0 else feature,
                                 reversed_features[i-1] if i > 0 else feature,
                                 kernel_size=4, stride=2, padding=1)
            )
            
            self.decoder.append(nn.ModuleList([
                ResidualBlock(in_ch, feature, time_emb_dim, dropout),
                ResidualBlock(feature, feature, time_emb_dim, dropout),
                AttentionBlock(feature, num_heads) if feature >= 256 else nn.Identity()
            ]))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x, timestep):
        # Time embedding
        t_emb = self.time_mlp(timestep)
        
        # Encoder
        skip_connections = []
        for (res1, res2, attn), pool in zip(self.encoder, self.pool):
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        
        # Decoder
        skip_connections = list(reversed(skip_connections))
        for i, ((res1, res2, attn), upconv) in enumerate(zip(self.decoder, self.upconv)):
            x = upconv(x)
            
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[i]
                # Asegurar que las dimensiones coincidan
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x)
        
        return self.final_conv(x)

class DiffusionScheduler:
    """Scheduler para el proceso de difusión"""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001,
                 beta_end: float = 0.02, schedule_type: str = 'linear'):
        
        self.num_timesteps = num_timesteps
        
        # Definir schedule de betas
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Schedule type {schedule_type} not supported")
        
        # Calcular alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calcular valores útiles para sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Para reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """Cosine schedule como en Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        """Agregar ruido a las imágenes según el timestep"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape para broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.image_size = config.get('image_size', 64)
        self.in_channels = config.get('in_channels', 3)
        self.num_timesteps = config.get('num_timesteps', 1000)
        
        # Crear modelo de ruido (U-Net)
        self.noise_model = UNet(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            features=config.get('features', [64, 128, 256, 512]),
            time_emb_dim=config.get('time_emb_dim', 256),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1)
        )
        
        # Scheduler de difusión
        self.scheduler = DiffusionScheduler(
            num_timesteps=self.num_timesteps,
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02),
            schedule_type=config.get('schedule_type', 'linear')
        )
        
    def forward(self, x):
        """Forward pass durante entrenamiento"""
        batch_size = x.shape[0]
        device = x.device
        
        # Muestrear timesteps aleatorios
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Muestrear ruido
        noise = torch.randn_like(x)
        
        # Agregar ruido
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        
        # Predecir ruido
        predicted_noise = self.noise_model(x_noisy, timesteps)
        
        return {
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'timesteps': timesteps,
            'noisy_images': x_noisy
        }
    
    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device = None, 
               guidance_scale: float = 1.0) -> torch.Tensor:
        """Generar muestras usando el proceso de reverse diffusion"""
        
        if device is None:
            device = next(self.parameters()).device
        
        # Comenzar con ruido puro
        x = torch.randn(num_samples, self.in_channels, self.image_size, self.image_size, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predecir ruido
            predicted_noise = self.noise_model(x, timesteps)
            
            # Calcular x_{t-1}
            alpha_t = self.scheduler.alphas[t]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
            beta_t = self.scheduler.betas[t]
            
            # Fórmula de reverse process
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )
            
            # Agregar ruido si no es el último paso
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.scheduler.posterior_variance[t]
                x = x + torch.sqrt(variance) * noise
        
        return x
    
    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                   num_steps: int = 10, t: int = None) -> torch.Tensor:
        """Interpolar entre dos imágenes en el espacio latente de difusión"""
        
        if t is None:
            t = self.num_timesteps // 2
        
        device = x1.device
        
        # Agregar ruido a ambas imágenes
        noise1 = torch.randn_like(x1)
        noise2 = torch.randn_like(x2)
        timesteps = torch.full((1,), t, device=device, dtype=torch.long)
        
        x1_noisy = self.scheduler.add_noise(x1, noise1, timesteps)
        x2_noisy = self.scheduler.add_noise(x2, noise2, timesteps)
        
        # Interpolar en el espacio ruidoso
        alphas = torch.linspace(0, 1, num_steps, device=device)
        interpolations = []
        
        for alpha in alphas:
            x_interp = (1 - alpha) * x1_noisy + alpha * x2_noisy
            
            # Denoising desde t hasta 0
            for denoising_t in reversed(range(t)):
                timesteps_denoise = torch.full((1,), denoising_t, device=device, dtype=torch.long)
                predicted_noise = self.noise_model(x_interp, timesteps_denoise)
                
                alpha_t = self.scheduler.alphas[denoising_t]
                alpha_cumprod_t = self.scheduler.alphas_cumprod[denoising_t]
                beta_t = self.scheduler.betas[denoising_t]
                
                x_interp = (1 / torch.sqrt(alpha_t)) * (
                    x_interp - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
                )
                
                if denoising_t > 0:
                    noise = torch.randn_like(x_interp)
                    variance = self.scheduler.posterior_variance[denoising_t]
                    x_interp = x_interp + torch.sqrt(variance) * noise
            
            interpolations.append(x_interp)
        
        return torch.cat(interpolations, dim=0)

class ConditionalDDPM(DDPM):
    """DDPM Condicional para generación controlada"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_classes = config['num_classes']
        self.class_emb_dim = config.get('class_emb_dim', 256)
        
        # Embedding para clases
        self.class_embedding = nn.Embedding(self.num_classes, self.class_emb_dim)
        
        # Modificar U-Net para incluir condición de clase
        # Esto requeriría modificar la arquitectura del U-Net
        
    def forward(self, x, class_labels):
        """Forward pass con condición de clase"""
        batch_size = x.shape[0]
        device = x.device
        
        # Muestrear timesteps
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Embedding de clase
        class_emb = self.class_embedding(class_labels)
        
        # Muestrear ruido
        noise = torch.randn_like(x)
        
        # Agregar ruido
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        
        # Predecir ruido (necesitaría modificar U-Net para usar class_emb)
        predicted_noise = self.noise_model(x_noisy, timesteps)
        
        return {
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'timesteps': timesteps,
            'class_labels': class_labels
        }

def diffusion_loss_function(predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """Función de pérdida para modelos de difusión"""
    return F.mse_loss(predicted_noise, target_noise)

def create_diffusion_model(model_type: str = 'ddpm', **kwargs) -> DDPM:
    """Factory function para crear modelos de difusión"""
    
    default_config = {
        'image_size': 64,
        'in_channels': 3,
        'num_timesteps': 1000,
        'features': [64, 128, 256, 512],
        'time_emb_dim': 256,
        'num_heads': 4,
        'dropout': 0.1,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'schedule_type': 'linear'
    }
    
    # Actualizar configuración
    config = {**default_config, **kwargs}
    
    if model_type == 'ddpm':
        return DDPM(config)
    elif model_type == 'conditional':
        if 'num_classes' not in config:
            raise ValueError("num_classes required for conditional DDPM")
        return ConditionalDDPM(config)
    else:
        raise ValueError(f"Model type {model_type} not supported")

def create_adaptive_diffusion_model(data_characteristics: Dict[str, Any]) -> DDPM:
    """Crear modelo de difusión adaptado a las características de los datos"""
    
    image_size = data_characteristics.get('image_size', 64)
    channels = data_characteristics.get('channels', 3)
    complexity = data_characteristics.get('complexity_score', 0.5)
    
    # Configuración adaptativa
    if complexity < 0.3:
        config = {
            'features': [32, 64, 128, 256],
            'num_timesteps': 500,
            'time_emb_dim': 128
        }
    elif complexity < 0.7:
        config = {
            'features': [64, 128, 256, 512],
            'num_timesteps': 1000,
            'time_emb_dim': 256
        }
    else:
        config = {
            'features': [128, 256, 512, 1024],
            'num_timesteps': 1500,
            'time_emb_dim': 512
        }
    
    config.update({
        'image_size': image_size,
        'in_channels': channels
    })
    
    # Modelo condicional si hay clases
    if 'num_classes' in data_characteristics:
        config['num_classes'] = data_characteristics['num_classes']
        return create_diffusion_model('conditional', **config)
    else:
        return create_diffusion_model('ddpm', **config)

if __name__ == "__main__":
    # Ejemplo de uso
    print("🌊 Creando Modelo de Difusión Avanzado...")
    
    # DDPM básico
    ddpm = create_diffusion_model('ddpm', image_size=32, in_channels=1)
    print(f"DDPM creado: {ddpm}")
    
    # DDPM condicional
    conditional_ddpm = create_diffusion_model('conditional', num_classes=10, image_size=64)
    print(f"DDPM condicional creado: {conditional_ddpm}")
    
    # Modelo adaptativo
    data_chars = {
        'image_size': 128,
        'channels': 3,
        'complexity_score': 0.8,
        'num_classes': 5
    }
    adaptive_ddpm = create_adaptive_diffusion_model(data_chars)
    print(f"DDPM adaptativo creado: {adaptive_ddpm}")