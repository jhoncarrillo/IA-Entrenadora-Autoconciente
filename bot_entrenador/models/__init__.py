"""
Módulo de Modelos Avanzados para el Ecosistema Autónomo
Contiene implementaciones de redes neuronales de última generación
"""

# Modelos básicos
from .cnn_model import create_cnn
from .lstm_model import create_lstm
from .rnn_model import create_rnn

# Modelos avanzados
from .transformer_model import (
    create_transformer, 
    create_adaptive_transformer,
    AdvancedTransformer,
    HybridTransformer,
    TransformerFactory
)

from .vae_model import (
    create_vae,
    create_adaptive_vae,
    AdvancedVAE,
    BetaVAE,
    ConditionalVAE,
    vae_loss_function
)

from .diffusion_model import (
    create_diffusion_model,
    create_adaptive_diffusion_model,
    DDPM,
    ConditionalDDPM,
    diffusion_loss_function
)

from .gan_model import (
    create_gan,
    create_adaptive_gan,
    WGAN_GP,
    ConditionalGAN,
    CycleGAN,
    StyleGAN2Generator,
    adversarial_loss,
    cycle_consistency_loss
)

# Factory functions para creación automática
def create_model_by_type(model_type: str, **kwargs):
    """Factory function universal para crear cualquier tipo de modelo"""
    
    if model_type.lower() in ['cnn', 'convolutional']:
        return create_cnn(**kwargs)
    
    elif model_type.lower() in ['lstm', 'long_short_term_memory']:
        return create_lstm(**kwargs)
    
    elif model_type.lower() in ['rnn', 'recurrent']:
        return create_rnn(**kwargs)
    
    elif model_type.lower() in ['transformer', 'bert', 'gpt', 't5']:
        return create_transformer(model_type.lower(), **kwargs)
    
    elif model_type.lower() in ['vae', 'variational_autoencoder', 'beta_vae', 'conditional_vae']:
        vae_type = 'tabular' if 'input_dim' in kwargs else 'image'
        return create_vae(vae_type, **kwargs)
    
    elif model_type.lower() in ['diffusion', 'ddpm', 'conditional_ddpm']:
        diff_type = 'conditional' if 'num_classes' in kwargs else 'ddpm'
        return create_diffusion_model(diff_type, **kwargs)
    
    elif model_type.lower() in ['gan', 'wgan', 'conditional_gan', 'cycle_gan', 'stylegan']:
        if 'cycle' in model_type.lower():
            gan_type = 'cycle'
        elif 'conditional' in model_type.lower():
            gan_type = 'conditional'
        elif 'style' in model_type.lower():
            gan_type = 'stylegan2'
        else:
            gan_type = 'wgan_gp'
        return create_gan(gan_type, **kwargs)
    
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

def create_adaptive_model(model_type: str, data_characteristics: dict):
    """Crear modelo adaptado automáticamente a las características de los datos"""
    
    if model_type.lower() in ['transformer', 'bert', 'gpt']:
        return create_adaptive_transformer(model_type.lower(), data_characteristics)
    
    elif model_type.lower() in ['vae', 'variational_autoencoder']:
        return create_adaptive_vae(data_characteristics)
    
    elif model_type.lower() in ['diffusion', 'ddpm']:
        return create_adaptive_diffusion_model(data_characteristics)
    
    elif model_type.lower() in ['gan', 'wgan']:
        return create_adaptive_gan(data_characteristics)
    
    else:
        # Para modelos básicos, usar configuración estándar
        return create_model_by_type(model_type, **data_characteristics)

# Diccionario de modelos disponibles
AVAILABLE_MODELS = {
    'basic': ['cnn', 'lstm', 'rnn'],
    'transformers': ['transformer', 'bert', 'gpt2', 't5', 'roberta'],
    'generative': ['vae', 'beta_vae', 'conditional_vae'],
    'diffusion': ['ddpm', 'conditional_ddpm'],
    'adversarial': ['wgan_gp', 'conditional_gan', 'cycle_gan', 'stylegan2']
}

def get_available_models():
    """Obtener lista de todos los modelos disponibles"""
    all_models = []
    for category, models in AVAILABLE_MODELS.items():
        all_models.extend(models)
    return all_models

def get_models_by_category(category: str):
    """Obtener modelos por categoría"""
    return AVAILABLE_MODELS.get(category, [])

def get_model_info(model_type: str):
    """Obtener información sobre un tipo de modelo específico"""
    
    model_info = {
        'cnn': {
            'description': 'Red Neuronal Convolucional para procesamiento de imágenes',
            'use_cases': ['clasificación de imágenes', 'detección de objetos', 'segmentación'],
            'data_types': ['image']
        },
        'lstm': {
            'description': 'Long Short-Term Memory para secuencias temporales',
            'use_cases': ['predicción de series temporales', 'procesamiento de texto', 'análisis secuencial'],
            'data_types': ['sequence', 'text', 'time_series']
        },
        'transformer': {
            'description': 'Arquitectura Transformer para tareas de NLP avanzadas',
            'use_cases': ['traducción', 'generación de texto', 'análisis de sentimientos'],
            'data_types': ['text', 'sequence']
        },
        'vae': {
            'description': 'Variational Autoencoder para generación y representación',
            'use_cases': ['generación de datos', 'reducción de dimensionalidad', 'detección de anomalías'],
            'data_types': ['tabular', 'image']
        },
        'ddpm': {
            'description': 'Denoising Diffusion Probabilistic Model para generación de alta calidad',
            'use_cases': ['generación de imágenes', 'síntesis de datos', 'inpainting'],
            'data_types': ['image']
        },
        'wgan_gp': {
            'description': 'Wasserstein GAN con Gradient Penalty para generación estable',
            'use_cases': ['generación de imágenes', 'aumento de datos', 'síntesis'],
            'data_types': ['image']
        }
    }
    
    return model_info.get(model_type.lower(), {
        'description': 'Modelo avanzado de red neuronal',
        'use_cases': ['tareas generales de aprendizaje automático'],
        'data_types': ['general']
    })

__all__ = [
    # Funciones de creación básicas
    'create_cnn', 'create_lstm', 'create_rnn',
    
    # Funciones de creación avanzadas
    'create_transformer', 'create_adaptive_transformer',
    'create_vae', 'create_adaptive_vae',
    'create_diffusion_model', 'create_adaptive_diffusion_model',
    'create_gan', 'create_adaptive_gan',
    
    # Clases principales
    'AdvancedTransformer', 'HybridTransformer',
    'AdvancedVAE', 'BetaVAE', 'ConditionalVAE',
    'DDPM', 'ConditionalDDPM',
    'WGAN_GP', 'ConditionalGAN', 'CycleGAN', 'StyleGAN2Generator',
    
    # Factory functions
    'create_model_by_type', 'create_adaptive_model',
    
    # Utilidades
    'get_available_models', 'get_models_by_category', 'get_model_info',
    'AVAILABLE_MODELS'
]
from .rnn_model import create_rnn
from .lstm_model import create_lstm
from .transformer_model import create_transformer