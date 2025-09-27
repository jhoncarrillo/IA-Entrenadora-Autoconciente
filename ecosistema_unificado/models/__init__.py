"""
Módulo de Modelos Unificados del Ecosistema de Redes Neuronales
==============================================================

Este módulo contiene todas las implementaciones unificadas de modelos
de redes neuronales del ecosistema, combinando funcionalidades de
project_root y red_neuronal.

Modelos disponibles:
- CNN: Redes Neuronales Convolucionales
- RNN: Redes Neuronales Recurrentes (LSTM, GRU)
- GAN: Redes Generativas Adversarias
- VAE: Autoencoders Variacionales
- Transformer: Modelos de Transformers
- Attention: Mecanismos de Atención
- Graph: Redes Neuronales de Grafos

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

# Importaciones de modelos unificados
from .cnn_unified import CNNUnified
from .rnn_unified import RNNUnified
from .gan_unified import GANUnified
from .transformer_unified import TransformerUnified
from .vae_unified import VAEUnified
from .attention_unified import AttentionUnified
from .graph_unified import GraphUnified

__all__ = [
    'CNNUnified',
    'RNNUnified', 
    'GANUnified',
    'TransformerUnified',
    'VAEUnified',
    'AttentionUnified',
    'GraphUnified'
]